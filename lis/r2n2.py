r"""R2N2 — the recursively recurrent neural network superstructure of Doncevic, Mitsos,
Guo, Li, Dietrich, Dahmen & Kevrekidis [5].

Notebook 04 learned a Butcher tableau for *integrators*.  R2N2 is the same idea generalised
to *solvers*: one forward pass is one iteration of a numerical algorithm, with an inner
recurrence that generates a subspace by repeated function evaluations and an outer layer
that assembles the update from it ([5], eqs. 9–10):

.. math::
    v_0 = f(x_k), \qquad
    v_j = f\Big(x_k + h\sum_{l<j}\theta_{j,l}\,v_l\Big),\quad j=1,\dots,n-1

.. math::
    x_{k+1} = x_k + h\sum_{j=0}^{n-1}\theta_{n,j}\,v_j

The parameter count is :math:`n(n+1)/2`, exactly as for the Runge-Kutta network — the
strictly lower-triangular inner coefficients :math:`\theta_{j,l}` plus the :math:`n` outer
weights :math:`\theta_{n,j}`.

**Why this is the fix for notebook 02.**  Take a linear residual :math:`f(z)=\mathbf{A}z-b`.
Then the inner recurrence becomes

.. math::
    v_j = v_0 + h\,\mathbf{A}\sum_{l<j}\theta_{j,l}\,v_l

so :math:`\operatorname{span}\{v_0,\dots,v_{n-1}\}` is the Krylov subspace
:math:`\mathcal{K}_n(\mathbf{A}, v_0)` — the *same* space notebook 02 used.  What changes is
the **basis** that spans it.  Setting :math:`\theta_{j,l}=\delta_{l,j-1}` recovers the
monomial basis whose condition number exploded past :math:`n\approx5`; Arnoldi instead
orthonormalises at :math:`O(n^2)` inner products per step.  R2N2 does neither — it *learns*
the recurrence coefficients offline, and pays nothing at runtime.

Note what is absent, and deliberately so ([5], §3.2.1): there is no orthonormalisation and
no explicit residual minimisation.  A GMRES iteration and an R2N2 pass both need
:math:`n-1` matrix-vector products to build an :math:`n`-dimensional subspace, but R2N2
skips the Gram-Schmidt and the least-squares solve entirely.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["R2N2", "monomial_coefficients", "basis_condition"]

torch.set_default_dtype(torch.float64)


class R2N2(torch.nn.Module):
    r"""One iteration of a learned iterative algorithm.

    Parameters
    ----------
    n
        Number of function evaluations per outer iteration (the subspace dimension).
    h
        Fixed scaling on the layer computations ([5] §3.1).  For equation solving there is
        no natural step size, so we keep ``h = 1`` and let the coefficients absorb it.
    init
        ``"monomial"`` starts from the fixed recurrence of notebook 02 — a deliberately
        informative initialisation, since it lets us ask whether training *moves away* from
        the basis we already know fails.  ``"randn"`` starts from noise.
    """

    def __init__(self, n: int, h: float = 1.0, init: str = "monomial",
                 scale: float = 0.1, seed: int | None = None):
        super().__init__()
        self.n, self.h = n, h
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        n_inner = n * (n - 1) // 2

        if init == "monomial":
            theta = torch.zeros(n_inner)
            idx = {(j, l): k for k, (j, l) in
                   enumerate((j, l) for j in range(1, n) for l in range(j))}
            for j in range(1, n):
                theta[idx[(j, j - 1)]] = 1.0        # v_j built only from v_{j-1}
            theta = theta + scale * torch.randn(n_inner, generator=g)
        else:
            theta = scale * torch.randn(n_inner, generator=g)

        self.inner = torch.nn.Parameter(theta)
        self.outer = torch.nn.Parameter(scale * torch.randn(n, generator=g))

    def inner_matrix(self) -> torch.Tensor:
        """Inner coefficients as a strictly lower triangular ``n x n`` matrix."""
        a = torch.zeros(self.n, self.n, dtype=self.inner.dtype, device=self.inner.device)
        idx = torch.tril_indices(self.n, self.n, offset=-1)
        return a.index_put((idx[0], idx[1]), self.inner)

    def basis(self, f, x: torch.Tensor) -> list[torch.Tensor]:
        r"""Generate the subspace :math:`\{v_0,\dots,v_{n-1}\}` — the *inner* recurrence.

        ``f`` maps a batch of iterates to a batch of residuals.  Costs ``n`` function
        evaluations, of which the first is the residual at the current iterate and would
        have been computed anyway.
        """
        theta = self.inner_matrix()
        vs = [f(x)]
        for j in range(1, self.n):
            step = sum(theta[j, l] * vs[l] for l in range(j))
            vs.append(f(x + self.h * step))
        return vs

    def step(self, f, x: torch.Tensor) -> torch.Tensor:
        """One outer iteration: generate the subspace, then assemble the update."""
        vs = self.basis(f, x)
        upd = sum(self.outer[j] * vs[j] for j in range(self.n))
        return x + self.h * upd

    def solve(self, f, x0: torch.Tensor, n_outer: int) -> list[torch.Tensor]:
        """Apply ``n_outer`` outer iterations, returning every iterate."""
        xs = [x0]
        for _ in range(n_outer):
            xs.append(self.step(f, xs[-1]))
        return xs

    def tableau(self) -> dict:
        return {"inner": self.inner_matrix().detach().cpu().numpy(),
                "outer": self.outer.detach().cpu().numpy()}


def monomial_coefficients(n: int) -> torch.Tensor:
    r"""The inner coefficients that reproduce notebook 02's monomial basis.

    :math:`\theta_{j,l} = \delta_{l,j-1}`, i.e. each new vector is built from the previous
    one alone.  Provided so that "the fixed basis we already measured failing" is an actual
    point in the same parameter space, not a separate implementation.
    """
    theta = torch.zeros(n * (n - 1) // 2)
    k = 0
    for j in range(1, n):
        for l in range(j):
            if l == j - 1:
                theta[k] = 1.0
            k += 1
    return theta


def basis_condition(vs: list[torch.Tensor]) -> np.ndarray:
    r"""Condition number of the generated basis, per batch element.

    Stacks :math:`[v_0,\dots,v_{n-1}]` columnwise and returns :math:`\kappa_2`.  This is
    the diagnostic notebook 02 ended on: the monomial basis reached
    :math:`\kappa_2>10^{14}` by :math:`n\approx6`, at which point its coefficients carried
    no information.  Columns are normalised first, so this measures genuine near-parallelism
    of the directions rather than a spread of magnitudes that any solver would rescale away.
    """
    v = torch.stack(vs, dim=-1)                       # (batch, m, n)
    v = v / (v.norm(dim=-2, keepdim=True) + 1e-300)
    out = []
    for mat in v.detach().cpu().numpy():
        if not np.all(np.isfinite(mat)):
            out.append(np.inf); continue
        try:
            out.append(np.linalg.cond(mat))
        except np.linalg.LinAlgError:     # SVD fails on a diverged basis; that is a result
            out.append(np.inf)
    return np.array(out)
