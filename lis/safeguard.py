r"""Safeguarding a learned solver so it can never be worse than Newton.

Notebooks 02–03 built a completely frozen solver — one preconditioner, one coefficient
vector, no runtime adaptation — and validated it on held-out contingencies drawn from the
same distribution it was trained on.  That is interpolation.  Nothing in it guarantees
anything once the operating point leaves that distribution, and a contingency screening
tool that silently returns a wrong answer is worse than a slow one.

The remedy is the oldest trick in numerical software: **check, and fall back**.  The
learned step is a proposal; accept it only if it actually reduces the residual, and
otherwise take the Newton step we would have taken anyway.  Since the true mismatch
:math:`\mathbf{f}(x)` is evaluated every iteration regardless — it is what tells us we have
converged — the test costs one extra residual evaluation, which is a few percent of a
Newton iteration.

The resulting guarantee is worth stating precisely:

* every accepted step strictly reduces :math:`\lVert\mathbf{f}\rVert_\infty`;
* in the worst case every step falls back and the method **is** Newton, up to the wasted
  residual evaluations;
* so the safeguarded solver converges wherever Newton does.

This is the same instinct as the softmax in [4]'s Runge-Kutta network, which makes every
member of the family consistent by construction: put the guarantee in the algorithm, not
in the training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from lis.jacobian_free import PolarResidual, fd_matvec
from lis.krylov import apply_poly
from lis.powerflow import Case, build_jacobian

__all__ = ["SafeguardResult", "safeguarded_solve"]


@dataclass
class SafeguardResult:
    """Outcome of a safeguarded solve, with the fallback pattern recorded."""

    converged: bool
    n_iter: int
    norm_history: list[float]
    accepted: list[bool] = field(default_factory=list)   # True = learned step taken

    @property
    def n_fallback(self) -> int:
        return sum(1 for a in self.accepted if not a)

    @property
    def fallback_rate(self) -> float:
        return self.n_fallback / max(1, len(self.accepted))

    def cost(self, t_residual: float, t_precond: float, t_assembly: float,
             t_lu: float, m: int) -> float:
        r"""Wall-clock estimate.

        An accepted iteration costs :math:`(m+2)` residual evaluations (one for
        :math:`\mathbf{f}`, :math:`m` for the finite-difference matvecs, one for the
        safeguard test) plus :math:`m` preconditioner applications.  A fallback iteration
        additionally pays a Jacobian assembly and an LU solve — it has already spent the
        learned attempt before rejecting it.
        """
        total = 0.0
        for acc in self.accepted:
            total += (m + 2) * t_residual + m * t_precond
            if not acc:
                total += t_assembly + t_lu
        return total

    def __repr__(self) -> str:
        s = "converged" if self.converged else "DIVERGED"
        return (f"SafeguardResult({s} in {self.n_iter} iters, "
                f"{self.n_fallback}/{len(self.accepted)} fallbacks)")


def safeguarded_solve(case: Case, alpha: np.ndarray, precond, tol: float = 1e-8,
                      max_iter: int = 50, use_safeguard: bool = True,
                      require_decrease: float = 1.0) -> SafeguardResult:
    r"""Matrix-free Newton with a frozen polynomial step and a Newton fallback.

    Parameters
    ----------
    alpha
        The frozen polynomial coefficients fitted in notebook 02.
    precond
        Callable applying the reused preconditioner :math:`\mathbf{M}^{-1}`.
    use_safeguard
        Set ``False`` to run the unguarded solver of notebook 03, for comparison.
    require_decrease
        Accept the learned step only if it multiplies the residual norm by less than this.
        ``1.0`` demands any strict decrease; a value below 1 demands real progress and
        falls back more eagerly.

    Notes
    -----
    The fallback assembles :math:`\mathbf{J}` and factorises it, which is exactly the cost
    the matrix-free method exists to avoid.  The fallback *rate* is therefore the quantity
    that decides whether the learned solver is still worth using at a given operating
    point, and it is reported in :class:`SafeguardResult`.
    """
    f = PolarResidual.from_case(case, case.v0)
    x = f.pack(case.v0)
    m = len(alpha)

    fx = f(x)
    norms = [float(np.linalg.norm(fx, np.inf))]
    accepted: list[bool] = []

    for _ in range(max_iter):
        if norms[-1] < tol:
            return SafeguardResult(True, len(accepted), norms, accepted)
        if not np.isfinite(norms[-1]) or norms[-1] > 1e8:
            return SafeguardResult(False, len(accepted), norms, accepted)

        mv = lambda v: precond(fd_matvec(f, x, v, fx))       # noqa: E731
        dx = apply_poly(mv, precond(-fx), alpha)

        x_try = x + dx
        f_try = f(x_try)
        n_try = float(np.linalg.norm(f_try, np.inf))

        take_learned = np.isfinite(n_try) and n_try < require_decrease * norms[-1]
        if take_learned or not use_safeguard:
            x, fx = x_try, f_try
            norms.append(n_try if np.isfinite(n_try) else np.inf)
            accepted.append(True)
        else:
            # Fall back to the Newton step we would otherwise have taken.
            va, vm = f.unpack(x)
            v = vm * np.exp(1j * va)
            j = build_jacobian(v, case.ybus, case.pv, case.pq)
            dx_n = spla.spsolve(sp.csc_matrix(j), -fx)
            x = x + dx_n
            fx = f(x)
            norms.append(float(np.linalg.norm(fx, np.inf)))
            accepted.append(False)

    return SafeguardResult(norms[-1] < tol, len(accepted), norms, accepted)
