r"""The Runge-Kutta neural network of Guo, Dietrich, Bertalan, Doncevic, Dahmen,
Kevrekidis & Li [4] — a learned Butcher tableau.

Notebook 02 measured the monomial Krylov basis dying past :math:`m\approx5`
(:math:`\kappa_2(\mathbf{W})>10^{14}`), because the *inner* recurrence that generates the
subspace was fixed to raw powers :math:`\mathbf{A}^j\mathbf{b}`.  The fix that [4] and [5]
are built around is to let the inner recurrence carry its **own learnable coefficients**.
This module is the smallest instance of that idea.

An explicit :math:`m`-stage Runge-Kutta step is

.. math::
    k_1 = h\,f(y_n), \qquad
    k_i = h\,f\!\Big(y_n + \sum_{j<i}\theta_{i-1,j}\,k_j\Big), \qquad
    y_{n+1} = y_n + \sum_{i=1}^m \theta_{c_i} k_i

Classically the :math:`\theta` are derived by hand from order conditions.  Here they are
*trained* on a distribution of ODEs.  Note the two roles, which are exactly the split we
have been circling since notebook 01:

* :math:`\theta_{i,j}` — **inner** coefficients, deciding how each stage is built from the
  previous ones.  This is the piece the monomial basis lacked.
* :math:`\theta_{c_i}` — **outer** weights, assembling the stages into the update.  A
  softmax parametrization forces :math:`\sum_i\theta_{c_i}=1`, which makes every network in
  the family *consistent by construction* ([4], Prop. 3.1) — no amount of bad training can
  produce an integrator that fails to converge as :math:`h\to0`.

Everything runs in float64: the experiments resolve errors down to :math:`10^{-11}`, and
float32 would bottom out four orders above that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["RKNN", "TaskFamily", "LINEAR_FAMILY", "SQUARE_FAMILY",
           "rk_reference", "global_error", "taylor_regulariser"]

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------------------
# Task families
# --------------------------------------------------------------------------------------

@dataclass
class TaskFamily:
    r"""A distribution over ODE problem instances :math:`F = (f, y_0)`.

    Following [4] §4.1, each family fixes the *form* of the vector field and randomises
    its parameter and initial condition.  Both families used here have closed-form
    solutions, so training never needs a reference integrator and evaluation is exact.
    """

    name: str
    a_range: tuple[float, float]
    y0_range: tuple[float, float]

    def sample(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=gen)  # noqa: E731
        return u(*self.a_range), u(*self.y0_range)

    def f(self, y, a):
        raise NotImplementedError

    def exact(self, y0, a, t):
        raise NotImplementedError


class _Linear(TaskFamily):
    r""":math:`\dot y = -a y`, solution :math:`y(t) = y_0 e^{-at}`."""

    def f(self, y, a):
        return -a * y

    def exact(self, y0, a, t):
        return y0 * torch.exp(-a * t)


class _Square(TaskFamily):
    r""":math:`\dot y = -a y^2`, solution :math:`y(t) = (at + 1/y_0)^{-1}`.

    This is the family for which [4] shows a *two*-stage method can reach third order —
    impossible for general Lipschitz vector fields, where an explicit :math:`m`-stage
    method has order at most :math:`m`.
    """

    def f(self, y, a):
        return -a * y**2

    def exact(self, y0, a, t):
        return 1.0 / (a * t + 1.0 / y0)


LINEAR_FAMILY = _Linear("linear", a_range=(1.0, 5.0), y0_range=(-5.0, 5.0))
SQUARE_FAMILY = _Square("square", a_range=(0.1, 0.5), y0_range=(1.0, 3.0))


# --------------------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------------------

class RKNN(torch.nn.Module):
    r"""An :math:`m`-stage explicit Runge-Kutta step with trainable coefficients.

    Trainable parameters number :math:`m(m+1)/2`: the strictly-lower-triangular inner
    coefficients :math:`\theta_{i,j}` plus :math:`m` logits whose softmax gives the outer
    weights.  For :math:`m=3` that is six numbers — the entire "neural network".
    """

    def __init__(self, m: int, init_scale: float = 0.5, seed: int | None = None):
        super().__init__()
        self.m = m
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        n_inner = m * (m - 1) // 2
        self.inner = torch.nn.Parameter(
            init_scale * torch.randn(n_inner, generator=g) if n_inner else torch.zeros(0))
        self.logits = torch.nn.Parameter(torch.zeros(m))

    @property
    def weights(self) -> torch.Tensor:
        r"""Outer weights :math:`\theta_c`, softmaxed so :math:`\sum_i\theta_{c_i}=1`."""
        return torch.softmax(self.logits, dim=0)

    def inner_matrix(self) -> torch.Tensor:
        r"""Inner coefficients as a strictly lower triangular :math:`m\times m` matrix."""
        a = torch.zeros(self.m, self.m, dtype=self.inner.dtype)
        idx = torch.tril_indices(self.m, self.m, offset=-1)
        a = a.index_put((idx[0], idx[1]), self.inner)
        return a

    def step(self, family: TaskFamily, y, a, h):
        r"""One integration step :math:`y_n \mapsto y_{n+1}`."""
        theta = self.inner_matrix()
        w = self.weights
        ks = []
        for i in range(self.m):
            arg = y if i == 0 else y + sum(theta[i, j] * ks[j] for j in range(i))
            ks.append(h * family.f(arg, a))
        return y + sum(w[i] * ks[i] for i in range(self.m))

    def integrate(self, family: TaskFamily, y0, a, h, n_steps: int):
        """Take ``n_steps`` steps of size ``h``."""
        y = y0
        for _ in range(n_steps):
            y = self.step(family, y, a, h)
        return y

    def tableau(self) -> dict:
        """The learned coefficients, as plain numbers."""
        return {"inner": self.inner_matrix().detach().numpy(),
                "weights": self.weights.detach().numpy()}


# --------------------------------------------------------------------------------------
# Classical references
# --------------------------------------------------------------------------------------

def rk_reference(order: int) -> RKNN:
    r"""A classical explicit RK method, expressed in the same parametrization.

    ``order=2`` is Heun's method (:math:`\theta_1=1`, :math:`\theta_c=(\tfrac12,\tfrac12)`),
    the RK2 that [4] compares against; ``order=3`` is Kutta's third-order rule
    (:math:`\theta_c=(\tfrac16,\tfrac46,\tfrac16)`); ``order=4`` is the classical RK4.
    Encoding them in the same class guarantees the comparison differs only in the
    coefficients, never in the code path.
    """
    tables = {
        2: ([[0, 0], [1, 0]], [0.5, 0.5]),
        3: ([[0, 0, 0], [0.5, 0, 0], [-1.0, 2.0, 0]], [1/6, 4/6, 1/6]),
        4: ([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1.0, 0]],
            [1/6, 2/6, 2/6, 1/6]),
    }
    if order not in tables:
        raise ValueError(f"no reference tableau for order {order}")
    inner, w = tables[order]
    m = len(w)
    net = RKNN(m)
    idx = torch.tril_indices(m, m, offset=-1)
    with torch.no_grad():
        net.inner.copy_(torch.tensor([inner[i][j] for i, j in zip(*idx.tolist())]))
        net.logits.copy_(torch.log(torch.tensor(w)))     # softmax(log w) == w
    return net


# --------------------------------------------------------------------------------------
# Loss ingredients
# --------------------------------------------------------------------------------------

def taylor_regulariser(net: RKNN, family: TaskFamily, y0, a, alpha: int) -> torch.Tensor:
    r"""[4] eq. (16): penalise the low-order :math:`h`-derivatives of the one-step error.

    A global order of :math:`\alpha` requires a local truncation error of
    :math:`O(h^{\alpha+1})`, i.e.

    .. math::
        \frac{d^i}{dh^i}\Big|_{h=0}\big(y_1 - \hat y_1\big) = 0, \quad i=1,\dots,\alpha

    so the sum of their squared norms is a scalar that is zero exactly when the method
    attains the desired order.  The derivatives are taken **at exactly** :math:`h=0` by
    repeated automatic differentiation, which is why this measures order rather than
    merely accuracy at the step sizes that happened to be sampled.

    Without this term the plain squared loss is dominated by the largest :math:`h` in the
    batch, and the resulting integrator degrades sharply outside the sampled range —
    reproduced in the notebook.

    Note the shape of ``h``.  It is a *vector*, one entry per batch element, so that
    ``h[k]`` influences only ``err[k]`` and ``grad(err.sum(), h)`` returns the genuine
    per-element derivatives.  Differentiating with respect to a single shared scalar
    instead returns the derivative of the *sum*, and re-broadcasting it multiplies the
    penalty by the batch size at every order — which makes the regulariser grow like
    :math:`n^\alpha` and drown the loss.
    """
    h = torch.zeros(torch.as_tensor(y0).shape, requires_grad=True)
    d = family.exact(y0, a, h) - net.step(family, y0, a, h)
    total = torch.zeros(())
    for _ in range(alpha):
        d = torch.autograd.grad(d.sum(), h, create_graph=True)[0]
        total = total + (d**2).mean()
    return total


def global_error(net: RKNN, family: TaskFamily, y0, a, h: float, t_end: float
                 ) -> torch.Tensor:
    r"""Global truncation error :math:`\lVert \hat y_n - y(t_{end})\rVert` at step size ``h``."""
    n_steps = max(1, int(round(t_end / h)))
    h_eff = t_end / n_steps
    with torch.no_grad():
        y_hat = net.integrate(family, y0, a, torch.tensor(h_eff), n_steps)
        y_true = family.exact(y0, a, torch.tensor(t_end))
    return (y_hat - y_true).abs()
