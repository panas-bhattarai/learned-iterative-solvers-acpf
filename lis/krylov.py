r"""Arnoldi and GMRES, written out explicitly.

Newton's method spends all of its work in one place: solving the linear system

.. math::
    \mathbf{J}\,\Delta x = -\mathbf{f}

Sparse LU does this by *factorizing* :math:`\mathbf{J}`.  Krylov methods never form or
factorize anything -- they only ever multiply by :math:`\mathbf{J}`.  Given a starting
residual :math:`\mathbf{r}_0 = \mathbf{b} - \mathbf{A}\mathbf{x}_0`, they build the
**Krylov subspace**

.. math::
    \mathcal{K}_m(\mathbf{A}, \mathbf{r}_0)
        = \operatorname{span}\{\mathbf{r}_0, \mathbf{A}\mathbf{r}_0,
          \mathbf{A}^2\mathbf{r}_0, \dots, \mathbf{A}^{m-1}\mathbf{r}_0\}

and look for a correction inside it.  GMRES picks the member of that subspace which
minimizes the residual norm [3, Ch. 6]:

.. math::
    \mathbf{x}_m = \arg\min_{\mathbf{x} \in \mathbf{x}_0 + \mathcal{K}_m}
                   \lVert \mathbf{b} - \mathbf{A}\mathbf{x} \rVert_2

Read that once more, because it is the hinge of this whole project.  GMRES splits into
two distinct jobs:

1. **generate information** -- build a basis of :math:`\mathcal{K}_m` by repeated
   matrix-vector products (the Arnoldi process);
2. **assemble a solution** -- take a *linear combination* of those basis vectors.

Step 2 is a small least-squares problem, re-solved from scratch at runtime for every
single system you hand it.  The R2N2 architecture [5] keeps step 1 and replaces step 2's
runtime optimization with weights *learned offline* from a family of related problems.
The same two-module split -- generate, then assemble -- is stated explicitly in the
abstract of [5], and is why a Runge-Kutta integrator [4] (stages, then a weighted sum)
turns out to be the same kind of object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["arnoldi", "gmres", "GmresResult", "ilu_preconditioner", "lu_solve"]


# --------------------------------------------------------------------------------------
# Arnoldi
# --------------------------------------------------------------------------------------

def arnoldi(a_mv, r0: np.ndarray, m: int, breakdown_tol: float = 1e-14
            ) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Arnoldi process: an orthonormal basis of :math:`\mathcal{K}_m(\mathbf{A}, r_0)`.

    Builds :math:`\mathbf{Q}_{m+1} = [q_1, \dots, q_{m+1}]` with orthonormal columns and
    an upper Hessenberg :math:`\tilde{\mathbf{H}}_m \in \mathbb{R}^{(m+1)\times m}`
    satisfying the *Arnoldi relation* [3, Sec. 6.3]

    .. math::
        \mathbf{A}\mathbf{Q}_m = \mathbf{Q}_{m+1}\tilde{\mathbf{H}}_m

    Each step is one matrix-vector product followed by modified Gram-Schmidt against all
    previous basis vectors.  The cost per step therefore *grows*: step :math:`j` needs
    :math:`j` inner products.  This is why full GMRES is normally restarted, and why the
    small fixed :math:`m` used throughout this project is the interesting regime.

    Parameters
    ----------
    a_mv
        Either a matrix or a callable performing :math:`v \mapsto \mathbf{A}v`.  Krylov
        methods need nothing else -- no entries, no factorization.  This is precisely
        what makes them "matrix-free".
    r0
        Starting vector, normally the initial residual.
    m
        Maximum subspace dimension.

    Returns
    -------
    q, h, k
        ``q`` of shape ``(n, k+1)``, ``h`` of shape ``(k+1, k)``, and the dimension ``k``
        actually reached.  ``k < m`` signals *lucky breakdown*: the Krylov subspace
        became invariant, and the exact solution already lies inside it.
    """
    mv = a_mv if callable(a_mv) else (lambda v: a_mv @ v)

    n = r0.shape[0]
    q = np.zeros((n, m + 1), dtype=float)
    h = np.zeros((m + 1, m), dtype=float)

    beta = np.linalg.norm(r0)
    if beta == 0.0:
        return q[:, :1], h[:1, :0], 0
    q[:, 0] = r0 / beta

    for j in range(m):
        w = mv(q[:, j])
        for i in range(j + 1):                    # modified Gram-Schmidt
            h[i, j] = q[:, i] @ w
            w = w - h[i, j] * q[:, i]
        h[j + 1, j] = np.linalg.norm(w)
        if h[j + 1, j] < breakdown_tol:           # lucky breakdown
            return q[:, :j + 2], h[:j + 2, :j + 1], j + 1
        q[:, j + 1] = w / h[j + 1, j]

    return q, h, m


# --------------------------------------------------------------------------------------
# GMRES
# --------------------------------------------------------------------------------------

@dataclass
class GmresResult:
    """Solution plus the full residual-vs-subspace-dimension curve."""

    x: np.ndarray
    residuals: np.ndarray          # ||b - A x_j||_2 for j = 0, 1, ..., k
    dim: int                       # Krylov dimension actually built
    y: np.ndarray = field(repr=False, default=None)   # coefficients at dimension k
    q: np.ndarray = field(repr=False, default=None)   # Krylov basis
    h: np.ndarray = field(repr=False, default=None)   # Hessenberg matrix
    breakdown: bool = False

    @property
    def relative_residuals(self) -> np.ndarray:
        return self.residuals / self.residuals[0]


def gmres(a_mv, b: np.ndarray, m: int, x0: np.ndarray | None = None,
          precond=None) -> GmresResult:
    r"""GMRES, returning the residual at *every* subspace dimension.

    At each dimension :math:`j` the Arnoldi relation turns the :math:`n`-dimensional
    minimization into a tiny :math:`(j+1) \times j` least-squares problem:

    .. math::
        \lVert \mathbf{b} - \mathbf{A}(\mathbf{x}_0 + \mathbf{Q}_j \mathbf{y}) \rVert_2
        = \lVert \beta \mathbf{e}_1 - \tilde{\mathbf{H}}_j \mathbf{y} \rVert_2,
        \qquad \beta = \lVert \mathbf{r}_0 \rVert_2

    so the update is :math:`\Delta x = \mathbf{Q}_j \mathbf{y}_j` -- a linear combination
    of the Krylov basis vectors with coefficients :math:`\mathbf{y}_j` obtained by
    solving that least-squares problem *at runtime*.

    Those coefficients :math:`\mathbf{y}` are the objects R2N2 [5] learns instead of
    computes.  Everything in this project is a variation on where they come from.

    Parameters
    ----------
    precond
        Optional left preconditioner :math:`\mathbf{M}^{-1}`, given as a callable
        ``v -> M^{-1} v``.  GMRES is then applied to
        :math:`\mathbf{M}^{-1}\mathbf{A}\mathbf{x} = \mathbf{M}^{-1}\mathbf{b}`, so the
        reported residuals are *preconditioned* residuals.

    Notes
    -----
    Because ``residuals`` is read directly off the least-squares problem at each
    dimension, this returns the entire convergence curve for the price of one solve --
    which is exactly the curve we want to look at.
    """
    mv_raw = a_mv if callable(a_mv) else (lambda v: a_mv @ v)
    mv = mv_raw if precond is None else (lambda v: precond(mv_raw(v)))
    rhs = b if precond is None else precond(b)

    n = rhs.shape[0]
    x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
    r0 = rhs - mv(x0)
    beta = float(np.linalg.norm(r0))

    if beta == 0.0:
        return GmresResult(x=x0, residuals=np.array([0.0]), dim=0)

    q, h, k = arnoldi(mv, r0, m)
    breakdown = k < m

    # Residual at every dimension j: min || beta e1 - H_j y ||.
    residuals = np.empty(k + 1)
    residuals[0] = beta
    y = np.zeros(0)
    for j in range(1, k + 1):
        e1 = np.zeros(j + 1)
        e1[0] = beta
        y, res, *_ = np.linalg.lstsq(h[:j + 1, :j], e1, rcond=None)
        residuals[j] = float(np.linalg.norm(e1 - h[:j + 1, :j] @ y))

    x = x0 + q[:, :k] @ y if k > 0 else x0
    return GmresResult(x=x, residuals=residuals, dim=k, y=y, q=q, h=h,
                       breakdown=breakdown)


# --------------------------------------------------------------------------------------
# Preconditioning and the direct reference
# --------------------------------------------------------------------------------------

def krylov_basis(a_mv, r0: np.ndarray, m: int) -> np.ndarray:
    r"""The raw (un-orthogonalized) Krylov basis
    :math:`\mathbf{K}_m = [r_0,\ \mathbf{A}r_0,\ \dots,\ \mathbf{A}^{m-1}r_0]`.

    Costs :math:`m-1` matrix-vector products and nothing else: no Gram-Schmidt, no
    orthogonalization, no inner products.  Contrast :func:`arnoldi`, which spans the same
    space but pays :math:`O(m^2)` inner products to make the basis orthonormal.
    """
    mv = a_mv if callable(a_mv) else (lambda v: a_mv @ v)
    k = np.empty((r0.shape[0], m), dtype=float)
    k[:, 0] = r0
    for j in range(1, m):
        k[:, j] = mv(k[:, j - 1])
    return k


def optimal_poly_coeffs(a_mv, b: np.ndarray, m: int, rcond: float = 1e-12):
    r"""Best degree-:math:`m` polynomial iteration for a *single* system.

    Restrict the solution to the Krylov subspace written in the **monomial** basis:

    .. math::
        \Delta x = \sum_{j=0}^{m-1} \alpha_j \mathbf{A}^j \mathbf{b} = p(\mathbf{A})\,\mathbf{b}

    and choose :math:`\boldsymbol{\alpha}` to minimize the true residual
    :math:`\lVert \mathbf{b} - \mathbf{A}\,p(\mathbf{A})\mathbf{b} \rVert_2`.  Since
    :math:`\mathbf{A}p(\mathbf{A})\mathbf{b} = \mathbf{W}\boldsymbol{\alpha}` with
    :math:`\mathbf{W} = [\mathbf{A}\mathbf{b}, \dots, \mathbf{A}^m\mathbf{b}]`, this is a
    dense :math:`n \times m` least-squares problem.

    **Why this parametrization and not GMRES's.**  GMRES writes its update as
    :math:`\Delta x = \mathbf{Q}_m \mathbf{y}` in the *orthonormal Arnoldi basis*.  That
    basis is rebuilt for every system, so freezing :math:`\mathbf{y}` would still leave
    all the Gram-Schmidt work at runtime and save only the tiny least-squares.  The
    monomial coefficients :math:`\boldsymbol{\alpha}` are different: freeze them and you
    have a complete, self-contained algorithm -- :math:`m-1` matvecs and a weighted sum,
    with *no runtime optimization of any kind*.  That is the object worth asking whether
    one can learn, and it is the same "generate, then assemble" skeleton as R2N2 [5] and
    Runge-Kutta [4].

    Note also that :math:`\boldsymbol{\alpha}` is invariant to scaling of :math:`\mathbf{b}`
    -- if :math:`\mathbf{b} \to c\mathbf{b}` then :math:`\Delta x \to c\,\Delta x` with the
    same coefficients -- so coefficients from different instances are directly comparable
    without normalization.  They do, however, carry the units of
    :math:`\mathbf{A}^{-1}`, and so scale with the magnitude of :math:`\mathbf{A}`.

    Returns
    -------
    alpha, rel_residual, cond_w
        Coefficients, the achieved :math:`\lVert r_m\rVert/\lVert r_0\rVert`, and the
        condition number of :math:`\mathbf{W}`.  **Watch ``cond_w``**: powers
        :math:`\mathbf{A}^j\mathbf{b}` align with the dominant eigenvector as :math:`j`
        grows, so the monomial basis degenerates quickly.  That numerical fragility is
        precisely why Arnoldi exists, and it bounds how large an :math:`m` this analysis
        can honestly use.
    """
    mv = a_mv if callable(a_mv) else (lambda v: a_mv @ v)
    w = np.empty((b.shape[0], m), dtype=float)
    w[:, 0] = mv(b)
    for j in range(1, m):
        w[:, j] = mv(w[:, j - 1])

    alpha, *_ = np.linalg.lstsq(w, b, rcond=rcond)
    rel = float(np.linalg.norm(b - w @ alpha) / np.linalg.norm(b))
    return alpha, rel, float(np.linalg.cond(w))


def apply_poly(a_mv, b: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r"""Run a *fixed* polynomial iteration: :math:`\Delta x = \sum_j \alpha_j \mathbf{A}^j\mathbf{b}`.

    This is the whole algorithm.  Given ``alpha``, solving a system costs ``len(alpha)-1``
    matrix-vector products and a weighted sum -- no factorization, no orthogonalization,
    no least squares, no adaptivity of any kind.
    """
    mv = a_mv if callable(a_mv) else (lambda v: a_mv @ v)
    x = alpha[0] * b
    v = b
    for j in range(1, len(alpha)):
        v = mv(v)
        x = x + alpha[j] * v
    return x


def ilu_preconditioner(a: sp.spmatrix, drop_tol: float = 1e-4, fill_factor: float = 10.0):
    r"""Incomplete LU preconditioner :math:`\mathbf{M}^{-1} \approx \mathbf{A}^{-1}`.

    An *incomplete* LU factorization computes :math:`\mathbf{A} \approx \mathbf{L}\mathbf{U}`
    while discarding fill-in below ``drop_tol``, giving a cheap, sparse, approximate
    inverse.  GMRES applied to :math:`\mathbf{M}^{-1}\mathbf{A}` converges far faster
    because that operator's eigenvalues are clustered near 1 [3, Ch. 10].

    Worth noticing for later: an ILU is exactly "an approximate inverse learned from the
    matrix by a fixed hand-designed rule".  Replacing that rule with something fitted to
    a family of matrices is one of the most direct ways to learn a solver.
    """
    ilu = spla.spilu(sp.csc_matrix(a), drop_tol=drop_tol, fill_factor=fill_factor)
    return ilu.solve


def lu_solve(a: sp.spmatrix, b: np.ndarray) -> np.ndarray:
    """Direct sparse LU solve -- the baseline every learned method must be judged against."""
    return spla.spsolve(sp.csc_matrix(a), b)
