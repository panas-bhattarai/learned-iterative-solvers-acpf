r"""Jacobian-free matrix-vector products for AC power flow.

Notebook 02 ended with a frozen solver that matched Newton+LU's iteration count but not
its wall clock, because every extra Newton step paid for a Jacobian *assembly* — 116× the
cost of a single residual evaluation in our implementation.

The escape is that the frozen solver never needs :math:`\mathbf{J}` as a matrix.  It only
ever forms the product :math:`\mathbf{J}v`, which is a **directional derivative** and can
be taken straight from the residual:

.. math::
    \mathbf{J}(x)\,v \;=\; \lim_{\varepsilon\to 0}
        \frac{\mathbf{f}(x + \varepsilon v) - \mathbf{f}(x)}{\varepsilon}

This module provides two ways to evaluate that product without ever building
:math:`\mathbf{J}`:

* :func:`fd_matvec` — the classical finite difference, cheap but with an accuracy floor
  set by the trade-off between truncation and round-off error;
* :func:`cs_matvec` — the **complex-step** derivative, which is exact to machine precision
  because it never subtracts two nearly equal numbers.

The complex-step trick needs a residual that is real-analytic in the real unknowns and
free of ``conj``/``abs``.  The usual complex form
:math:`\mathbf{f} = \mathbf{V}\odot\overline{(\mathbf{Y}_{bus}\mathbf{V})}` is *not*
usable: conjugation is not complex-differentiable, so perturbing into the imaginary
direction gets conjugated away.  :func:`mismatch_polar` rewrites the same physics in the
purely real polar form

.. math::
    P_i = |V_i|\sum_k |V_k| \big(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}\big), \qquad
    Q_i = |V_i|\sum_k |V_k| \big(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}\big)

with :math:`\theta_{ik} = \theta_i - \theta_k`, which contains only real operations and
therefore extends to complex arguments correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from lis.powerflow import Case

__all__ = ["PolarResidual", "mismatch_polar", "fd_matvec", "cs_matvec", "fd_epsilon"]


# --------------------------------------------------------------------------------------
# A residual as a function of the real unknown vector
# --------------------------------------------------------------------------------------

@dataclass
class PolarResidual:
    r"""The power flow residual as a plain function :math:`\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^n`.

    The unknown vector matches the Newton ordering used everywhere else in this project,

    .. math:: x = \big[\;\boldsymbol{\theta}_{[pv,pq]},\;\; |\mathbf{V}|_{[pq]}\;\big]

    with the slack angle, slack magnitude and PV magnitudes held at their specified values.
    Evaluating ``F(x)`` costs one residual; nothing here ever forms a Jacobian.
    """

    case: Case
    va0: np.ndarray               # full angle vector, fixed entries at their setpoints
    vm0: np.ndarray               # full magnitude vector, likewise
    indptr: np.ndarray
    indices: np.ndarray
    g_data: np.ndarray
    b_data: np.ndarray
    src: np.ndarray               # row index of each nonzero, precomputed once
    pvpq_idx: np.ndarray
    sbus_p: np.ndarray
    sbus_q: np.ndarray

    @classmethod
    def from_case(cls, case: Case, v: np.ndarray | None = None) -> "PolarResidual":
        v = case.v0 if v is None else v
        y = sp.csr_matrix(case.ybus)
        pvpq = np.r_[case.pv, case.pq]
        return cls(case=case, va0=np.angle(v).copy(), vm0=np.abs(v).copy(),
                   indptr=y.indptr, indices=y.indices,
                   g_data=y.data.real.copy(), b_data=y.data.imag.copy(),
                   src=np.repeat(np.arange(y.shape[0]), np.diff(y.indptr)),
                   pvpq_idx=pvpq,
                   sbus_p=case.sbus.real[pvpq].copy(),
                   sbus_q=case.sbus.imag[case.pq].copy())

    @property
    def pvpq(self) -> np.ndarray:
        return self.pvpq_idx

    @property
    def n(self) -> int:
        return len(self.case.pv) + 2 * len(self.case.pq)

    def pack(self, v: np.ndarray) -> np.ndarray:
        """Complex voltages -> the real unknown vector :math:`x`."""
        return np.r_[np.angle(v)[self.pvpq], np.abs(v)[self.case.pq]]

    def unpack(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Real unknown vector -> full ``(angle, magnitude)`` arrays, preserving dtype.

        The dtype is taken from ``x`` so that a complex perturbation propagates through
        untouched -- this is what makes :func:`cs_matvec` work.
        """
        va = np.asarray(self.va0, dtype=x.dtype).copy()
        vm = np.asarray(self.vm0, dtype=x.dtype).copy()
        n_pv = len(self.case.pv)
        va[self.pvpq] = x[:len(self.pvpq)]
        vm[self.case.pq] = x[len(self.pvpq):]
        return va, vm

    def __call__(self, x: np.ndarray) -> np.ndarray:
        va, vm = self.unpack(np.asarray(x))
        theta = va[self.src] - va[self.indices]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        vm_k = vm[self.indices]

        p = vm * np.add.reduceat((self.g_data * cos_t + self.b_data * sin_t) * vm_k,
                                 self.indptr[:-1])
        q = vm * np.add.reduceat((self.g_data * sin_t - self.b_data * cos_t) * vm_k,
                                 self.indptr[:-1])
        return np.r_[p[self.pvpq_idx] - self.sbus_p, q[self.case.pq] - self.sbus_q]


def mismatch_polar(va, vm, indptr, indices, g_data, b_data, sbus, pv, pq):
    r"""Power mismatch in real polar form, dtype-preserving.

    Written with only real arithmetic on ``va``/``vm`` — no ``conj``, no ``abs`` — so that
    it is analytic and may be evaluated at complex arguments for the complex-step
    derivative.  The row sums are done with ``np.add.reduceat`` over the CSR row pointers,
    which is exactly a sparse matvec but works for any dtype.
    """
    rows_of = np.diff(indptr)
    src = np.repeat(np.arange(len(indptr) - 1), rows_of)     # row index per nonzero
    theta = va[src] - va[indices]
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    contrib_p = (g_data * cos_t + b_data * sin_t) * vm[indices]
    contrib_q = (g_data * sin_t - b_data * cos_t) * vm[indices]

    p = vm * np.add.reduceat(contrib_p, indptr[:-1])
    q = vm * np.add.reduceat(contrib_q, indptr[:-1])

    pvpq = np.r_[pv, pq]
    return np.r_[p[pvpq] - sbus.real[pvpq], q[pq] - sbus.imag[pq]]


# --------------------------------------------------------------------------------------
# Directional derivatives
# --------------------------------------------------------------------------------------

def fd_epsilon(x: np.ndarray, v: np.ndarray, rel: float | None = None) -> float:
    r"""Standard finite-difference step for a Jacobian-vector product.

    The total error in a forward difference is the sum of a truncation term
    :math:`O(\varepsilon\lVert\mathbf{f}''\rVert)` and a round-off term
    :math:`O(\epsilon_{mach}\lVert \mathbf{f}\rVert/\varepsilon)`.  Balancing the two puts
    the optimum near :math:`\varepsilon \sim \sqrt{\epsilon_{mach}}\approx 1.5\times10^{-8}`,
    scaled by the size of the iterate so it stays meaningful in the units of the problem
    (Brown & Saad; Pernice & Walker):

    .. math::
        \varepsilon = \sqrt{\epsilon_{mach}}\;\frac{1 + \lVert x\rVert_2}{\lVert v\rVert_2}
    """
    rel = np.sqrt(np.finfo(float).eps) if rel is None else rel
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return rel
    return rel * (1.0 + np.linalg.norm(x)) / nv


def fd_matvec(f, x: np.ndarray, v: np.ndarray, fx: np.ndarray | None = None,
              eps: float | None = None) -> np.ndarray:
    r"""Forward-difference approximation of :math:`\mathbf{J}(x)\,v`.

    Costs **one** residual evaluation when ``fx`` is supplied (it is constant across all
    matvecs at a given Newton iterate, so it should be).  No Jacobian is formed.
    """
    fx = f(x) if fx is None else fx
    eps = fd_epsilon(x, v) if eps is None else eps
    return (f(x + eps * v) - fx) / eps


def cs_matvec(f, x: np.ndarray, v: np.ndarray, h: float = 1e-20) -> np.ndarray:
    r"""Complex-step approximation of :math:`\mathbf{J}(x)\,v` — exact to machine precision.

    For a real-analytic :math:`\mathbf{f}`, expanding along an imaginary perturbation gives

    .. math::
        \mathbf{f}(x + i h v) = \mathbf{f}(x) + i h\,\mathbf{J}(x)v
            - \tfrac{h^2}{2}\mathbf{f}''(x)[v,v] + O(h^3)

    so :math:`\mathbf{J}(x)v = \Im\{\mathbf{f}(x + ihv)\}/h + O(h^2)`.  The crucial point
    is that **no subtraction of nearly equal numbers occurs** — the derivative is read off
    the imaginary part directly.  Round-off therefore does not blow up as :math:`h\to 0`,
    and one may simply take :math:`h = 10^{-20}`, making the truncation term
    :math:`O(h^2)=10^{-40}` utterly negligible.

    The cost is complex arithmetic throughout: roughly 2–4× a real residual evaluation.
    Requires ``f`` to be free of ``conj``, ``abs``, and comparison-based branching --
    see :func:`mismatch_polar`.
    """
    return np.imag(f(np.asarray(x, dtype=complex) + 1j * h * v)) / h
