"""AC power flow: admittance matrix, mismatch, Jacobian, Newton-Raphson.

Everything here is written from scratch in NumPy/SciPy, following the polar-form
formulation used by MATPOWER's ``makeYbus.m`` and ``newtonpf.m``.  It is deliberately
*not* written in PyTorch: notebooks 01-02 are about classical numerics, and autograd
machinery would only get in the way of seeing what the solver does.  A differentiable
PyTorch version arrives later, when we actually need to backpropagate through a solver.

Case data is obtained from ``pandapower``, but only as a *source of MATPOWER matrices*.
After ``pp.runpp(net)``, ``net["_ppc"]`` holds the familiar tables::

    bus     [BUS_I, BUS_TYPE, PD, QD, GS, BS, ..., VM, VA, BASE_KV, ...]
    branch  [F_BUS, T_BUS, BR_R, BR_X, BR_B, ..., TAP, SHIFT, BR_STATUS, ...]
    gen     [GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...]

with bus types 1 = PQ, 2 = PV, 3 = slack (``REF``), exactly as in MATPOWER.
``net["_ppc"]["internal"]`` additionally holds pandapower's own ``Ybus``, ``J`` and
converged ``V``, which we use purely as independent references for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# MATPOWER column indices, imported rather than hard-coded so they cannot drift.
from pandapower.pypower.idx_bus import BUS_TYPE, PD, QD, GS, BS, VM, VA, REF, PV, PQ
from pandapower.pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_G, TAP, SHIFT, BR_STATUS,
    BR_R_ASYM, BR_X_ASYM, BR_G_ASYM, BR_B_ASYM,
)
from pandapower.pypower.idx_gen import GEN_BUS, PG, QG, VG, GEN_STATUS

__all__ = [
    "Case", "load_case", "build_ybus", "make_sbus", "bus_types", "flat_start",
    "mismatch", "build_jacobian", "newton_pf", "NewtonResult",
]


# --------------------------------------------------------------------------------------
# Case container
# --------------------------------------------------------------------------------------

@dataclass
class Case:
    """A power flow problem in MATPOWER form, plus everything derived from it."""

    name: str
    base_mva: float
    bus: np.ndarray
    branch: np.ndarray
    gen: np.ndarray

    ybus: sp.csr_matrix          # our own, built by build_ybus
    sbus: np.ndarray             # complex scheduled injections, p.u.
    ref: np.ndarray              # slack bus indices
    pv: np.ndarray               # PV bus indices
    pq: np.ndarray               # PQ bus indices
    v0: np.ndarray               # flat-start complex voltage

    # Independent references from pandapower, for validation only.
    ybus_ref: sp.csr_matrix | None = field(default=None, repr=False)
    v_ref: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_bus(self) -> int:
        return self.bus.shape[0]

    @property
    def n_branch(self) -> int:
        return self.branch.shape[0]

    @property
    def pvpq(self) -> np.ndarray:
        """Buses with a P equation: PV and PQ, in MATPOWER's ordering."""
        return np.r_[self.pv, self.pq]

    @property
    def n_unknowns(self) -> int:
        """Size of the Newton system: one angle per PV/PQ bus, one magnitude per PQ bus."""
        return len(self.pv) + 2 * len(self.pq)

    def __repr__(self) -> str:
        return (
            f"Case({self.name}: {self.n_bus} buses, {self.n_branch} branches, "
            f"{len(self.ref)} slack / {len(self.pv)} PV / {len(self.pq)} PQ, "
            f"n_unknowns={self.n_unknowns})"
        )


def load_case(name: str = "case14") -> Case:
    """Load a standard test case and build everything the Newton solver needs.

    Parameters
    ----------
    name
        Any case in ``pandapower.networks``, e.g. ``"case9"``, ``"case14"``,
        ``"case30"``, ``"case118"``, ``"case300"``, ``"case1354pegase"``.
    """
    import warnings

    import pandapower as pp
    import pandapower.networks as pn

    if not hasattr(pn, name):
        raise ValueError(f"unknown case {name!r}; not found in pandapower.networks")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = getattr(pn, name)()
        pp.runpp(net, numba=False)

    ppc = net["_ppc"]
    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float).real
    gen = np.asarray(ppc["gen"], dtype=float).real

    ybus = build_ybus(base_mva, bus, branch)
    sbus = make_sbus(base_mva, bus, gen)
    ref, pv, pq = bus_types(bus, gen)
    v0 = flat_start(bus, gen)

    internal = ppc.get("internal", {})
    ybus_ref = internal.get("Ybus")
    v_ref = internal.get("V")

    return Case(
        name=name, base_mva=base_mva, bus=bus, branch=branch, gen=gen,
        ybus=ybus, sbus=sbus, ref=ref, pv=pv, pq=pq, v0=v0,
        ybus_ref=sp.csr_matrix(ybus_ref) if ybus_ref is not None else None,
        v_ref=np.asarray(v_ref) if v_ref is not None else None,
    )


# --------------------------------------------------------------------------------------
# Network model
# --------------------------------------------------------------------------------------

def build_ybus(base_mva: float, bus: np.ndarray, branch: np.ndarray) -> sp.csr_matrix:
    r"""Build the bus admittance matrix :math:`\mathbf{Y}_{bus}`.

    Each branch is a :math:`\pi`-model with series admittance
    :math:`y_s = 1/(r + jx)`, total charging admittance :math:`y_c = g_c + j b_c` split
    equally between the two ends, and an ideal transformer of complex ratio
    :math:`\tau = a\,e^{j\theta_{shift}}` on the *from* side:

    .. math::
        \begin{bmatrix} i_f \\ i_t \end{bmatrix} =
        \begin{bmatrix}
            (y_s + y_c/2)/|\tau|^2 & -y_s/\bar{\tau} \\
            -y_s/\tau              & \;\;y_s + y_c/2
        \end{bmatrix}
        \begin{bmatrix} v_f \\ v_t \end{bmatrix}

    Classic MATPOWER carries only the susceptance :math:`b_c` here.  The ``BR_G`` column
    is a pandapower extension holding a branch shunt *conductance* (dielectric / corona
    losses), nonzero in e.g. ``case118`` and ``case300``.  Dropping it perturbs only the
    real part of the :math:`\mathbf{Y}_{bus}` diagonal -- a ~1e-5 relative error that is
    easy to miss and shifts the converged solution in the fifth decimal.

    Branch contributions are scattered into the bus frame with the connection matrices
    :math:`\mathbf{C}_f, \mathbf{C}_t`, and bus shunts :math:`(g_s + j b_s)/S_{base}`
    are added on the diagonal:

    .. math::
        \mathbf{Y}_{bus} = \mathbf{C}_f^\top \mathbf{Y}_f
                         + \mathbf{C}_t^\top \mathbf{Y}_t
                         + \operatorname{diag}(\mathbf{y}_{sh})

    This is a direct transcription of MATPOWER's ``makeYbus.m``.
    """
    n_bus, n_branch = bus.shape[0], branch.shape[0]

    for col, label in ((BR_R_ASYM, "BR_R_ASYM"), (BR_X_ASYM, "BR_X_ASYM"),
                       (BR_G_ASYM, "BR_G_ASYM"), (BR_B_ASYM, "BR_B_ASYM")):
        if col < branch.shape[1] and np.any(branch[:, col]):
            raise NotImplementedError(
                f"branch column {label} is nonzero; asymmetric branches are not modelled"
            )

    status = branch[:, BR_STATUS].real
    ys = status / (branch[:, BR_R] + 1j * branch[:, BR_X])   # out-of-service -> 0
    yc = status * (branch[:, BR_G] + 1j * branch[:, BR_B])   # shunt g + jb, total

    # Tap ratio: MATPOWER stores 0 for "no transformer", meaning a ratio of 1.
    tap = np.ones(n_branch, dtype=complex)
    has_tap = branch[:, TAP].real != 0
    tap[has_tap] = branch[has_tap, TAP].real
    tap = tap * np.exp(1j * np.pi / 180.0 * branch[:, SHIFT].real)

    y_tt = ys + yc / 2.0
    y_ff = y_tt / (tap * np.conj(tap))
    y_ft = -ys / np.conj(tap)
    y_tf = -ys / tap

    y_sh = (bus[:, GS] + 1j * bus[:, BS]) / base_mva

    f = branch[:, F_BUS].real.astype(int)
    t = branch[:, T_BUS].real.astype(int)
    rows = np.arange(n_branch)
    c_f = sp.csr_matrix((np.ones(n_branch), (rows, f)), shape=(n_branch, n_bus))
    c_t = sp.csr_matrix((np.ones(n_branch), (rows, t)), shape=(n_branch, n_bus))

    y_f = sp.diags(y_ff) @ c_f + sp.diags(y_ft) @ c_t
    y_t = sp.diags(y_tf) @ c_f + sp.diags(y_tt) @ c_t

    ybus = c_f.T @ y_f + c_t.T @ y_t + sp.diags(y_sh)
    return sp.csr_matrix(ybus)


def make_sbus(base_mva: float, bus: np.ndarray, gen: np.ndarray) -> np.ndarray:
    r"""Scheduled complex power injection per bus, in per unit.

    .. math::
        \mathbf{S}^{spec} = \frac{1}{S_{base}}
            \left( \mathbf{C}_g (\mathbf{P}_g + j\mathbf{Q}_g)
                 - (\mathbf{P}_d + j\mathbf{Q}_d) \right)

    Note that the entry at the slack bus and the imaginary parts at PV buses are never
    used by the Newton iteration -- those equations are simply not enforced.
    """
    n_bus = bus.shape[0]
    on = gen[:, GEN_STATUS].real > 0
    gbus = gen[on, GEN_BUS].real.astype(int)

    s_gen = np.zeros(n_bus, dtype=complex)
    np.add.at(s_gen, gbus, gen[on, PG].real + 1j * gen[on, QG].real)

    s_load = bus[:, PD] + 1j * bus[:, QD]
    return (s_gen - s_load) / base_mva


def bus_types(bus: np.ndarray, gen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split buses into (slack, PV, PQ) index arrays.

    A bus keeps its PV/slack status only if it has at least one in-service generator;
    otherwise it is demoted to PQ.  This mirrors MATPOWER's ``bustypes.m``.
    """
    n_bus = bus.shape[0]
    on = gen[:, GEN_STATUS].real > 0
    gbus = gen[on, GEN_BUS].real.astype(int)
    has_gen = np.zeros(n_bus, dtype=bool)
    has_gen[gbus] = True

    btype = bus[:, BUS_TYPE].real.astype(int)
    ref = np.flatnonzero((btype == REF) & has_gen)
    pv = np.flatnonzero((btype == PV) & has_gen)
    pq = np.flatnonzero((btype == PQ) | ~has_gen)
    pq = np.setdiff1d(pq, np.r_[ref, pv])

    if len(ref) == 0:  # no generator at the slack bus: fall back to the declared one
        ref = np.flatnonzero(btype == REF)
        pq = np.setdiff1d(pq, ref)
    return ref, pv, pq


def flat_start(bus: np.ndarray, gen: np.ndarray) -> np.ndarray:
    r"""Flat start: :math:`|V| = 1`, :math:`\theta = \theta_{slack}` everywhere.

    Voltage magnitudes at slack and PV buses are overwritten with their generator
    setpoints :math:`V_g`, since those are held fixed by the iteration, not solved for.

    All angles start at the *slack* bus angle rather than at zero.  The power flow
    equations only involve angle *differences*, so a uniform rotation of every
    :math:`\mathbf{V}` is an exact symmetry of the problem -- both choices converge in
    the same number of iterations to the same physical state.  We adopt the slack angle
    anyway so that our solution is directly comparable to pandapower's without a
    post-hoc rotation (``case118``, for instance, specifies a 30 deg slack angle).
    """
    n_bus = bus.shape[0]

    ref = np.flatnonzero(bus[:, BUS_TYPE].real.astype(int) == REF)
    theta_slack = np.radians(bus[ref[0], VA].real) if len(ref) else 0.0
    v = np.full(n_bus, np.exp(1j * theta_slack), dtype=complex)

    on = gen[:, GEN_STATUS].real > 0
    gbus = gen[on, GEN_BUS].real.astype(int)
    vg = gen[on, VG].real
    valid = np.isfinite(vg) & (vg > 0)
    v[gbus[valid]] = vg[valid] * np.exp(1j * theta_slack)
    return v


# --------------------------------------------------------------------------------------
# Newton-Raphson
# --------------------------------------------------------------------------------------

def mismatch(v: np.ndarray, ybus: sp.spmatrix, sbus: np.ndarray,
             pv: np.ndarray, pq: np.ndarray) -> np.ndarray:
    r"""Real-valued power mismatch vector :math:`\mathbf{f}(x)`.

    The complex mismatch at every bus is

    .. math::
        \Delta\mathbf{S} = \mathbf{V} \odot
            \overline{(\mathbf{Y}_{bus}\mathbf{V})} - \mathbf{S}^{spec}

    of which we enforce the active part at PV and PQ buses and the reactive part at PQ
    buses only, stacked as

    .. math::
        \mathbf{f} = \begin{bmatrix}
            \Re\{\Delta\mathbf{S}\}_{[pv,pq]} \\ \Im\{\Delta\mathbf{S}\}_{[pq]}
        \end{bmatrix} \in \mathbb{R}^{n_{pv} + 2 n_{pq}}
    """
    d_s = v * np.conj(ybus @ v) - sbus
    return np.r_[d_s[np.r_[pv, pq]].real, d_s[pq].imag]


def build_jacobian(v: np.ndarray, ybus: sp.spmatrix,
                   pv: np.ndarray, pq: np.ndarray) -> sp.csr_matrix:
    r"""Polar-form power flow Jacobian :math:`\mathbf{J} = \partial\mathbf{f}/\partial x`.

    Starting from the complex-valued partials (MATPOWER's ``dSbus_dV.m``), with
    :math:`\mathbf{I} = \mathbf{Y}_{bus}\mathbf{V}`:

    .. math::
        \frac{\partial \mathbf{S}}{\partial \boldsymbol{\theta}}
            = j\operatorname{diag}(\mathbf{V})
              \overline{\left(\operatorname{diag}(\mathbf{I})
              - \mathbf{Y}_{bus}\operatorname{diag}(\mathbf{V})\right)},
        \qquad
        \frac{\partial \mathbf{S}}{\partial |\mathbf{V}|}
            = \operatorname{diag}(\mathbf{V})
              \overline{\left(\mathbf{Y}_{bus}\operatorname{diag}(\hat{\mathbf{V}})\right)}
            + \operatorname{diag}(\overline{\mathbf{I}})\operatorname{diag}(\hat{\mathbf{V}})

    where :math:`\hat{\mathbf{V}} = \mathbf{V}/|\mathbf{V}|`.  Taking real and imaginary
    parts on the appropriate bus subsets gives the familiar four blocks:

    .. math::
        \mathbf{J} = \begin{bmatrix}
            \partial \mathbf{P}/\partial\boldsymbol{\theta} & \partial \mathbf{P}/\partial|\mathbf{V}| \\
            \partial \mathbf{Q}/\partial\boldsymbol{\theta} & \partial \mathbf{Q}/\partial|\mathbf{V}|
        \end{bmatrix}
    """
    ybus = sp.csr_matrix(ybus)
    i_bus = ybus @ v
    diag_v = sp.diags(v)
    diag_i = sp.diags(i_bus)
    diag_vnorm = sp.diags(v / np.abs(v))

    ds_dva = 1j * diag_v @ (diag_i - ybus @ diag_v).conj()
    ds_dvm = diag_v @ (ybus @ diag_vnorm).conj() + diag_i.conj() @ diag_vnorm

    pvpq = np.r_[pv, pq]
    j11 = ds_dva[pvpq, :][:, pvpq].real
    j12 = ds_dvm[pvpq, :][:, pq].real
    j21 = ds_dva[pq, :][:, pvpq].imag
    j22 = ds_dvm[pq, :][:, pq].imag

    return sp.csr_matrix(sp.vstack([
        sp.hstack([j11, j12]),
        sp.hstack([j21, j22]),
    ], format="csr"))


@dataclass
class NewtonResult:
    """Outcome of a Newton-Raphson solve, including the full iteration history."""

    v: np.ndarray                 # converged complex voltage
    converged: bool
    n_iter: int
    norm_history: list[float]     # ||f||_inf at iterations 0, 1, ..., n_iter
    v_history: list[np.ndarray] = field(default_factory=list, repr=False)
    j_history: list[sp.csr_matrix] = field(default_factory=list, repr=False)
    f_history: list[np.ndarray] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        status = "converged" if self.converged else "DIVERGED"
        return (f"NewtonResult({status} in {self.n_iter} iterations, "
                f"final ||f||_inf = {self.norm_history[-1]:.3e})")


def newton_pf(case: Case, v0: np.ndarray | None = None, tol: float = 1e-10,
              max_iter: int = 20, store_history: bool = False,
              linear_solve=None) -> NewtonResult:
    r"""Solve the AC power flow by Newton-Raphson in polar coordinates.

    The iteration is the textbook one, identical to MATPOWER's ``newtonpf.m``:

    .. math::
        \mathbf{J}(x_k)\,\Delta x_k = -\mathbf{f}(x_k), \qquad x_{k+1} = x_k + \Delta x_k

    with :math:`x = [\boldsymbol{\theta}_{[pv,pq]}, |\mathbf{V}|_{[pq]}]`.

    Parameters
    ----------
    tol
        Convergence tolerance on :math:`\lVert\mathbf{f}\rVert_\infty` (p.u.).
        MATPOWER's default is 1e-8; we use a tighter 1e-10 so that the quadratic
        convergence tail is visible in the plots.
    store_history
        Keep every iterate, Jacobian and residual.  Notebook 01 needs these to study
        the linear subproblem, so it is off by default but cheap for small cases.
    linear_solve
        Callable ``(J, f) -> dx`` replacing the sparse LU solve.  This is the hook that
        every later notebook plugs into: GMRES, preconditioned GMRES, and eventually a
        learned iteration all enter here without touching the outer loop.
    """
    if linear_solve is None:
        linear_solve = lambda j, f: sp.linalg.spsolve(sp.csc_matrix(j), -f)  # noqa: E731

    v = (case.v0 if v0 is None else v0).astype(complex).copy()
    ybus, sbus, pv, pq = case.ybus, case.sbus, case.pv, case.pq
    n_pv, n_pq = len(pv), len(pq)

    f = mismatch(v, ybus, sbus, pv, pq)
    norm_history = [float(np.linalg.norm(f, np.inf))]
    v_hist, j_hist, f_hist = ([v.copy()], [], [f.copy()]) if store_history else ([], [], [])

    converged = norm_history[0] < tol
    k = 0
    while not converged and k < max_iter:
        j = build_jacobian(v, ybus, pv, pq)
        dx = linear_solve(j, f)

        v_a, v_m = np.angle(v), np.abs(v)
        v_a[pv] += dx[:n_pv]
        v_a[pq] += dx[n_pv:n_pv + n_pq]
        v_m[pq] += dx[n_pv + n_pq:]
        v = v_m * np.exp(1j * v_a)

        f = mismatch(v, ybus, sbus, pv, pq)
        norm_history.append(float(np.linalg.norm(f, np.inf)))
        if store_history:
            j_hist.append(j)
            v_hist.append(v.copy())
            f_hist.append(f.copy())

        k += 1
        converged = norm_history[-1] < tol

    return NewtonResult(v=v, converged=bool(converged), n_iter=k,
                        norm_history=norm_history, v_history=v_hist,
                        j_history=j_hist, f_history=f_hist)
