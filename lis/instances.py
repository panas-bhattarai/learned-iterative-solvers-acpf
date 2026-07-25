r"""Sampling a *family* of power flow problems from one grid.

Notebook 01 established that no learned method beats sparse LU on a single power flow.
The only way a learned solver can pay for itself is by amortizing work across many
related problems -- so the object of study from here on is not a problem, it is a
*distribution over problems*.

The physically meaningful variation on a fixed transmission network is:

* **loading** -- daily and seasonal demand, which moves every :math:`P_d, Q_d` together
  (a global scale) plus bus-to-bus variation (local jitter);
* **dispatch** -- which generators cover that demand;
* **topology** -- N-1 contingencies, i.e. one branch out of service.

This module samples the first two by perturbing the ``bus`` and ``gen`` tables, and the
third by flipping ``BR_STATUS``.  Everything else -- line impedances, bus types, voltage
setpoints -- is held fixed, because that is what "the same grid" means.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pandapower.pypower.idx_bus import PD, QD
from pandapower.pypower.idx_brch import BR_STATUS
from pandapower.pypower.idx_gen import GEN_BUS, PG, GEN_STATUS

from lis.powerflow import Case, build_ybus, make_sbus, newton_pf

__all__ = ["perturb_case", "sample_family", "n1_variants"]


def perturb_case(case: Case, rng: np.random.Generator,
                 scale_range: tuple[float, float] = (0.8, 1.2),
                 jitter_sigma: float = 0.10,
                 track_generation: bool = True) -> Case:
    r"""One load/dispatch sample from the neighbourhood of ``case``.

    Loads are perturbed multiplicatively,

    .. math::
        P_d^{(i)} = s \cdot \eta_i \cdot P_d^{base}, \qquad
        s \sim \mathcal{U}(s_{\min}, s_{\max}), \quad
        \eta_i \sim \mathrm{LogNormal}(0, \sigma)

    with a single global scale :math:`s` shared by every bus (the daily demand curve) and
    an independent per-bus factor :math:`\eta_i` (local variation).  :math:`Q_d` is scaled
    by the same factor as :math:`P_d` at each bus, holding power factor fixed.

    Parameters
    ----------
    track_generation
        Scale non-slack generator setpoints by the same global :math:`s`, so the slack bus
        only absorbs losses rather than the entire demand swing.  This keeps the sampled
        operating points realistic; without it, large :math:`s` drives all the extra power
        through the slack bus and distorts the Jacobian.
    """
    bus = case.bus.copy()
    gen = case.gen.copy()

    scale = rng.uniform(*scale_range)
    jitter = rng.lognormal(mean=0.0, sigma=jitter_sigma, size=case.n_bus)
    factor = scale * jitter

    bus[:, PD] = case.bus[:, PD] * factor
    bus[:, QD] = case.bus[:, QD] * factor      # constant power factor per bus

    if track_generation:
        on = gen[:, GEN_STATUS].real > 0
        non_slack = on & ~np.isin(gen[:, GEN_BUS].real.astype(int), case.ref)
        gen[non_slack, PG] = case.gen[non_slack, PG] * scale

    return replace(case, bus=bus, gen=gen,
                   sbus=make_sbus(case.base_mva, bus, gen),
                   ybus_ref=None, v_ref=None)


def n1_variants(case: Case, branch_ids: np.ndarray | None = None) -> list[Case]:
    r"""One case per single-branch outage (N-1).

    Taking a branch out of service changes :math:`\mathbf{Y}_{bus}` but leaves the bus
    tables alone, so these samples probe *topological* rather than loading variation.
    Outages that island the network are skipped -- they make the power flow singular,
    which is a different problem from the one being studied here.
    """
    if branch_ids is None:
        branch_ids = np.arange(case.n_branch)

    out = []
    for b in branch_ids:
        if case.branch[b, BR_STATUS].real == 0:
            continue
        branch = case.branch.copy()
        branch[b, BR_STATUS] = 0
        ybus = build_ybus(case.base_mva, case.bus, branch)
        # A bus left with only its own diagonal entry has been islanded by the outage.
        if np.any(np.diff(ybus.tocsr().indptr) <= 1):
            continue
        out.append(replace(case, name=f"{case.name}_n1_{b}", branch=branch,
                           ybus=ybus, ybus_ref=None, v_ref=None))
    return out


def sample_family(case: Case, n_samples: int, seed: int = 0,
                  require_convergence: bool = True, **kwargs) -> list[Case]:
    """Draw ``n_samples`` perturbed cases, optionally keeping only those that solve.

    Rejecting non-convergent draws keeps the study focused on the ordinary operating
    envelope.  The count of rejections is worth watching: if it is large, the sampling
    range is pushing the grid past its loadability limit rather than exploring it.
    """
    rng = np.random.default_rng(seed)
    kept, tried = [], 0
    while len(kept) < n_samples:
        tried += 1
        c = perturb_case(case, rng, **kwargs)
        if require_convergence and not newton_pf(c, tol=1e-10, max_iter=20).converged:
            continue
        kept.append(c)
        if tried > 20 * n_samples:
            raise RuntimeError(
                f"only {len(kept)}/{n_samples} samples converged after {tried} draws; "
                "the perturbation range is probably past the loadability limit"
            )
    return kept
