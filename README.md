# Learned Iterative Solvers for AC Power Flow

Learning tailored iterative algorithms for AC power flow — from Newton–Raphson to Krylov
to learned solvers, one notebook at a time.

This is a **learning repository**. The goal is not a benchmark win; it is to understand,
from first principles and with honest numbers, what it means to *learn a solver* rather
than *learn a solution*.

---

## The question

Solving an AC power flow means finding the bus voltage angles and magnitudes
$x = [\boldsymbol{\theta}, |\mathbf{V}|]$ that zero the power mismatch:

$$\mathbf{f}(x) = \mathbf{S}^{\text{inj}}(x) - \mathbf{S}^{\text{spec}} = \mathbf{0}$$

Newton–Raphson does this by repeatedly forming the Jacobian $\mathbf{J}$ and solving
$\mathbf{J}\,\Delta x = -\mathbf{f}(x)$ by sparse LU. It works, and it converges
quadratically in 3–5 iterations. So why change anything?

Because in practice you rarely solve *one* power flow. Contingency screening on a
2000-bus grid means thousands of power flows that are all *nearly the same problem* —
same network, same loading, one branch removed. A classical solver throws away
everything it learned from instance $k$ before starting instance $k+1$.

The idea explored here, following [4] and [5], is to keep it: use historical solve data
to learn an iteration that is *tailored* to a family of problems, while still evaluating
the true physical mismatch $\mathbf{f}(x)$ at every step.

## Learn the solver, not the solution

|  | Direct prediction (e.g. DNN-OPF) | Learned iterative algorithm |
|---|---|---|
| Network outputs | the **solution** $\hat{x}$ | the **update rule** $\Delta x$ |
| Physics evaluated at runtime | no | yes — true mismatch every iteration |
| Error known online | no | yes, it *is* the mismatch |
| A bad network gives you | a wrong answer, silently | a slower answer, still correct |
| Accuracy ceiling | network quality | machine precision |

The second column can be safeguarded: if a learned step fails to reduce
$\lVert\mathbf{f}\rVert$, fall back to a Newton step. The worst case degrades to Newton.

## Roadmap

Each milestone is a self-contained notebook with figures and a plain-language reading.

| # | Notebook | Content | Status |
|---|---|---|---|
| 01 | [`01_newton_and_krylov.ipynb`](notebooks/01_newton_and_krylov.ipynb) | AC power flow Newton–Raphson from scratch, validated against pandapower. Then the linear subproblem $\mathbf{J}\Delta x = -\mathbf{f}$ three ways: sparse LU, GMRES, preconditioned GMRES. | **done** |
| 02 | [`02_does_the_family_share_structure.ipynb`](notebooks/02_does_the_family_share_structure.ipynb) | Does a family of power flows share enough structure to learn from? Where the variation lives, what object can be frozen, preconditioner reuse across N-1, and a fully fixed solver tested end-to-end on unseen contingencies. | **done** |
| 03 | — | Jacobian-free: take $\mathbf{J}v$ from a finite difference of the residual and drop matrix assembly entirely — the cost that currently dominates. | next |
| 04 | — | The Runge–Kutta neural network [4] reproduced on ODEs — the simplest instance of a learned algorithm, and where learnable *inner* coefficients come from. | planned |
| 05 | — | R2N2 [5] on the linear subproblem: learned inner recurrence vs. the doomed monomial basis. | planned |
| 06 | — | R2N2 on the full nonlinear AC power flow. | planned |
| 07 | — | Where it breaks: distribution shift, near-collapse loading, safeguarded variants. | planned |
| 08 | — | Batched GPU contingency screening — the regime where matrix-free can actually win. | planned |

## Findings so far

**Notebook 01.** Validated a from-scratch AC power flow against pandapower to machine
precision on six cases, then measured the linear subproblem honestly:

- Newton converges in **4–5 iterations** from a flat start on every case from 9 to 300
  buses, with measured quadratic order. There is no room to win on outer iterations.
- The linear solve does **not** dominate at small sizes — on `case118` it is ~10% of a
  Newton iteration. Its share crosses 50% only near $n \approx 2400$ and reaches ~65% on
  `case2869pegase`. (The crossover is implementation-dependent; the scaling is not.)
- Plain GMRES needs $m = 120$ of $n = 181$ Krylov dimensions to reach $10^{-8}$ on
  `case118` — **2.2× slower** than the sparse LU it was meant to replace.
- ILU-preconditioned GMRES converges in **~4 matvecs**, but building the ILU costs
  **1.13× a full LU solve**, so rebuilt per system it is *also* slower than LU.
- **On a single power flow, nothing beats sparse LU.** Any speed-up must come from
  amortizing work across many related systems — which is precisely the premise of
  learning a solver, and precisely the regime contingency analysis lives in.

![GMRES residual vs Krylov dimension](figures/01_gmres_residual_vs_dimension.png)

**Notebook 02.** Sampled a family from `case118` — load/dispatch perturbations and all 177
non-islanding N-1 outages — and asked whether it shares enough structure to learn from:

- **$\mathbf{J} = \mathbf{J}(\mathbf{V}, \mathbf{Y}_{bus})$ — the Jacobian does not depend
  on the injections.** Under load variation at a fixed state it is *bit-identical*; a
  branch outage moves it 10–100× more. Topology, not loading, is the interesting axis.
- **A reused preconditioner transfers.** One ILU factored from the intact grid conditions
  every contingency Jacobian ($\kappa_2: 3000 \to \approx 1$), even for outages that move
  $\mathbf{J}$ by 30%.
- **A single frozen coefficient vector cannot match per-instance refitting** — it plateaus
  around $2\times10^{-4}$ against the oracle's $6\times10^{-11}$, and stops responding to
  degree past $m\approx4$.
- **Training on the wrong distribution cost 0/97 convergence.** Fitting only on flat-start
  systems failed completely; fitting over the states Newton actually visits gave 97/97.
- **End to end: it works, but is not yet faster.** A fully frozen solver — one
  preconditioner, one coefficient vector, zero runtime adaptation — converges all 97 unseen
  contingencies in the same 4 Newton iterations as sparse LU at $m=12$. Cheaper linear
  solves buy extra Newton steps, and each of those pays for a Jacobian assembly.

![A fully frozen solver on unseen contingencies](figures/02_end_to_end.png)

## Layout

```
lis/            reusable library code (power flow, Krylov, plotting)
notebooks/      the readable narrative — one per milestone, figures baked in
refs/           papers (gitignored) + bibliography with citation keys [1]–[5]
figures/        exported figures
data/           generated datasets (gitignored)
```

## References

Full bibliography with links: [`refs/README.md`](refs/README.md).
Citation keys `[1]`–`[5]` are used consistently across all notebooks.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, PyTorch 2.11 (CUDA 12.8), pandapower 3.5. A GPU is only needed from
notebook 08 onward.
