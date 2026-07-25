# Bibliography

The PDFs themselves are gitignored (copyright). Links below are to the open-access
versions. Citation keys `[1]`–`[5]` are used throughout the notebooks.

---

**[1]** V. Monga, Y. Li, and Y. C. Eldar,
*"Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image
Processing,"* IEEE Signal Processing Magazine, vol. 38, no. 2, pp. 18–44, 2021.
[arXiv:1912.10557](https://arxiv.org/abs/1912.10557) · `refs/1912.10557v3.pdf`

> The conceptual frame for the whole project: take a hand-designed iterative algorithm,
> unroll a fixed number of its iterations into a computational graph, and learn the
> parameters that the algorithm designer would otherwise have chosen analytically.

**[2]** K. Gregor and Y. LeCun,
*"Learning Fast Approximations of Sparse Coding,"* ICML 2010, pp. 399–406.
[PDF](https://icml.cc/Conferences/2010/papers/449.pdf) · `refs/gregor_lecun_2010_LISTA.pdf`

> LISTA — the paper that started algorithm unrolling. Unrolls ISTA into a small recurrent
> network and learns its matrices, reaching in ~10 layers what ISTA needs hundreds of
> iterations for.

**[3]** Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM, 2003.
[Author's free PDF](https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf) ·
`refs/IterMethBook_2ndEd.pdf`

> Reference text. We use Ch. 6 (Krylov subspace methods — Arnoldi, GMRES, FOM) and
> Ch. 9–10 (preconditioning).

**[4]** Y. Guo, F. Dietrich, T. Bertalan, D. T. Doncevic, M. Dahmen, I. G. Kevrekidis,
and Q. Li, *"Personalized Algorithm Generation: A Case Study in Learning ODE
Integrators,"* SIAM J. Sci. Comput., vol. 44, no. 4, pp. A1911–A1933, 2022.
[arXiv:2105.01303](https://arxiv.org/abs/2105.01303) · `refs/2105.01303v3.pdf`

> The Runge–Kutta neural network. Learns RK coefficients (the Butcher tableau) from data
> for a *targeted family* of ODEs, rather than deriving them by hand for all ODEs.
> Direct predecessor of [5].

**[5]** D. T. Doncevic, A. Mitsos, Y. Guo, Q. Li, F. Dietrich, M. Dahmen, and
I. G. Kevrekidis, *"A Recursively Recurrent Neural Network (R2N2) Architecture for
Learning Iterative Algorithms,"* SIAM J. Sci. Comput., vol. 46, no. 2,
pp. A719–A743, 2024.
[arXiv:2211.12386](https://arxiv.org/abs/2211.12386) · `refs/2211.12386v2.pdf`

> The destination. Generalizes [4] into a superstructure with inner iterations (generate
> a subspace by repeated function evaluations) and an outer update (learned linear
> combination of those evaluations). Trained on different problem classes it recovers
> Krylov solvers, Newton–Krylov solvers, and Runge–Kutta integrators.

---

## Reading order

`[1] → [2] → [3] Ch.6 → [4] → [5]`

## To be added later

Flagged here when the notebooks reach them, not before:

- Amos & Kolter, *OptNet: Differentiable Optimization as a Layer in Neural Networks*
- Agrawal et al., *Differentiable Convex Optimization Layers*
- Bai, Kolter & Koltun, *Deep Equilibrium Models* — a converged power flow is a fixed point
- Blondel et al., *Efficient and Modular Implicit Differentiation*
- Häusner et al., *Neural Incomplete Factorization*; Li et al., *Learning Preconditioners
  for Conjugate Gradient PDE Solvers*
- R. D. Zimmerman et al., *MATPOWER* (IEEE Trans. Power Syst., 2011) — case data provenance
