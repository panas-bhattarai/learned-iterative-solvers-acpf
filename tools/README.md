# Notebook builders

Each notebook in `notebooks/` is **generated** by the matching script here, then executed
with outputs baked in. The scripts are the source of truth: edit `build_nbNN.py`, not the
`.ipynb`.

```bash
python tools/build_nb03.py                                    # regenerate (no outputs)
python -m jupyter nbconvert --to notebook --execute --inplace \
       --ExecutePreprocessor.timeout=7200 notebooks/03_jacobian_free.ipynb
```

Why generate rather than hand-edit: the notebooks carry long LaTeX narratives interleaved
with code, and several went through many revisions as measurements contradicted the draft
text. Keeping prose and code in one plain-text file made those revisions reviewable in
`git diff`, which editing JSON would not.

**Exception.** Notebook 05's markdown was patched directly in the `.ipynb` after execution,
to correct three numbers without paying for a 13-minute re-run. `tools/build_nb05.py`
therefore still contains the pre-patch wording for those three sentences. If you regenerate
05 from the builder, re-apply: the equal-budget ratio is 1.6–2.6× (not 1.2–2.2×), training
leaves the basis *worse* conditioned at every $n$ (there is no improvement at $n=3$ in a
3000-epoch run), and the nonlinear $n=5$ trace reaches $9.5\times10^{-10}$.

## Approximate run times

Measured on an RTX 3050 laptop with a Python 3.11 global install.

| notebook | execution | dominant cost |
|---|---|---|
| 01 | ~1 min | timing sweeps up to `case2869pegase` |
| 02 | ~1 min | 177 N-1 solves, repeated least-squares fits |
| 03 | ~1 min | best-of-5 timings across four grid sizes |
| 04 | ~4 min | RK-NN training, up to 12000 epochs |
| 05 | ~13 min | R2N2 training, linear and nonlinear, on GPU |
| 06 | ~4 min | loadability sweep and N-$k$ sampling, all CPU |
| 07 | ~2 min | batched GPU solves and the throughput sweep |
