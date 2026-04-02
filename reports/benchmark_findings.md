# OSQP.m Benchmark Findings

Date: 2026-04-01
Workspace: OSQP.m
MATLAB: R2025b

## Summary

MATLAB `ldl()` is the fastest backend overall in this benchmark run.

- Total median runtime across all benchmark problems:
  - Optimised QDLDL: 8763.20 ms
  - QDLDL C MEX: 3350.71 ms
  - Pardiso MKL: 960.72 ms
  - MATLAB `ldl()`: 181.81 ms
- Relative to Python `osqp` baseline:
  - Optimised QDLDL: 158.7x slower
  - QDLDL C MEX: 60.7x slower
  - Pardiso MKL: 17.4x slower
  - MATLAB `ldl()`: 3.3x slower

## Benchmark Command

The benchmark was run with:

```bash
/home/jason/Programs/MATLAB/2025b/bin/matlab -batch "run('test/bench_all.m')"
```

## Per-Example Median Timings

| Example | Optimised QDLDL | QDLDL C MEX | Pardiso MKL | MATLAB ldl() |
|---|---:|---:|---:|---:|
| Ex1: Setup+Solve (2x2) | 3.53 ms | 3.13 ms | 5.30 ms | 2.47 ms |
| Ex2: Update Vectors | 4.94 ms | 5.19 ms | 6.50 ms | 4.38 ms |
| Ex3: Update Matrices | 9.50 ms | 6.28 ms | 7.19 ms | 4.46 ms |
| Ex4: Least Squares (50 vars) | 5.35 ms | 4.78 ms | 10.38 ms | 3.43 ms |
| Ex5: MPC (172 vars, 15 steps) | 182.64 ms | 106.38 ms | 176.84 ms | 58.33 ms |
| Ex6: Huber (310 vars) | 19.39 ms | 13.13 ms | 19.43 ms | 3.92 ms |
| Ex7: SVM (1010 vars) | 6463.26 ms | 2484.88 ms | 395.24 ms | 39.92 ms |
| Ex8: Lasso (1020v, 11 solves) | 2074.59 ms | 726.94 ms | 339.84 ms | 64.90 ms |

## Interpretation

- MATLAB `ldl()` is the best default choice for this MATLAB implementation.
- Pardiso MKL gives large gains over QDLDL and QDLDL C on larger problems (especially SVM and Lasso), but still trails MATLAB `ldl()` in total benchmark time.
- Pure MATLAB QDLDL remains useful for portability and algorithmic experimentation, but is not competitive for speed.

## Recommendation

- Use `linear_solver = 'matlab_ldl'` as the primary backend.
- Keep `pardiso_mkl` available as an optional backend for large sparse workloads and future tuning.
- Focus future speed work on MATLAB-side ADMM loop overhead, since linear-system speedups alone no longer explain most of the remaining gap to Python.
