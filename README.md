# OSQP.m

A pure-MATLAB implementation of the [OSQP](https://osqp.org/) (Operator Splitting
Quadratic Program) solver, using [QDLDL](https://github.com/osqp/qdldl) for
internal KKT linear system factorization. Ported from
[OSQP.jl](https://github.com/osqp/OSQP.jl).

## Requirements

- MATLAB R2021a or later (uses `arguments` blocks)
- [qdldl.m](https://github.com/JasonNicholsonMATLAB/qdldl.m) — a pure-MATLAB QDLDL implementation (included via submodule)

## Setup

```matlab
% From the repo root, add solver and qdldl to the MATLAB path
setupPath
```

## Problem class

OSQP solves convex quadratic programs of the form:

```
minimize        0.5 x' P x + q' x

subject to      l <= A x <= u
```

where `x in R^n` is the optimization variable. The objective function is defined
by a positive semidefinite matrix `P in S^n_+` and vector `q in R^n`. The linear
constraints are defined by matrix `A in R^{m x n}` and vectors
`l in R^m U {-inf}^m`, `u in R^m U {+inf}^m`.

## Usage

```matlab
solver = OSQP();
solver.setup(P, q, A, l, u, 'verbose', false, 'eps_abs', 1e-6);
results = solver.solve();

% Access solution
x_opt = results.x;
y_opt = results.y;
status = results.info.status;
```

### Updating problem data

```matlab
solver.update('q', q_new);                          % update linear cost
solver.update('l', l_new, 'u', u_new);              % update bounds
solver.update('Px', nonzeros(triu(P_new)));          % update P values
solver.update('Ax', nonzeros(A_new));                % update A values
```

### Warm starting

```matlab
solver.warm_start('x', x0, 'y', y0);
results = solver.solve();
```

### Changing settings

```matlab
solver.update_settings('max_iter', 10000, 'polish', true);
```

## Running tests

```matlab
setupPath
run_all_tests    % from the test/ directory
```

## Documentation

For algorithm details, see the [OSQP documentation](https://osqp.org/).

## License

OSQP.m is licensed under the [Apache-2.0 License](LICENSE.md).

    Copyright (c) 2026 Jason H. Nicholson
    Copyright (c) 2017 Bartolomeo Stellato, Baris Stellato, and OSQP contributors
