% OSQP - MATLAB implementation of the OSQP solver
%
% Pure-MATLAB port of the OSQP (Operator Splitting Quadratic Program) solver.
% Uses QDLDL for the internal KKT linear system factorization.
%
% Main class:
%   OSQP  - solver class (setup, solve, update, warm_start)
%
% See also OSQP, qdldl, setupPath

% Copyright (c) 2026 Jason H. Nicholson
% Copyright (c) 2017 Bartolomeo Stellato, Baris Stellato, and OSQP contributors
% SPDX-License-Identifier: Apache-2.0
