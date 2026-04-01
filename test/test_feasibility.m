classdef test_feasibility < matlab.unittest.TestCase
% TEST_FEASIBILITY  Tests for feasibility problems.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-3
    end

    methods (Test)
        function test_feasibility_problem(tc)
            rng(1);
            n = 30;
            m = 30;
            P = sparse(n, n);
            q = zeros(n, 1);
            A = sprandn(m, n, 0.8);
            u = randn(m, 1);
            l = u;

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-6, 'eps_rel', 1e-6, ...
                'max_iter', 5000);
            results = solver.solve();
            tc.verifyEqual(norm(A * results.x - u), 0, 'AbsTol', tc.tol);
        end
    end
end
