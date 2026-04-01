classdef test_primal_infeasibility < matlab.unittest.TestCase
% TEST_PRIMAL_INFEASIBILITY  Tests for primal infeasible QPs.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    methods (Test)
        function test_primal_infeasible_problem(tc)
            rng(1);
            n = 50;
            m = 500;
            P = sprandn(n, n, 0.6);
            P = P' * P;
            q = randn(n, 1);
            A = sprandn(m, n, 0.6);
            u = 3 + randn(m, 1);
            l = -3 + randn(m, 1);

            % Make problem infeasible
            half = floor(n / 2);
            A(half, :) = A(half + 1, :);
            l(half) = u(half + 1) + 10 * rand();
            u(half) = l(half) + 0.5;

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-5, 'eps_rel', 1e-5, ...
                'eps_dual_inf', 1e-18, 'scaling', true);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_PRIMAL_INFEASIBLE);
        end

        function test_primal_dual_infeasible_problem(tc)
            n = 2;
            P = sparse(n, n);
            q = [-1; -1];
            A = sparse([1 -1; -1 1; 1 0; 0 1]);
            l = [1; 1; 0; 0];
            u = inf(4, 1);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-5, 'eps_rel', 1e-5, ...
                'eps_dual_inf', 1e-18, 'scaling', true);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_PRIMAL_INFEASIBLE);
        end
    end
end
