classdef test_warm_start < matlab.unittest.TestCase
% TEST_WARM_START  Tests for warm starting.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-5
    end

    methods (Test)
        function test_warm_start_problem(tc)
            rng(1);
            n = 100;
            m = 200;
            P = sprandn(n, n, 0.9);
            P = P' * P;
            q = randn(n, 1);
            A = sprandn(m, n, 0.9);
            u = rand(m, 1) * 2;
            l = -rand(m, 1) * 2;

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'adaptive_rho', false, 'check_termination', 1);
            results = solver.solve();
            x_opt = results.x;
            y_opt = results.y;
            tot_iter = results.info.iter;

            % Warm start with zeros — should take same number of iterations
            solver.warm_start('x', zeros(n, 1), 'y', zeros(m, 1));
            results2 = solver.solve();
            tc.verifyEqual(results2.info.iter, tot_iter);

            % Warm start with optimum — should converge very quickly (< 10 iters)
            solver.warm_start('x', x_opt, 'y', y_opt);
            results3 = solver.solve();
            tc.verifyLessThanOrEqual(results3.info.iter, 10);
        end
    end
end
