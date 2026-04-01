classdef test_dual_infeasibility < matlab.unittest.TestCase
% TEST_DUAL_INFEASIBILITY  Tests for dual infeasible QPs.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-5
    end

    methods (Test)
        function test_dual_infeasible_lp(tc)
            P = sparse(2, 2);
            q = [2; -1];
            A = speye(2);
            u = inf(2, 1);
            l = [0; 0];

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-5, 'eps_rel', 1e-5, ...
                'eps_prim_inf', 1e-15, 'check_termination', 1);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_DUAL_INFEASIBLE);
        end

        function test_dual_infeasible_qp(tc)
            P = sparse(diag([4; 0]));
            q = [0; 2];
            A = sparse([1 1; -1 1]);
            u = [2; 3];
            l = -inf(2, 1);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-5, 'eps_rel', 1e-5, ...
                'eps_prim_inf', 1e-15, 'check_termination', 1);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_DUAL_INFEASIBLE);
        end

        function test_primal_dual_infeasible(tc)
            P = sparse(2, 2);
            q = [-1; -1];
            A = sparse([1 -1; -1 1; 1 0; 0 1]);
            u = inf(4, 1);
            l = [1; 1; 0; 0];

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-5, 'eps_rel', 1e-5, ...
                'eps_prim_inf', 1e-15, 'check_termination', 1);
            solver.warm_start('x', [50; 30], 'y', [-2; -2; -2; -2]);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_DUAL_INFEASIBLE);
        end
    end
end
