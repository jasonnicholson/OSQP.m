classdef test_basic < matlab.unittest.TestCase
% TEST_BASIC  Tests for basic QP solve and updates.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        P
        q
        A
        u
        l
        m
        n
        solver
        tol = 1e-4
    end

    methods (TestMethodSetup)
        function setup_problem(tc)
            tc.P = sparse([11 0; 0 0]);
            tc.q = [3; 4];
            tc.A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            tc.u = [0; 0; -15; 100; 80];
            tc.l = -1e30 * ones(5, 1);
            tc.n = 2;
            tc.m = 5;
            tc.solver = OSQP();
            tc.solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'eps_abs', 1e-9, 'eps_rel', 1e-9, ...
                'verbose', false, 'adaptive_rho', false, ...
                'check_termination', 1, 'max_iter', 4000, ...
                'rho', 0.1, 'warm_start', true);
        end
    end

    methods (Test)
        function test_basic_QP(tc)
            results = tc.solver.solve();
            tc.verifyEqual(results.x, [0; 5], 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, [1.6667; 0; 1.3333; 0; 0], 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, 20.0, 'AbsTol', tc.tol);
        end

        function test_update_q(tc)
            tc.solver.update('q', [10; 20]);
            results = tc.solver.solve();
            tc.verifyEqual(results.x, [0; 5], 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, [3.3333; 0; 6.6667; 0; 0], 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, 100.0, 'AbsTol', tc.tol);
        end

        function test_update_l(tc)
            tc.solver.update('l', -100 * ones(tc.m, 1));
            results = tc.solver.solve();
            tc.verifyEqual(results.x, [0; 5], 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, [1.6667; 0; 1.3333; 0; 0], 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, 20.0, 'AbsTol', tc.tol);
        end

        function test_update_u(tc)
            tc.solver.update('u', 1000 * ones(tc.m, 1));
            results = tc.solver.solve();
            tc.verifyEqual(results.x, [-0.1515; -333.2828], 'AbsTol', 1e-3);
            tc.verifyEqual(results.info.obj_val, -1333.4596, 'AbsTol', 1);
        end

        function test_update_max_iter(tc)
            tc.solver.update_settings('max_iter', 80);
            results = tc.solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_MAX_ITER_REACHED);
        end

        function test_update_check_termination(tc)
            max_iter = 4000;
            tc.solver.update_settings('check_termination', false, 'max_iter', max_iter);
            results = tc.solver.solve();
            tc.verifyEqual(results.info.iter, max_iter);
        end

        function test_update_rho(tc)
            results_default = tc.solver.solve();

            % Create solver with different rho, then update back
            solver2 = OSQP();
            solver2.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'eps_abs', 1e-9, 'eps_rel', 1e-9, ...
                'verbose', false, 'adaptive_rho', false, ...
                'check_termination', 1, 'max_iter', 4000, ...
                'rho', 0.7, 'warm_start', true);
            solver2.update_settings('rho', 0.1);
            results_new_rho = solver2.solve();
            tc.verifyEqual(results_default.info.iter, results_new_rho.info.iter);
        end

        function test_time_limit(tc)
            results = tc.solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);

            tc.solver.update_settings(...
                'eps_abs', 1e-20, 'eps_rel', 1e-20, ...
                'time_limit', 1e-6, 'max_iter', 1000000, ...
                'check_termination', 0);
            results_tl = tc.solver.solve();
            tc.verifyEqual(results_tl.info.status_val, OSQP.STATUS_TIME_LIMIT_REACHED);
        end
    end
end
