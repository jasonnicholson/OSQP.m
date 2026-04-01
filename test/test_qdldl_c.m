classdef test_qdldl_c < matlab.unittest.TestCase
% TEST_QDLDL_C  Backend regression tests for the QDLDL C MEX integration.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    methods (TestClassSetup)
        function ensure_mex(tc)
            if exist(['qdldl_c_factor_mex.' mexext], 'file') ~= 3
                build_qdldl_c_mex();
            end
            tc.assumeEqual(exist(['qdldl_c_factor_mex.' mexext], 'file'), 3);
        end
    end

    methods (Test)
        function test_basic_solve_matches_matlab_ldl(tc)
            P = sparse([11 0; 0 0]);
            q = [3; 4];
            A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            u = [0; 0; -15; 100; 80];
            l = -1e30 * ones(5, 1);

            solver_c = OSQP();
            solver_c.setup(P, q, A, l, u, ...
                'eps_abs', 1e-9, 'eps_rel', 1e-9, ...
                'verbose', false, 'adaptive_rho', false, ...
                'check_termination', 1, 'max_iter', 4000, ...
                'rho', 0.1, 'warm_start', true, ...
                'linear_solver', 'qdldl_c');

            solver_ref = OSQP();
            solver_ref.setup(P, q, A, l, u, ...
                'eps_abs', 1e-9, 'eps_rel', 1e-9, ...
                'verbose', false, 'adaptive_rho', false, ...
                'check_termination', 1, 'max_iter', 4000, ...
                'rho', 0.1, 'warm_start', true, ...
                'linear_solver', 'matlab_ldl');

            res_c = solver_c.solve();
            res_ref = solver_ref.solve();

            tc.verifyEqual(res_c.x, res_ref.x, 'AbsTol', 1e-7);
            tc.verifyEqual(res_c.y, res_ref.y, 'AbsTol', 1e-7);
            tc.verifyEqual(res_c.info.obj_val, res_ref.info.obj_val, 'AbsTol', 1e-8);
        end

        function test_update_path(tc)
            P = sparse([4, 1; 1, 2]);
            q = [1; 1];
            A = sparse([1, 1; 1, 0; 0, 1]);
            l = [1; 0; 0];
            u = [1; 0.7; 0.7];

            solver = OSQP();
            solver.setup(P, q, A, l, u, 'verbose', false, 'linear_solver', 'qdldl_c');
            res1 = solver.solve();
            tc.verifyEqual(res1.info.status_val, OSQP.STATUS_SOLVED);

            solver.update('q', [2; 3], 'l', [2; -1; -1], 'u', [2; 2.5; 2.5]);
            res2 = solver.solve();
            tc.verifyEqual(res2.info.status_val, OSQP.STATUS_SOLVED);
            tc.verifyEqual(res2.x, [0.74995267; 1.24990098], 'AbsTol', 2e-3);
        end
    end
end