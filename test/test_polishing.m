classdef test_polishing < matlab.unittest.TestCase
% TEST_POLISHING  Tests for solution polishing.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-3
    end

    methods (Test)
        function test_polishing_problem(tc)
            P = sparse(diag([11; 0]));
            q = [3; 4];
            A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            u = [0; 0; -15; 100; 80];
            l = -inf(5, 1);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'polish', true, ...
                'eps_abs', 1e-3, 'eps_rel', 1e-3, 'max_iter', 5000);
            results = solver.solve();

            tc.verifyEqual(results.x, [0; 5], 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, [1.6667; 0; 1.3333; 0; 0], 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, 20.0, 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.status_polish, 1);
        end

        function test_polishing_unconstrained(tc)
            rng(1);
            n = 10;
            m = n;
            P = sparse(diag(rand(n, 1)) + 0.2 * speye(n));
            q = randn(n, 1);
            A = speye(n);
            l = -100 * ones(m, 1);
            u = 100 * ones(m, 1);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'polish', true, ...
                'eps_abs', 1e-3, 'eps_rel', 1e-3, 'max_iter', 5000);
            results = solver.solve();

            Pfull = P + P' - diag(diag(P));
            x_test = -(Pfull \ q);
            obj_test = -0.5 * q' * (Pfull \ q);

            tc.verifyEqual(results.x, x_test, 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, zeros(m, 1), 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, obj_test, 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.status_polish, 1);
        end

        function test_polish_random(tc)
            data = load(fullfile(fileparts(mfilename('fullpath')), ...
                'problem_data', 'random_polish_qp.mat'));
            P = sparse(data.P);
            q = data.q(:);
            A = sparse(data.A);
            u = data.u(:);
            l = data.l(:);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'polish', true, ...
                'eps_abs', 1e-3, 'eps_rel', 1e-3, 'max_iter', 5000);
            results = solver.solve();

            tc.verifyEqual(results.x, data.x_test(:), 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, data.y_test(:), 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, data.obj_test, 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.status_polish, 1);
        end
    end
end
