classdef test_unconstrained < matlab.unittest.TestCase
% TEST_UNCONSTRAINED  Tests for unconstrained QP problems.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-5
    end

    methods (Test)
        function test_unconstrained_problem(tc)
            rng(1);
            n = 30;
            P = sparse(diag(rand(n, 1)) + 0.2 * speye(n));
            q = randn(n, 1);
            A = sparse(0, n);
            l = zeros(0, 1);
            u = zeros(0, 1);

            Pfull = P + P' - diag(diag(P));
            x_test = -(Pfull \ q);
            obj_test = -0.5 * q' * (Pfull \ q);

            solver = OSQP();
            solver.setup(P, q, A, l, u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'eps_dual_inf', 1e-18);
            results = solver.solve();

            tc.verifyEqual(results.x, x_test, 'AbsTol', tc.tol);
            tc.verifyEqual(results.y, zeros(0, 1), 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.obj_val, obj_test, 'AbsTol', tc.tol);
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end
    end
end
