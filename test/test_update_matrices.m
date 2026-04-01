classdef test_update_matrices < matlab.unittest.TestCase
% TEST_UPDATE_MATRICES  Tests for updating P and A matrices.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties
        tol = 1e-4
        P
        P_new
        q
        A
        A_new
        l
        u
        n
        m
    end

    methods (TestMethodSetup)
        function setup_data(tc)
            rng(1);
            tc.n = 5;
            tc.m = 8;
            p = 0.7;
            Pt = sprandn(tc.n, tc.n, p);
            tc.P = Pt * Pt' + speye(tc.n);
            tc.q = randn(tc.n, 1);

            Pt_new = Pt;
            Pt_newx = nonzeros(Pt) + 0.1 * randn(nnz(Pt), 1);
            [ri, ci] = find(Pt_new);
            Pt_new = sparse(ri, ci, Pt_newx, tc.n, tc.n);
            tc.P_new = Pt_new * Pt_new' + speye(tc.n);

            tc.A = sprandn(tc.m, tc.n, p);
            [Ai, Aj, Ax] = find(tc.A);
            tc.A_new = sparse(Ai, Aj, Ax + randn(numel(Ax), 1), tc.m, tc.n);

            tc.l = zeros(tc.m, 1);
            tc.u = 30 + randn(tc.m, 1);
        end
    end

    methods (Test)
        function test_update_P(tc)
            solver = OSQP();
            solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'check_termination', 1);

            % Update P (all nonzero values)
            Pnew_triu = triu(tc.P_new);
            solver.update('Px', nonzeros(Pnew_triu));
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end

        function test_update_P_indexed(tc)
            solver = OSQP();
            solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'check_termination', 1);

            Pnew_triu = triu(tc.P_new);
            nnz_P = nnz(triu(tc.P));
            idx = (1:nnz_P)';
            solver.update('Px', nonzeros(Pnew_triu), 'Px_idx', idx);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end

        function test_update_A(tc)
            solver = OSQP();
            solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'check_termination', 1);

            solver.update('Ax', nonzeros(tc.A_new));
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end

        function test_update_A_indexed(tc)
            solver = OSQP();
            solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'check_termination', 1);

            nnz_A = nnz(tc.A);
            idx = (1:nnz_A)';
            solver.update('Ax', nonzeros(tc.A_new), 'Ax_idx', idx);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end

        function test_update_P_and_A(tc)
            solver = OSQP();
            solver.setup(tc.P, tc.q, tc.A, tc.l, tc.u, ...
                'verbose', false, 'eps_abs', 1e-8, 'eps_rel', 1e-8, ...
                'polish', false, 'check_termination', 1);

            Pnew_triu = triu(tc.P_new);
            nnz_A = nnz(tc.A);
            Px_idx = (1:nnz(Pnew_triu))';
            Ax_idx = (1:nnz_A)';
            solver.update('Px', nonzeros(Pnew_triu), 'Px_idx', Px_idx, ...
                          'Ax', nonzeros(tc.A_new), 'Ax_idx', Ax_idx);
            results = solver.solve();
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_SOLVED);
        end
    end
end
