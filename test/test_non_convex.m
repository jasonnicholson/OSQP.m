classdef test_non_convex < matlab.unittest.TestCase
% TEST_NON_CONVEX  Tests that non-convex QPs are rejected / flagged.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    methods (Test)
        function test_non_convex_small_sigma(tc)
            P = sparse([2 5; 5 1]);
            q = [3; 4];
            A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            u = [0; 0; -15; 100; 80];
            l = -inf(5, 1);

            solver = OSQP();
            % Setting sigma small so P + sigma*I has negative eigenvalue
            % setup() should throw
            tc.verifyError(@() solver.setup(P, q, A, l, u, ...
                'verbose', false, 'sigma', 1e-6), ...
                'OSQP:NonConvex');
        end

        function test_non_convex_big_sigma(tc)
            P = sparse([2 5; 5 1]);
            q = [3; 4];
            A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            u = [0; 0; -15; 100; 80];
            l = -inf(5, 1);

            solver = OSQP();
            % With sigma=5: P + 5*I = [7 5; 5 6] which is PD
            solver.setup(P, q, A, l, u, 'verbose', false, 'sigma', 5.0);
            results = solver.solve();
            tc.verifyTrue(isnan(results.info.obj_val));
            tc.verifyEqual(results.info.status_val, OSQP.STATUS_NON_CONVEX);
        end
    end
end
