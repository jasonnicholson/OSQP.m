classdef test_interface < matlab.unittest.TestCase
% TEST_INTERFACE  Tests for error handling and basic interface.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    methods (Test)
        function test_solve_without_setup(tc)
            solver = OSQP();
            tc.verifyError(@() solver.solve(), 'OSQP:solve');
        end

        function test_default_settings(tc)
            s = OSQP.default_settings();
            tc.verifyTrue(isstruct(s));
            tc.verifyTrue(isfield(s, 'eps_abs'));
            tc.verifyTrue(isfield(s, 'max_iter'));
        end

        function test_version(tc)
            v = OSQP.version();
            tc.verifyTrue(ischar(v) || isstring(v));
        end

        function test_get_dimensions(tc)
            P = sparse([11 0; 0 0]);
            q = [3; 4];
            A = sparse([-1 0; 0 -1; -1 -3; 2 5; 3 4]);
            u = [0; 0; -15; 100; 80];
            l = -inf(5, 1);
            solver = OSQP();
            solver.setup(P, q, A, l, u, 'verbose', false);
            [n, m] = solver.get_dimensions();
            tc.verifyEqual(n, 2);
            tc.verifyEqual(m, 5);
        end
    end
end
