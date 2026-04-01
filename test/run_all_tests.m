function run_all_tests()
% RUN_ALL_TESTS  Run all OSQP MATLAB unit tests.
%
%   run_all_tests()  runs the full test suite and prints a summary.
%
% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

% Make sure the solver and qdldl are on the path
setupPath(true);

testFiles = {
    'test_interface'
    'test_basic'
    'test_unconstrained'
    'test_feasibility'
    'test_primal_infeasibility'
    'test_dual_infeasibility'
    'test_non_convex'
    'test_warm_start'
    'test_polishing'
    'test_update_matrices'
};

if exist(['qdldl_c_factor_mex.' mexext], 'file') == 3
    testFiles{end + 1} = 'test_qdldl_c';
end

results = matlab.unittest.TestResult.empty;

for k = 1:numel(testFiles)
    suite = matlab.unittest.TestSuite.fromClass(meta.class.fromName(testFiles{k}));
    r = run(suite);
    results = [results, r]; %#ok<AGROW>
end

% Print summary
fprintf('\n=== OSQP Test Summary ===\n');
fprintf('Passed: %d / %d\n', sum([results.Passed]), numel(results));
if any(~[results.Passed])
    fprintf('FAILED tests:\n');
    for k = 1:numel(results)
        if ~results(k).Passed
            fprintf('  %s\n', results(k).Name);
        end
    end
end
end
