% smoke_pardiso.m — Smoke test for the rewritten pardiso_mkl_mex
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'src'));

% Small symmetric indefinite matrix (like a KKT system)
K = sparse([4 1 0; 1 -2 3; 0 3 5]);
b = [1; 2; 3];
x_ref = full(K \ b);
fprintf('Reference computed OK\n');

% Test the MEX
h = pardiso_mkl_mex('factorize', tril(K));
fprintf('Factorize OK, h = %g\n', h);

x = pardiso_mkl_mex('solve', h, b);
fprintf('Solve OK\n');

pardiso_mkl_mex('free', h);
fprintf('Free OK\n');

fprintf('Reference: [%g, %g, %g]\n', x_ref);
fprintf('Pardiso:   [%g, %g, %g]\n', x);
fprintf('Max error: %e\n', max(abs(x - x_ref)));

if max(abs(x - x_ref)) < 1e-10
    fprintf('PASS\n');
else
    error('FAIL: solution mismatch');
end
