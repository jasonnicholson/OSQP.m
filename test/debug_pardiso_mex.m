repoRoot = '/home/jason/Desktop/osqp_related/OSQP.m';
addpath(fullfile(repoRoot, 'src'));

clear pardiso_mkl_mex;
build_pardiso_mkl_mex(true);

P = sparse([4, 1; 1, 2]);
A = sparse([1, 1; 1, 0; 0, 1]);
sigma = 1e-6;
rho_vec = 0.1 * ones(3, 1);
Ktl = P + sigma * speye(2);
Kbr = -diag(sparse(1 ./ rho_vec));
K = [Ktl, A'; A, Kbr];
K = (K + K') / 2;

h = pardiso_mkl_mex('factorize', triu(K));
info = pardiso_mkl_mex('info', h)

b = (1:5)';
x = pardiso_mkl_mex('solve', h, b)

pardiso_mkl_mex('free', h);
