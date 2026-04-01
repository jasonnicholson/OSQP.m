% run_examples.m  — Solve OSQP website and application examples using OSQP.m
%                    and compare results with Python osqp reference.
%
% Run from the OSQP.m project root:
%   matlab -batch "run('test/run_examples.m')"

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'src'), fullfile(repoRoot, 'qdldl', 'src'));

if ~exist('linear_solver', 'var') || isempty(linear_solver)
    linear_solver = 'matlab_ldl';
end

tol = 2e-3;  % tolerance for comparing solutions (ADMM default eps=1e-3)
all_pass = true;

fprintf('\n========================================\n');
fprintf('  OSQP.m Examples — Comparison Tests\n');
fprintf('========================================\n\n');

%% ============================================================
%% Example 1: Setup and Solve (osqp.org demo)
%% ============================================================
fprintf('--- Example 1: Setup and Solve ---\n');
P = sparse([4, 1; 1, 2]);
q = [1; 1];
A = sparse([1, 1; 1, 0; 0, 1]);
l = [1; 0; 0];
u = [1; 0.7; 0.7];

prob = OSQP();
prob.setup(P, q, A, l, u, 'alpha', 1.0, 'verbose', false, 'linear_solver', linear_solver);
res = prob.solve();

ref_x = [0.29877108; 0.70122892];
ref_obj = 1.879755;

fprintf('  Status : %s\n', res.info.status);
fprintf('  x      : [%.6f, %.6f]\n', res.x(1), res.x(2));
fprintf('  obj    : %.6f  (ref: %.6f)\n', res.info.obj_val, ref_obj);
fprintf('  x_err  : %.2e\n', norm(res.x - ref_x));

if norm(res.x - ref_x) > tol || abs(res.info.obj_val - ref_obj) > tol
    fprintf('  ** MISMATCH **\n');
    all_pass = false;
else
    fprintf('  PASS\n');
end
fprintf('\n');

%% ============================================================
%% Example 2: Update Vectors
%% ============================================================
fprintf('--- Example 2: Update Vectors ---\n');
prob2 = OSQP();
prob2.setup(P, q, A, l, u, 'verbose', false, 'linear_solver', linear_solver);
res2a = prob2.solve();

fprintf('  Before update: x=[%.6f, %.6f], obj=%.6f\n', ...
    res2a.x(1), res2a.x(2), res2a.info.obj_val);

q_new = [2; 3];
l_new = [2; -1; -1];
u_new = [2; 2.5; 2.5];
prob2.update('q', q_new, 'l', l_new, 'u', u_new);
res2b = prob2.solve();

ref_x2 = [0.74995267; 1.24990098];
ref_obj2 = 8.874085;

fprintf('  After update : x=[%.6f, %.6f], obj=%.6f\n', ...
    res2b.x(1), res2b.x(2), res2b.info.obj_val);
fprintf('  x_err  : %.2e\n', norm(res2b.x - ref_x2));

if norm(res2b.x - ref_x2) > tol || abs(res2b.info.obj_val - ref_obj2) > tol
    fprintf('  ** MISMATCH **\n');
    all_pass = false;
else
    fprintf('  PASS\n');
end
fprintf('\n');

%% ============================================================
%% Example 3: Update Matrices
%% ============================================================
fprintf('--- Example 3: Update Matrices ---\n');
prob3 = OSQP();
prob3.setup(P, q, A, l, u, 'verbose', false, 'linear_solver', linear_solver);
res3a = prob3.solve();

P_new = sparse([5, 1.5; 1.5, 1]);
A_new = sparse([1.2, 1.1; 1.5, 0; 0, 0.8]);
prob3.update('Px', nonzeros(triu(P_new)), 'Ax', nonzeros(A_new));
res3b = prob3.solve();

ref_x3 = [0.03125036; 0.87499961];
ref_obj3 = 1.332520;

fprintf('  x      : [%.6f, %.6f]\n', res3b.x(1), res3b.x(2));
fprintf('  obj    : %.6f  (ref: %.6f)\n', res3b.info.obj_val, ref_obj3);
fprintf('  x_err  : %.2e\n', norm(res3b.x - ref_x3));

if norm(res3b.x - ref_x3) > tol || abs(res3b.info.obj_val - ref_obj3) > tol
    fprintf('  ** MISMATCH **\n');
    all_pass = false;
else
    fprintf('  PASS\n');
end
fprintf('\n');

%% ============================================================
%% Example 4: Least Squares (from OSQP-super-project)
%% ============================================================
fprintf('--- Example 4: Least Squares ---\n');
rng(1);
m_ls = 30;
n_ls = 20;
Ad_ls = sprandn(m_ls, n_ls, 0.7);
b_ls = randn(m_ls, 1);

P_ls = blkdiag(sparse(n_ls, n_ls), speye(m_ls));
q_ls = zeros(n_ls + m_ls, 1);
A_ls = [Ad_ls, -speye(m_ls);
        speye(n_ls), sparse(n_ls, m_ls)];
l_ls = [b_ls; zeros(n_ls, 1)];
u_ls = [b_ls; ones(n_ls, 1)];

prob4 = OSQP();
prob4.setup(P_ls, q_ls, A_ls, l_ls, u_ls, 'verbose', false, 'linear_solver', linear_solver);
res4 = prob4.solve();

fprintf('  Status : %s\n', res4.info.status);
fprintf('  obj    : %.6f\n', res4.info.obj_val);
fprintf('  (Note: RNG differs from Python so obj value will differ)\n');

if ~strcmp(res4.info.status, 'solved')
    fprintf('  ** FAILED (not solved) **\n');
    all_pass = false;
else
    fprintf('  PASS (solved successfully)\n');
end
fprintf('\n');

%% ============================================================
%% Example 5: MPC (Quadcopter)
%% ============================================================
fprintf('--- Example 5: MPC (Quadcopter) ---\n');
Ad_mpc = [1       0       0   0   0   0   0.1     0       0    0       0       0;
          0       1       0   0   0   0   0       0.1     0    0       0       0;
          0       0       1   0   0   0   0       0       0.1  0       0       0;
          0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
          0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
          0       0       0   0   0   1   0       0       0    0       0       0.0992;
          0       0       0   0   0   0   1       0       0    0       0       0;
          0       0       0   0   0   0   0       1       0    0       0       0;
          0       0       0   0   0   0   0       0       1    0       0       0;
          0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
          0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
          0       0       0   0   0   0   0       0       0    0       0       0.9846];
Bd_mpc = [0      -0.0726  0       0.0726;
         -0.0726  0       0.0726  0;
         -0.0152  0.0152 -0.0152  0.0152;
          0      -0.0006 -0.0000  0.0006;
          0.0006  0      -0.0006  0;
          0.0106  0.0106  0.0106  0.0106;
          0      -1.4512  0       1.4512;
         -1.4512  0       1.4512  0;
         -0.3049  0.3049 -0.3049  0.3049;
          0      -0.0236  0       0.0236;
          0.0236  0      -0.0236  0;
          0.2107  0.2107  0.2107  0.2107];
[nx, nu] = size(Bd_mpc);

u0 = 10.5916;
umin = [9.6; 9.6; 9.6; 9.6] - u0;
umax = [13; 13; 13; 13] - u0;
xmin = [-pi/6; -pi/6; -Inf; -Inf; -Inf; -1; -Inf(6,1)];
xmax = [ pi/6;  pi/6;  Inf;  Inf;  Inf; Inf; Inf(6,1)];

Q = diag([0 0 10 10 10 10 0 0 0 5 5 5]);
QN = Q;
R = 0.1*eye(4);

x0_mpc = zeros(12,1);
xr = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];

N_mpc = 10;

P_mpc = blkdiag( kron(speye(N_mpc), Q), QN, kron(speye(N_mpc), R) );
q_mpc = [repmat(-Q*xr, N_mpc, 1); -QN*xr; zeros(N_mpc*nu, 1)];

Ax_mpc = kron(speye(N_mpc+1), -speye(nx)) + kron(sparse(diag(ones(N_mpc, 1), -1)), Ad_mpc);
Bu_mpc = kron([sparse(1, N_mpc); speye(N_mpc)], Bd_mpc);
Aeq = [Ax_mpc, Bu_mpc];
leq = [-x0_mpc; zeros(N_mpc*nx, 1)];
ueq = leq;

Aineq = speye((N_mpc+1)*nx + N_mpc*nu);
lineq = [repmat(xmin, N_mpc+1, 1); repmat(umin, N_mpc, 1)];
uineq = [repmat(xmax, N_mpc+1, 1); repmat(umax, N_mpc, 1)];

A_mpc_full = [Aeq; Aineq];
l_mpc = [leq; lineq];
u_mpc = [ueq; uineq];

prob5 = OSQP();
prob5.setup(P_mpc, q_mpc, A_mpc_full, l_mpc, u_mpc, 'warm_start', true, 'verbose', false, 'linear_solver', linear_solver);

nsim = 15;
x0_sim = zeros(12, 1);
mpc_ok = true;
for i = 1:nsim
    res5 = prob5.solve();
    if ~strcmp(res5.info.status, 'solved')
        fprintf('  MPC step %d: OSQP did not solve: %s\n', i, res5.info.status);
        mpc_ok = false;
        break;
    end
    ctrl = res5.x((N_mpc+1)*nx+1:(N_mpc+1)*nx+nu);
    x0_sim = Ad_mpc*x0_sim + Bd_mpc*ctrl;
    l_mpc(1:nx) = -x0_sim;
    u_mpc(1:nx) = -x0_sim;
    prob5.update('l', l_mpc, 'u', u_mpc);
end

% Python ref: final x3 ≈ 0.999495 (target = 1.0)
ref_x3_mpc = 0.999495;
fprintf('  Status : %s (all %d steps)\n', 'solved', nsim);
fprintf('  Final x3 (altitude): %.6f  (ref: %.6f, target: 1.0)\n', x0_sim(3), ref_x3_mpc);
fprintf('  x3 error : %.2e\n', abs(x0_sim(3) - ref_x3_mpc));

if ~mpc_ok || abs(x0_sim(3) - ref_x3_mpc) > 0.01
    fprintf('  ** MISMATCH **\n');
    all_pass = false;
else
    fprintf('  PASS\n');
end
fprintf('\n');

%% ============================================================
%% Example 6: Huber Fitting (from OSQP-super-project)
%% ============================================================
fprintf('--- Example 6: Huber Fitting ---\n');
rng(1);
n_hub = 10;
m_hub = 100;
Ad_hub = sprandn(m_hub, n_hub, 0.5);
x_true = randn(n_hub, 1) / sqrt(n_hub);
ind95 = rand(m_hub, 1) > 0.95;
b_hub = Ad_hub*x_true + 10*rand(m_hub, 1).*ind95 + 0.5*randn(m_hub, 1).*(1-ind95);

Im = speye(m_hub);
Om = sparse(m_hub, m_hub);
Omn = sparse(m_hub, n_hub);
P_hub = blkdiag(sparse(n_hub, n_hub), 2*Im, sparse(2*m_hub, 2*m_hub));
q_hub = [zeros(m_hub + n_hub, 1); 2*ones(2*m_hub, 1)];
A_hub = [Ad_hub,  -Im, -Im, Im;
         Omn,  Om,  Im, Om;
         Omn,  Om,  Om, Im];
l_hub = [b_hub; zeros(2*m_hub, 1)];
u_hub = [b_hub; inf*ones(2*m_hub, 1)];

prob6 = OSQP();
prob6.setup(P_hub, q_hub, A_hub, l_hub, u_hub, 'verbose', false, 'linear_solver', linear_solver);
res6 = prob6.solve();

fprintf('  Status : %s\n', res6.info.status);
fprintf('  obj    : %.6f\n', res6.info.obj_val);

if ~strcmp(res6.info.status, 'solved')
    fprintf('  ** FAILED (not solved) **\n');
    all_pass = false;
else
    fprintf('  PASS (solved successfully)\n');
end
fprintf('\n');

%% ============================================================
%% Example 7: SVM (from OSQP-super-project)
%% ============================================================
fprintf('--- Example 7: SVM ---\n');
rng(1);
n_svm = 10;
m_svm = 1000;
N_svm = ceil(m_svm/2);
gamma_svm = 1;
A_upp = sprandn(N_svm, n_svm, 0.5);
A_low = sprandn(N_svm, n_svm, 0.5);
Ad_svm = [A_upp / sqrt(n_svm) + (A_upp ~= 0) / n_svm;
          A_low / sqrt(n_svm) - (A_low ~= 0) / n_svm];
b_svm = [ones(N_svm, 1); -ones(N_svm,1)];

P_svm = blkdiag(speye(n_svm), sparse(m_svm, m_svm));
q_svm = [zeros(n_svm,1); gamma_svm*ones(m_svm,1)];
A_svm = [diag(b_svm)*Ad_svm, -speye(m_svm);
         sparse(m_svm, n_svm), speye(m_svm)];
l_svm = [-inf*ones(m_svm, 1); zeros(m_svm, 1)];
u_svm = [-ones(m_svm, 1); inf*ones(m_svm, 1)];

prob7 = OSQP();
prob7.setup(P_svm, q_svm, A_svm, l_svm, u_svm, 'verbose', false, 'linear_solver', linear_solver);
res7 = prob7.solve();

fprintf('  Status : %s\n', res7.info.status);
fprintf('  obj    : %.6f\n', res7.info.obj_val);

if ~strcmp(res7.info.status, 'solved')
    fprintf('  ** FAILED (not solved) **\n');
    all_pass = false;
else
    fprintf('  PASS (solved successfully)\n');
end
fprintf('\n');

%% ============================================================
%% Example 8: Lasso (from OSQP-super-project)
%% ============================================================
fprintf('--- Example 8: Lasso ---\n');
rng(1);
n_la = 10;
m_la = 1000;
Ad_la = sprandn(m_la, n_la, 0.5);
x_true_la = (randn(n_la, 1) > 0.8) .* randn(n_la, 1) / sqrt(n_la);
b_la = Ad_la * x_true_la + 0.5 * randn(m_la, 1);
gammas = linspace(1, 10, 11);

P_la = blkdiag(sparse(n_la, n_la), speye(m_la), sparse(n_la, n_la));
q_la = zeros(2*n_la+m_la, 1);
A_la = [Ad_la, -speye(m_la), sparse(m_la,n_la);
        speye(n_la), sparse(n_la, m_la), -speye(n_la);
        speye(n_la), sparse(n_la, m_la), speye(n_la)];
l_la = [b_la; -inf*ones(n_la, 1); zeros(n_la, 1)];
u_la = [b_la; zeros(n_la, 1); inf*ones(n_la, 1)];

prob8 = OSQP();
prob8.setup(P_la, q_la, A_la, l_la, u_la, 'warm_start', true, 'verbose', false, 'linear_solver', linear_solver);

lasso_ok = true;
for i = 1:length(gammas)
    gamma_la = gammas(i);
    q_new_la = [zeros(n_la+m_la,1); gamma_la*ones(n_la,1)];
    prob8.update('q', q_new_la);
    res8 = prob8.solve();
    if ~strcmp(res8.info.status, 'solved')
        fprintf('  Lasso gamma=%g: %s\n', gamma_la, res8.info.status);
        lasso_ok = false;
    end
end

fprintf('  Status : %s (last solve)\n', res8.info.status);
fprintf('  obj    : %.6f (last gamma=%.1f)\n', res8.info.obj_val, gammas(end));

if ~lasso_ok
    fprintf('  ** FAILED (some gammas not solved) **\n');
    all_pass = false;
else
    fprintf('  PASS (all %d gamma values solved)\n', length(gammas));
end
fprintf('\n');

%% ============================================================
%% Summary
%% ============================================================
fprintf('========================================\n');
if all_pass
    fprintf('  ALL EXAMPLES PASSED\n');
else
    fprintf('  SOME EXAMPLES FAILED\n');
end
fprintf('========================================\n');
