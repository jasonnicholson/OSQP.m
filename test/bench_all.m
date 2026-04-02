% bench_all.m — Compare available OSQP linear solver backends.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'src'), fullfile(repoRoot, 'qdldl', 'src'));

nrep  = 20;
nrep5 = 10;
nrep8 = 10;

configs = {'qdldl', 'matlab_ldl'};
labels  = {'Optimised QDLDL', 'MATLAB ldl()'};
if exist(['qdldl_c_factor_mex.' mexext], 'file') == 3
    configs = {'qdldl', 'qdldl_c', 'matlab_ldl'};
    labels  = {'Optimised QDLDL', 'QDLDL C MEX', 'MATLAB ldl()'};
end
if exist(['pardiso_mkl_mex.' mexext], 'file') == 3
    if exist(['qdldl_c_factor_mex.' mexext], 'file') == 3
        configs = {'qdldl', 'qdldl_c', 'pardiso_mkl', 'matlab_ldl'};
        labels  = {'Optimised QDLDL', 'QDLDL C MEX', 'Pardiso MKL', 'MATLAB ldl()'};
    else
        configs = {'qdldl', 'pardiso_mkl', 'matlab_ldl'};
        labels  = {'Optimised QDLDL', 'Pardiso MKL', 'MATLAB ldl()'};
    end
end
nc = numel(configs);

total_ms = zeros(1, nc);

fprintf('\n================================================\n');
fprintf('  OSQP.m — Linear Solver Speed Comparison\n');
fprintf('================================================\n');
fprintf('  %-35s', 'Example');
for c = 1:nc
    fprintf('  %15s', labels{c});
end
fprintf('\n  %s\n', repmat('-', 1, 35 + 17*nc));

%% Helper
function med = bench(fn, n)
    t = zeros(n,1);
    for r = 1:n; tic; fn(); t(r) = toc; end
    med = median(t)*1000;
end

%% ---------- Ex1: Setup+Solve 2x2 ----------
P = sparse([4,1;1,2]); q=[1;1]; A=sparse([1,1;1,0;0,1]); l=[1;0;0]; u=[1;0.7;0.7];
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_ex1(P,q,A,l,u,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex1: Setup+Solve (2×2)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex2: Update Vectors ----------
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_ex2(P,q,A,l,u,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex2: Update Vectors');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex3: Update Matrices ----------
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_ex3(P,q,A,l,u,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex3: Update Matrices');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex4: Least Squares 50-var ----------
rng(1); m4=30; n4=20;
Ad4=sprandn(m4,n4,0.7); b4=randn(m4,1);
P4=blkdiag(sparse(n4,n4),speye(m4)); q4=zeros(n4+m4,1);
A4=[Ad4,-speye(m4);speye(n4),sparse(n4,m4)];
l4=[b4;zeros(n4,1)]; u4=[b4;ones(n4,1)];
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_ex4(P4,q4,A4,l4,u4,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex4: Least Squares (50 vars)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex5: MPC 172-var 15 steps ----------
[P5,q5,A5,l5,u5,nx5,nu5,N5,Ad5,Bd5] = build_mpc_data();
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_ex5(P5,q5,A5,l5,u5,nx5,nu5,N5,Ad5,Bd5,cfg);
    med = bench(fn, nrep5);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex5: MPC (172 vars, 15 steps)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex6: Huber 310-var ----------
rng(1); n6=10; m6=100;
Ad6=sprandn(m6,n6,0.5); x_true6=randn(n6,1)/sqrt(n6);
ind95=rand(m6,1)>0.95;
b6=Ad6*x_true6+10*rand(m6,1).*ind95+0.5*randn(m6,1).*(1-ind95);
Im=speye(m6); P6=blkdiag(sparse(n6,n6),2*Im,sparse(2*m6,2*m6));
q6=[zeros(m6+n6,1);2*ones(2*m6,1)];
A6=[Ad6,-Im,-Im,Im; sparse(m6,n6),sparse(m6,m6),Im,sparse(m6,m6);
    sparse(m6,n6),sparse(m6,m6),sparse(m6,m6),Im];
l6=[b6;zeros(2*m6,1)]; u6=[b6;inf*ones(2*m6,1)];
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_simple(P6,q6,A6,l6,u6,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex6: Huber (310 vars)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex7: SVM 1010-var ----------
rng(1); n7=10; m7=1000; N7=ceil(m7/2);
Aup=sprandn(N7,n7,0.5); Alo=sprandn(N7,n7,0.5);
Ad7=[Aup/sqrt(n7)+(Aup~=0)/n7; Alo/sqrt(n7)-(Alo~=0)/n7];
b7=[ones(N7,1);-ones(N7,1)];
P7=blkdiag(speye(n7),sparse(m7,m7)); q7=[zeros(n7,1);ones(m7,1)];
A7=[diag(b7)*Ad7,-speye(m7); sparse(m7,n7),speye(m7)];
l7=[-inf*ones(m7,1);zeros(m7,1)]; u7=[-ones(m7,1);inf*ones(m7,1)];
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_simple(P7,q7,A7,l7,u7,cfg);
    med = bench(fn, nrep);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex7: SVM (1010 vars)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Ex8: Lasso 1020-var 11 solves ----------
rng(1); n8=10; m8=1000;
Ad8=sprandn(m8,n8,0.5); x_true8=(randn(n8,1)>0.8).*randn(n8,1)/sqrt(n8);
b8=Ad8*x_true8+0.5*randn(m8,1); gammas8=linspace(1,10,11);
P8=blkdiag(sparse(n8,n8),speye(m8),sparse(n8,n8)); q8=zeros(2*n8+m8,1);
A8=[Ad8,-speye(m8),sparse(m8,n8); speye(n8),sparse(n8,m8),-speye(n8);
    speye(n8),sparse(n8,m8),speye(n8)];
l8=[b8;-inf*ones(n8,1);zeros(n8,1)]; u8=[b8;zeros(n8,1);inf*ones(n8,1)];
row = [];
for c = 1:nc
    cfg = configs{c};
    fn = @() run_lasso(P8,q8,A8,l8,u8,gammas8,n8,m8,cfg);
    med = bench(fn, nrep8);
    total_ms(c) = total_ms(c) + med;
    row(end+1) = med;
end
fprintf('  %-35s', 'Ex8: Lasso (1020v, 11 solves)');
for c = 1:nc; fprintf('  %12.2f ms', row(c)); end; fprintf('\n');

%% ---------- Summary ----------
fprintf('  %s\n', repmat('-', 1, 35 + 17*nc));
fprintf('  %-35s', 'Total (sum of medians)');
for c = 1:nc; fprintf('  %12.2f ms', total_ms(c)); end; fprintf('\n');
fprintf('  %-35s', 'vs. Python osqp 1.1.1 (C-backed)');
py_total = 55.23;
for c = 1:nc
    fprintf('  %12.1fx slower', total_ms(c)/py_total);
end
fprintf('\n================================================\n');

%% ---- Helper functions ----
function run_ex1(P,q,A,l,u,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'alpha',1.0,'verbose',false,'linear_solver',cfg); p.solve();
end
function run_ex2(P,q,A,l,u,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'verbose',false,'linear_solver',cfg); p.solve();
    p.update('q',[2;3],'l',[2;-1;-1],'u',[2;2.5;2.5]); p.solve();
end
function run_ex3(P,q,A,l,u,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'verbose',false,'linear_solver',cfg); p.solve();
    Pn=sparse([5,1.5;1.5,1]); An=sparse([1.2,1.1;1.5,0;0,0.8]);
    p.update('Px',nonzeros(triu(Pn)),'Ax',nonzeros(An)); p.solve();
end
function run_ex4(P,q,A,l,u,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'verbose',false,'linear_solver',cfg); p.solve();
end
function run_ex5(P,q,A,l,u,nx,nu,N,Ad,Bd,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'warm_start',true,'verbose',false,'linear_solver',cfg);
    x0=zeros(12,1); lt=l; ut=u;
    for i=1:15
        r=p.solve(); ctrl=r.x((N+1)*nx+1:(N+1)*nx+nu);
        x0=Ad*x0+Bd*ctrl; lt(1:nx)=-x0; ut(1:nx)=-x0; p.update('l',lt,'u',ut);
    end
end
function run_simple(P,q,A,l,u,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'verbose',false,'linear_solver',cfg); p.solve();
end
function run_lasso(P,q,A,l,u,gammas,n,m,cfg)
    p=OSQP(); p.setup(P,q,A,l,u,'warm_start',true,'verbose',false,'linear_solver',cfg);
    for i=1:length(gammas)
        p.update('q',[zeros(n+m,1);gammas(i)*ones(n,1)]); p.solve();
    end
end
function [P,q,A,l,u,nx,nu,N,Ad,Bd] = build_mpc_data()
    Ad=[1 0 0 0 0 0 0.1 0 0 0 0 0; 0 1 0 0 0 0 0 0.1 0 0 0 0;
        0 0 1 0 0 0 0 0 0.1 0 0 0; 0.0488 0 0 1 0 0 0.0016 0 0 0.0992 0 0;
        0 -0.0488 0 0 1 0 0 -0.0016 0 0 0.0992 0; 0 0 0 0 0 1 0 0 0 0 0 0.0992;
        0 0 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0 0; 0.9734 0 0 0 0 0 0.0488 0 0 0.9846 0 0;
        0 -0.9734 0 0 0 0 0 -0.0488 0 0 0.9846 0; 0 0 0 0 0 0 0 0 0 0 0 0.9846];
    Bd=[0 -0.0726 0 0.0726; -0.0726 0 0.0726 0; -0.0152 0.0152 -0.0152 0.0152;
        0 -0.0006 0 0.0006; 0.0006 0 -0.0006 0; 0.0106 0.0106 0.0106 0.0106;
        0 -1.4512 0 1.4512; -1.4512 0 1.4512 0; -0.3049 0.3049 -0.3049 0.3049;
        0 -0.0236 0 0.0236; 0.0236 0 -0.0236 0; 0.2107 0.2107 0.2107 0.2107];
    nx=12; nu=4; N=10; u0=10.5916;
    umin=[9.6;9.6;9.6;9.6]-u0; umax=[13;13;13;13]-u0;
    xmin=[-pi/6;-pi/6;-Inf;-Inf;-Inf;-1;-Inf(6,1)]; xmax=[pi/6;pi/6;Inf;Inf;Inf;Inf;Inf(6,1)];
    Q=diag([0 0 10 10 10 10 0 0 0 5 5 5]); R=0.1*eye(4);
    xr=[0;0;1;0;0;0;0;0;0;0;0;0];
    P=blkdiag(kron(speye(N),Q),Q,kron(speye(N),R));
    q=[repmat(-Q*xr,N,1);-Q*xr;zeros(N*nu,1)];
    Ax=kron(speye(N+1),-speye(nx))+kron(sparse(diag(ones(N,1),-1)),Ad);
    Bu=kron([sparse(1,N);speye(N)],Bd);
    Aeq=[Ax,Bu]; x0=zeros(nx,1);
    leq=[-x0;zeros(N*nx,1)]; ueq=leq;
    Aineq=speye((N+1)*nx+N*nu);
    lineq=[repmat(xmin,N+1,1);repmat(umin,N,1)];
    uineq=[repmat(xmax,N+1,1);repmat(umax,N,1)];
    A=[Aeq;Aineq]; l=[leq;lineq]; u=[ueq;uineq];
end
