% bench_matlab.m — Time OSQP.m examples for speed comparison
%
% Run: matlab -batch "run('/home/jason/Desktop/osqp_related/OSQP.m/test/bench_matlab.m')"

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'src'), fullfile(repoRoot, 'qdldl', 'src'));

nrep = 20;  % repeats for median

fprintf('\n============================================================\n');
fprintf('  OSQP.m (pure MATLAB) — Speed Benchmark\n');
fprintf('============================================================\n\n');

total_ms = 0;

%% Ex1: Setup and Solve (2x2)
times = zeros(nrep,1);
for r = 1:nrep
    tic;
    P = sparse([4,1;1,2]); q=[1;1]; A=sparse([1,1;1,0;0,1]); l=[1;0;0]; u=[1;0.7;0.7];
    prob = OSQP(); prob.setup(P,q,A,l,u,'alpha',1.0,'verbose',false); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex1: Setup+Solve (2x2)', med, nrep);
total_ms = total_ms + med;

%% Ex2: Update Vectors
times = zeros(nrep,1);
for r = 1:nrep
    tic;
    P = sparse([4,1;1,2]); q=[1;1]; A=sparse([1,1;1,0;0,1]); l=[1;0;0]; u=[1;0.7;0.7];
    prob = OSQP(); prob.setup(P,q,A,l,u,'verbose',false); prob.solve();
    prob.update('q',[2;3],'l',[2;-1;-1],'u',[2;2.5;2.5]); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex2: Update Vectors', med, nrep);
total_ms = total_ms + med;

%% Ex3: Update Matrices
times = zeros(nrep,1);
for r = 1:nrep
    tic;
    P = sparse([4,1;1,2]); q=[1;1]; A=sparse([1,1;1,0;0,1]); l=[1;0;0]; u=[1;0.7;0.7];
    prob = OSQP(); prob.setup(P,q,A,l,u,'verbose',false); prob.solve();
    P_new = sparse([5,1.5;1.5,1]); A_new = sparse([1.2,1.1;1.5,0;0,0.8]);
    prob.update('Px',nonzeros(triu(P_new)),'Ax',nonzeros(A_new)); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex3: Update Matrices', med, nrep);
total_ms = total_ms + med;

%% Ex4: Least Squares (50 vars)
rng(1);
m4=30; n4=20;
Ad4 = sprandn(m4,n4,0.7); b4 = randn(m4,1);
P4 = blkdiag(sparse(n4,n4),speye(m4)); q4 = zeros(n4+m4,1);
A4 = [Ad4,-speye(m4); speye(n4),sparse(n4,m4)];
l4 = [b4; zeros(n4,1)]; u4 = [b4; ones(n4,1)];

times = zeros(nrep,1);
for r = 1:nrep
    tic;
    prob = OSQP(); prob.setup(P4,q4,A4,l4,u4,'verbose',false); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex4: Least Squares (50 vars)', med, nrep);
total_ms = total_ms + med;

%% Ex5: MPC (172 vars, 15 steps)
Ad5 = [1 0 0 0 0 0 0.1 0 0 0 0 0; 0 1 0 0 0 0 0 0.1 0 0 0 0;
       0 0 1 0 0 0 0 0 0.1 0 0 0; 0.0488 0 0 1 0 0 0.0016 0 0 0.0992 0 0;
       0 -0.0488 0 0 1 0 0 -0.0016 0 0 0.0992 0; 0 0 0 0 0 1 0 0 0 0 0 0.0992;
       0 0 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 1 0 0 0 0;
       0 0 0 0 0 0 0 0 1 0 0 0; 0.9734 0 0 0 0 0 0.0488 0 0 0.9846 0 0;
       0 -0.9734 0 0 0 0 0 -0.0488 0 0 0.9846 0; 0 0 0 0 0 0 0 0 0 0 0 0.9846];
Bd5 = [0 -0.0726 0 0.0726; -0.0726 0 0.0726 0; -0.0152 0.0152 -0.0152 0.0152;
       0 -0.0006 0 0.0006; 0.0006 0 -0.0006 0; 0.0106 0.0106 0.0106 0.0106;
       0 -1.4512 0 1.4512; -1.4512 0 1.4512 0; -0.3049 0.3049 -0.3049 0.3049;
       0 -0.0236 0 0.0236; 0.0236 0 -0.0236 0; 0.2107 0.2107 0.2107 0.2107];
nx5=12; nu5=4; N5=10; u0=10.5916;
umin5=[9.6;9.6;9.6;9.6]-u0; umax5=[13;13;13;13]-u0;
xmin5=[-pi/6;-pi/6;-Inf;-Inf;-Inf;-1;-Inf(6,1)];
xmax5=[pi/6;pi/6;Inf;Inf;Inf;Inf;Inf(6,1)];
Q5=diag([0 0 10 10 10 10 0 0 0 5 5 5]); R5=0.1*eye(4);
xr5=[0;0;1;0;0;0;0;0;0;0;0;0];
P5=blkdiag(kron(speye(N5),Q5),Q5,kron(speye(N5),R5));
q5=[repmat(-Q5*xr5,N5,1);-Q5*xr5;zeros(N5*nu5,1)];
Ax5=kron(speye(N5+1),-speye(nx5))+kron(sparse(diag(ones(N5,1),-1)),Ad5);
Bu5=kron([sparse(1,N5);speye(N5)],Bd5);
Aeq5=[Ax5,Bu5]; x05=zeros(nx5,1);
leq5=[-x05;zeros(N5*nx5,1)]; ueq5=leq5;
Aineq5=speye((N5+1)*nx5+N5*nu5);
lineq5=[repmat(xmin5,N5+1,1);repmat(umin5,N5,1)];
uineq5=[repmat(xmax5,N5+1,1);repmat(umax5,N5,1)];
A5=[Aeq5;Aineq5]; l5=[leq5;lineq5]; u5=[ueq5;uineq5];

nrep5 = 10;
times = zeros(nrep5,1);
for r = 1:nrep5
    tic;
    prob = OSQP(); prob.setup(P5,q5,A5,l5,u5,'warm_start',true,'verbose',false);
    x0s = zeros(12,1); l5t=l5; u5t=u5;
    for i=1:15
        res=prob.solve(); ctrl=res.x((N5+1)*nx5+1:(N5+1)*nx5+nu5);
        x0s=Ad5*x0s+Bd5*ctrl; l5t(1:nx5)=-x0s; u5t(1:nx5)=-x0s;
        prob.update('l',l5t,'u',u5t);
    end
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex5: MPC (172 vars, 15 steps)', med, nrep5);
total_ms = total_ms + med;

%% Ex6: Huber Fitting (310 vars)
rng(1);
n6=10; m6=100;
Ad6=sprandn(m6,n6,0.5); x_true6=randn(n6,1)/sqrt(n6);
ind95_6=rand(m6,1)>0.95;
b6=Ad6*x_true6+10*rand(m6,1).*ind95_6+0.5*randn(m6,1).*(1-ind95_6);
P6=blkdiag(sparse(n6,n6),2*speye(m6),sparse(2*m6,2*m6));
q6=[zeros(m6+n6,1);2*ones(2*m6,1)];
A6=[Ad6,-speye(m6),-speye(m6),speye(m6);
    sparse(m6,n6),sparse(m6,m6),speye(m6),sparse(m6,m6);
    sparse(m6,n6),sparse(m6,m6),sparse(m6,m6),speye(m6)];
l6=[b6;zeros(2*m6,1)]; u6=[b6;inf*ones(2*m6,1)];

times = zeros(nrep,1);
for r = 1:nrep
    tic;
    prob = OSQP(); prob.setup(P6,q6,A6,l6,u6,'verbose',false); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex6: Huber (310 vars)', med, nrep);
total_ms = total_ms + med;

%% Ex7: SVM (1010 vars)
rng(1);
n7=10; m7=1000; N7=ceil(m7/2);
A_upp7=sprandn(N7,n7,0.5); A_low7=sprandn(N7,n7,0.5);
Ad7=[A_upp7/sqrt(n7)+(A_upp7~=0)/n7; A_low7/sqrt(n7)-(A_low7~=0)/n7];
b7=[ones(N7,1);-ones(N7,1)];
P7=blkdiag(speye(n7),sparse(m7,m7));
q7=[zeros(n7,1);ones(m7,1)];
A7=[diag(b7)*Ad7,-speye(m7); sparse(m7,n7),speye(m7)];
l7=[-inf*ones(m7,1);zeros(m7,1)]; u7=[-ones(m7,1);inf*ones(m7,1)];

times = zeros(nrep,1);
for r = 1:nrep
    tic;
    prob = OSQP(); prob.setup(P7,q7,A7,l7,u7,'verbose',false); prob.solve();
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex7: SVM (1010 vars)', med, nrep);
total_ms = total_ms + med;

%% Ex8: Lasso (1020 vars, 11 solves)
rng(1);
n8=10; m8=1000;
Ad8=sprandn(m8,n8,0.5);
x_true8=(randn(n8,1)>0.8).*randn(n8,1)/sqrt(n8);
b8=Ad8*x_true8+0.5*randn(m8,1);
gammas8=linspace(1,10,11);
P8=blkdiag(sparse(n8,n8),speye(m8),sparse(n8,n8));
q8=zeros(2*n8+m8,1);
A8=[Ad8,-speye(m8),sparse(m8,n8); speye(n8),sparse(n8,m8),-speye(n8);
    speye(n8),sparse(n8,m8),speye(n8)];
l8=[b8;-inf*ones(n8,1);zeros(n8,1)]; u8=[b8;zeros(n8,1);inf*ones(n8,1)];

nrep8 = 10;
times = zeros(nrep8,1);
for r = 1:nrep8
    tic;
    prob = OSQP(); prob.setup(P8,q8,A8,l8,u8,'warm_start',true,'verbose',false);
    for i=1:length(gammas8)
        prob.update('q',[zeros(n8+m8,1);gammas8(i)*ones(n8,1)]); prob.solve();
    end
    times(r) = toc;
end
med = median(times)*1000;
fprintf('  %-35s %8.2f ms  (median of %d)\n', 'Ex8: Lasso (1020v, 11 solves)', med, nrep8);
total_ms = total_ms + med;

%% Summary
fprintf('\n============================================================\n');
fprintf('  Total (sum of medians): %.2f ms\n', total_ms);
fprintf('============================================================\n');
