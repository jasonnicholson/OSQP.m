% profile_qdldl.m — Profile the QDLDL hotspots across a suite of KKT matrices
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'src'), fullfile(repoRoot, 'qdldl', 'src'));

%% ---- Build KKT matrix suite ----
% Gather matrices representative of the examples we benchmark

% tiny 2x2
P1 = sparse([4,1;1,2]); q1=[1;1]; A1=sparse([1,1;1,0;0,1]); l1=[1;0;0]; u1=[1;0.7;0.7];
prob1 = OSQP(); prob1.setup(P1,q1,A1,l1,u1,'verbose',false);

% medium 50-var least-squares
rng(1); m4=30; n4=20;
Ad4=sprandn(m4,n4,0.7); b4=randn(m4,1);
P4=blkdiag(sparse(n4,n4),speye(m4)); q4=zeros(n4+m4,1);
A4=[Ad4,-speye(m4);speye(n4),sparse(n4,m4)];
l4=[b4;zeros(n4,1)]; u4=[b4;ones(n4,1)];
prob4 = OSQP(); prob4.setup(P4,q4,A4,l4,u4,'verbose',false);

% large SVM 1010-var
rng(1); n7=10; m7=1000; N7=ceil(m7/2);
A_upp7=sprandn(N7,n7,0.5); A_low7=sprandn(N7,n7,0.5);
Ad7=[A_upp7/sqrt(n7)+(A_upp7~=0)/n7; A_low7/sqrt(n7)-(A_low7~=0)/n7];
b7=[ones(N7,1);-ones(N7,1)];
P7=blkdiag(speye(n7),sparse(m7,m7)); q7=[zeros(n7,1);ones(m7,1)];
A7=[diag(b7)*Ad7,-speye(m7); sparse(m7,n7),speye(m7)];
l7=[-inf*ones(m7,1);zeros(m7,1)]; u7=[-ones(m7,1);inf*ones(m7,1)];
prob7 = OSQP(); prob7.setup(P7,q7,A7,l7,u7,'verbose',false);

% large Lasso 1020-var
rng(1); n8=10; m8=1000;
Ad8=sprandn(m8,n8,0.5);
x_true8=(randn(n8,1)>0.8).*randn(n8,1)/sqrt(n8);
b8=Ad8*x_true8+0.5*randn(m8,1);
P8=blkdiag(sparse(n8,n8),speye(m8),sparse(n8,n8)); q8=zeros(2*n8+m8,1);
A8=[Ad8,-speye(m8),sparse(m8,n8); speye(n8),sparse(n8,m8),-speye(n8);
    speye(n8),sparse(n8,m8),speye(n8)];
l8=[b8;-inf*ones(n8,1);zeros(n8,1)]; u8=[b8;zeros(n8,1);inf*ones(n8,1)];
prob8 = OSQP(); prob8.setup(P8,q8,A8,l8,u8,'warm_start',true,'verbose',false);

%% ---- Profile: run SVM (most expensive) ----
profile on -timer performance

% Run one SVM solve
prob7.solve();

% Run 5 Lasso solves
for i=1:5
    prob8.update('q',[zeros(n8+m8,1);i*ones(n8,1)]); prob8.solve();
end

profile off

%% ---- Export profile data ----
pdata = profile('info');

% Sort by self time
fnames = {pdata.FunctionTable.FunctionName};
self   = [pdata.FunctionTable.TotalTime];
calls  = [pdata.FunctionTable.NumCalls];
child  = [pdata.FunctionTable.TotalTime] - [pdata.FunctionTable.TotalTime]; % placeholder

% Sort descending
[self_sorted, idx] = sort(self, 'descend');

fprintf('\n====================================================\n');
fprintf('  QDLDL / OSQP Profile — Top 20 Functions\n');
fprintf('====================================================\n');
fprintf('  %-50s  %8s  %8s\n', 'Function', 'Time(s)', 'Calls');
fprintf('  %s\n', repmat('-', 1, 72));

top_n = min(20, numel(idx));
for k = 1:top_n
    i = idx(k);
    fname = fnames{i};
    % shorten long names
    if numel(fname) > 50
        fname = ['...' fname(end-46:end)];
    end
    fprintf('  %-50s  %8.4f  %8d\n', fname, self_sorted(k), calls(i));
end

% Save CSV for external analysis
fid = fopen('/tmp/qdldl_profile.csv', 'w');
fprintf(fid, 'Function,TotalTime_s,NumCalls\n');
for k = 1:numel(idx)
    i = idx(k);
    fprintf(fid, '"%s",%.6f,%d\n', fnames{i}, self_sorted(k), calls(i));
end
fclose(fid);
fprintf('\nProfile saved to /tmp/qdldl_profile.csv\n');
