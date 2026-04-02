% profile_compare_quick.m — Quick benchmark/profile loop for qdldl vs matlab_ldl
%
% Intended for interactive MATLAB use when iterating on the pure-MATLAB
% implementation. By default this script:
%   1. benchmarks qdldl and matlab_ldl on the MPC workload,
%   2. profiles both backends on the same workload,
%   3. keeps total runtime around 30-60 seconds on the current machine.
%
% Usage from the project root:
%   run('test/profile_compare_quick.m')
%

% Configuration (edit directly before running):
bench_reps = 150;
profile_reps_qdldl = 0;
profile_reps_matlab_ldl = 300;
% profile_backends = {'qdldl', 'matlab_ldl'};
profile_backends = {'matlab_ldl'};
scaling_method = {'ruiz', 'equilibrate'};
open_profiler_viewer = usejava('desktop');


[P,q,A,l,u,nx,nu,N,Ad,Bd] = build_mpc_data();

if iscell(scaling_method)
    scaling_methods = scaling_method;
else
    scaling_methods = {scaling_method};
end

fprintf('\n============================================================\n');
fprintf('  OSQP.m Quick Compare — MPC workload\n');
fprintf('============================================================\n');
fprintf('  bench_reps               : %d\n', bench_reps);
fprintf('  profile_reps_qdldl       : %d\n', profile_reps_qdldl);
fprintf('  profile_reps_matlab_ldl  : %d\n', profile_reps_matlab_ldl);
fprintf('  profile_backends         : %s\n', strjoin(profile_backends, ', '));
fprintf('  scaling_method(s)        : %s\n', strjoin(string(scaling_methods), ', '));

quick_compare_by_scaling = struct();
for sm = 1:numel(scaling_methods)
    scaling_method_i = char(string(scaling_methods{sm}));
    fprintf('\n------------------------------------------------------------\n');
    fprintf('Scaling method: %s\n', scaling_method_i);

    bench_qdldl = bench_backend('qdldl', bench_reps, scaling_method_i, P,q,A,l,u,nx,nu,N,Ad,Bd);
    bench_ldl   = bench_backend('matlab_ldl', bench_reps, scaling_method_i, P,q,A,l,u,nx,nu,N,Ad,Bd);

    fprintf('Benchmark summary (median over %d runs)\n', bench_reps);
    fprintf('  %-12s %10.2f ms\n', 'qdldl', bench_qdldl.median_ms);
    fprintf('  %-12s %10.2f ms\n', 'matlab_ldl', bench_ldl.median_ms);
    fprintf('  %-12s %10.2f x\n', 'speedup', bench_qdldl.median_ms / bench_ldl.median_ms);

    profile_results = struct();
    for k = 1:numel(profile_backends)
        backend = profile_backends{k};
        if strcmp(backend, 'qdldl')
            nprof = profile_reps_qdldl;
        elseif strcmp(backend, 'matlab_ldl')
            nprof = profile_reps_matlab_ldl;
        else
            error('Unknown backend: %s', backend);
        end

        fprintf('\nProfiling %s for %d runs...\n', backend, nprof);
        pdata = profile_backend(backend, nprof, scaling_method_i, P,q,A,l,u,nx,nu,N,Ad,Bd);
        profile_results.(backend) = pdata;
        print_top_functions(pdata, backend, 15);

        if open_profiler_viewer
            profview(0, pdata);
            fprintf('  Opened MATLAB profiler viewer for %s (%s).\n', backend, scaling_method_i);
        end
    end

    quick_compare_by_scaling.(matlab.lang.makeValidName(scaling_method_i)).bench_qdldl = bench_qdldl;
    quick_compare_by_scaling.(matlab.lang.makeValidName(scaling_method_i)).bench_matlab_ldl = bench_ldl;
    quick_compare_by_scaling.(matlab.lang.makeValidName(scaling_method_i)).profile_results = profile_results;
end

assignin('base', 'quick_compare_by_scaling', quick_compare_by_scaling);
if isfield(quick_compare_by_scaling, 'ruiz')
    assignin('base', 'quick_compare_bench_qdldl', quick_compare_by_scaling.ruiz.bench_qdldl);
    assignin('base', 'quick_compare_bench_matlab_ldl', quick_compare_by_scaling.ruiz.bench_matlab_ldl);
    assignin('base', 'quick_compare_profile_results', quick_compare_by_scaling.ruiz.profile_results);
end

fprintf('\nSaved variables in base workspace:\n');
fprintf('  quick_compare_by_scaling\n');
fprintf('  quick_compare_bench_qdldl (from ruiz, if present)\n');
fprintf('  quick_compare_bench_matlab_ldl (from ruiz, if present)\n');
fprintf('  quick_compare_profile_results (from ruiz, if present)\n');
fprintf('============================================================\n');

function stats = bench_backend(cfg, nrep, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd)
    t = zeros(nrep, 1);
    run_mpc_rollout(cfg, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd);
    for i = 1:nrep
        tic;
        run_mpc_rollout(cfg, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd);
        t(i) = toc;
    end
    stats.backend = cfg;
    stats.times_s = t;
    stats.median_ms = median(t) * 1000;
    stats.mean_ms = mean(t) * 1000;
end

function pdata = profile_backend(cfg, nrep, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd)
    profile clear;
    profile on -timer performance;
    for i = 1:nrep
        run_mpc_rollout(cfg, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd);
    end
    profile off;
    pdata = profile('info');
end

function print_top_functions(pdata, backend, top_n)
    fnames = {pdata.FunctionTable.FunctionName};
    total_time = [pdata.FunctionTable.TotalTime];
    calls = [pdata.FunctionTable.NumCalls];
    [sorted_time, idx] = sort(total_time, 'descend');

    fprintf('  Top %d functions for %s\n', top_n, backend);
    fprintf('  %-50s  %8s  %8s\n', 'Function', 'Time(s)', 'Calls');
    fprintf('  %s\n', repmat('-', 1, 72));
    for j = 1:min(top_n, numel(idx))
        i = idx(j);
        name = fnames{i};
        if numel(name) > 50
            name = ['...' name(end-46:end)];
        end
        fprintf('  %-50s  %8.4f  %8d\n', name, sorted_time(j), calls(i));
    end
end

function run_mpc_rollout(cfg, scaling_method, P,q,A,l,u,nx,nu,N,Ad,Bd)
    p = OSQP();
    p.setup(P,q,A,l,u, ...
        'warm_start',true, ...
        'verbose',false, ...
        'linear_solver',cfg, ...
        'scaling_method',scaling_method);
    x0 = zeros(nx,1);
    lt = l;
    ut = u;
    for i = 1:15
        r = p.solve();
        ctrl = r.x((N+1)*nx+1:(N+1)*nx+nu);
        x0 = Ad*x0 + Bd*ctrl;
        lt(1:nx) = -x0;
        ut(1:nx) = -x0;
        p.update('l', lt, 'u', ut);
    end
end

function [P,q,A,l,u,nx,nu,N,Ad,Bd] = build_mpc_data()
    Ad = [1 0 0 0 0 0 0.1 0 0 0 0 0; 0 1 0 0 0 0 0 0.1 0 0 0 0;
          0 0 1 0 0 0 0 0 0.1 0 0 0; 0.0488 0 0 1 0 0 0.0016 0 0 0.0992 0 0;
          0 -0.0488 0 0 1 0 0 -0.0016 0 0 0.0992 0; 0 0 0 0 0 1 0 0 0 0 0 0.0992;
          0 0 0 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 0 1 0 0 0 0;
          0 0 0 0 0 0 0 0 1 0 0 0; 0.9734 0 0 0 0 0 0.0488 0 0 0.9846 0 0;
          0 -0.9734 0 0 0 0 0 -0.0488 0 0 0.9846 0; 0 0 0 0 0 0 0 0 0 0 0 0.9846];
    Bd = [0 -0.0726 0 0.0726; -0.0726 0 0.0726 0; -0.0152 0.0152 -0.0152 0.0152;
          0 -0.0006 0 0.0006; 0.0006 0 -0.0006 0; 0.0106 0.0106 0.0106 0.0106;
          0 -1.4512 0 1.4512; -1.4512 0 1.4512 0; -0.3049 0.3049 -0.3049 0.3049;
          0 -0.0236 0 0.0236; 0.0236 0 -0.0236 0; 0.2107 0.2107 0.2107 0.2107];
    nx = 12;
    nu = 4;
    N = 10;
    u0 = 10.5916;
    umin = [9.6;9.6;9.6;9.6] - u0;
    umax = [13;13;13;13] - u0;
    xmin = [-pi/6;-pi/6;-Inf;-Inf;-Inf;-1;-Inf(6,1)];
    xmax = [pi/6;pi/6;Inf;Inf;Inf;Inf;Inf(6,1)];
    Q = diag([0 0 10 10 10 10 0 0 0 5 5 5]);
    R = 0.1 * eye(4);
    xr = [0;0;1;0;0;0;0;0;0;0;0;0];
    P = blkdiag(kron(speye(N), Q), Q, kron(speye(N), R));
    q = [repmat(-Q*xr, N, 1); -Q*xr; zeros(N*nu,1)];
    Ax = kron(speye(N+1), -speye(nx)) + kron(sparse(diag(ones(N,1), -1)), Ad);
    Bu = kron([sparse(1,N); speye(N)], Bd);
    Aeq = [Ax, Bu];
    x0 = zeros(nx,1);
    leq = [-x0; zeros(N*nx,1)];
    ueq = leq;
    Aineq = speye((N+1)*nx + N*nu);
    lineq = [repmat(xmin, N+1, 1); repmat(umin, N, 1)];
    uineq = [repmat(xmax, N+1, 1); repmat(umax, N, 1)];
    A = [Aeq; Aineq];
    l = [leq; lineq];
    u = [ueq; uineq];
end