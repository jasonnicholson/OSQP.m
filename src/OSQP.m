classdef OSQP < handle
  % OSQP  Pure-MATLAB implementation of the OSQP solver.
  %
  %   Solves convex quadratic programs of the form:
  %
  %       minimize    0.5 x' P x + q' x
  %       subject to  l <= A x <= u
  %
  %   This is a pure MATLAB port of the OSQP solver, using QDLDL for the
  %   direct KKT linear system solves.
  %
  % Usage:
  %   solver = OSQP()
  %   solver.setup(P, q, A, l, u, ...)           % configure problem
  %   results = solver.solve()                    % solve
  %   solver.update('q', q_new, ...)             % update vectors/matrices
  %   solver.update_settings('eps_abs', 1e-6, ...) % update settings
  %   solver.warm_start('x', x0, 'y', y0)        % warm start
  %
  % See also qdldl, setupPath

  % Copyright (c) 2026 Jason H. Nicholson
  % Copyright (c) 2017 Bartolomeo Stellato, Baris Stellato, and OSQP contributors
  % SPDX-License-Identifier: Apache-2.0
  % Ported to MATLAB from OSQP.jl (https://github.com/osqp/OSQP.jl)

  properties (Access = private)
    % Problem data (original, unscaled)
    P_triu  % upper triangular part of P (sparse)
    q       % linear cost (n x 1)
    A       % constraint matrix (m x n, sparse)
    l       % lower bounds (m x 1)
    u       % upper bounds (m x 1)
    n       % number of variables
    m       % number of constraints

    % Settings
    settings

    % Scaling data
    scl     % struct with fields: D, E, Dinv, Einv, c, cinv

    % Scaled problem matrices
    Ps      % scaled P_triu (sparse)
    qs      % scaled q
    As      % scaled A (sparse)
    ls      % scaled l
    us      % scaled u

    % ADMM iterates
    x       % primal variable (n x 1)
    z       % slack variable (m x 1)
    y       % dual variable (m x 1)
    x_prev  % x at previous iteration
    z_prev  % z at previous iteration

    % Per-constraint rho vector
    rho_vec      % (m x 1)
    rho_inv_vec  % (m x 1)
    constr_type  % 0=ineq, 1=eq, -1=box (m x 1)

    % Linear system (KKT matrix factorization)
    kkt_factor  % QDLDLFactorization of KKT system

    % Flags
    isSetup = false
    non_convex = false  % true when P is indefinite but P+sigma*I is PD

    % Problem setup time
    setup_time = 0
    solve_time = 0
    update_time = 0
    polish_time = 0
  end

  % =====================================================================
  % Public API
  % =====================================================================
  methods

    function obj = OSQP()
      % OSQP  Constructor. Creates an empty OSQP solver object.
    end

    function setup(obj, P, q, A, l, u, varargin)
      % SETUP  Configure solver with problem data and settings.
      %
      %   solver.setup(P, q, A, l, u)
      %   solver.setup(P, q, A, l, u, 'eps_abs', 1e-6, ...)
      %   solver.setup(P, q, A, l, u, settings_struct)

      t_start = tic;

      % ---- Parse settings ----
      obj.settings = OSQP.default_settings();
      if ~isempty(varargin)
        if isstruct(varargin{1})
          s = varargin{1};
          fnames = fieldnames(s);
          for k = 1:numel(fnames)
            obj.settings.(fnames{k}) = s.(fnames{k});
          end
        else
          if mod(numel(varargin), 2) ~= 0
            error('OSQP:setup', 'Settings must be name/value pairs or a struct.');
          end
          for k = 1:2:numel(varargin)
            obj.settings.(varargin{k}) = varargin{k+1};
          end
        end
      end

      % ---- Determine dimensions ----
      if isempty(P)
        if ~isempty(q)
          obj.n = numel(q);
        elseif ~isempty(A)
          obj.n = size(A, 2);
        else
          error('OSQP:setup', 'Problem has no variables.');
        end
      else
        obj.n = size(P, 1);
      end

      if isempty(A)
        obj.m = 0;
      else
        obj.m = size(A, 1);
      end

      % ---- Default missing data ----
      if isempty(P)
        P = sparse(obj.n, obj.n);
      end
      if isempty(q)
        q = zeros(obj.n, 1);
      end
      if isempty(A)
        A = sparse(0, obj.n);
        l = zeros(0, 1);
        u = zeros(0, 1);
      end
      if isempty(l)
        l = -inf(obj.m, 1);
      end
      if isempty(u)
        u = inf(obj.m, 1);
      end

      % ---- Validate dimensions ----
      q = q(:);
      l = l(:);
      u = u(:);
      if numel(q) ~= obj.n
        error('OSQP:setup', 'q must have length n=%d', obj.n);
      end
      if numel(l) ~= obj.m
        error('OSQP:setup', 'l must have length m=%d', obj.m);
      end
      if numel(u) ~= obj.m
        error('OSQP:setup', 'u must have length m=%d', obj.m);
      end

      % ---- Store upper triangular P ----
      P = sparse(P);
      if ~istriu(P)
        P = triu(P);
      end
      obj.P_triu = P;
      obj.q = double(q);
      obj.A = sparse(A);
      obj.l = max(double(l), -OSQP.OSQP_INFTY);
      obj.u = min(double(u),  OSQP.OSQP_INFTY);

      % ---- Check convexity of P ----
      % P must be PSD; if P has a negative eigenvalue, flag as non-convex.
      % We check P itself (not P+sigma*I) so sigma does not mask indefiniteness.
      % Use Cholesky as a fast PSD proxy: chol(P + eps*I) succeeds iff all
      % eigenvalues of P are > -eps, which is far cheaper than eigs.
      Pfull = obj.P_triu + obj.P_triu' - diag(diag(obj.P_triu));
      if obj.n > 0 && nnz(Pfull) > 0
        thresh = 1e-7;
        try
          chol(Pfull + thresh * speye(obj.n), 'lower');
          obj.non_convex = false;
        catch
          % At least one eigenvalue of P < -1e-7 → non-convex.
          % Check whether P + sigma*I is still PSD (sigma may regularise it).
          Psig = Pfull + obj.settings.sigma * speye(obj.n);
          try
            chol(Psig + thresh * speye(obj.n), 'lower');
            obj.non_convex = true;
          catch
            error('OSQP:NonConvex', ...
              'P is non-convex and sigma is too small: P+sigma*I is not PSD.');
          end
        end
      else
        obj.non_convex = false;
      end

      % ---- Initialize ADMM variables ----
      obj.x = zeros(obj.n, 1);
      obj.z = zeros(obj.m, 1);
      obj.y = zeros(obj.m, 1);

      % ---- Classify constraints ----
      obj.constr_type = OSQP.classify_constraints(obj.l, obj.u);

      % ---- Scale problem ----
      [obj.scl, obj.Ps, obj.qs, obj.As, obj.ls, obj.us] = ...
        OSQP.scale_problem(obj.P_triu, obj.q, obj.A, obj.l, obj.u, obj.settings);

      % ---- Build rho vector ----
      [obj.rho_vec, obj.rho_inv_vec] = OSQP.make_rho_vec( ...
        obj.constr_type, obj.settings.rho, obj.m);

      % ---- Factorize KKT matrix ----
      obj.kkt_factor = OSQP.factorize_kkt( ...
        obj.Ps, obj.As, obj.rho_vec, obj.settings.sigma, obj.n, obj.m, ...
        obj.settings.linear_solver);

      obj.isSetup = true;
      obj.setup_time = toc(t_start);
    end

    function results = solve(obj)
      % SOLVE  Solve the configured QP.
      %
      %   results = solver.solve()
      %
      %   results.x             - primal solution
      %   results.y             - dual solution
      %   results.prim_inf_cert - certificate of primal infeasibility
      %   results.dual_inf_cert - certificate of dual infeasibility
      %   results.info          - solver info struct

      if ~obj.isSetup
        error('OSQP:solve', ...
          'Solver not set up. Call setup() before solve().');
      end

      % Non-convex P detected at setup: immediately return failure.
      if obj.non_convex
        results = OSQP.empty_results(obj.n, obj.m);
        results.info.status_val = OSQP.STATUS_NON_CONVEX;
        results.info.status     = OSQP.status_val_to_str(OSQP.STATUS_NON_CONVEX);
        results.info.obj_val    = nan;
        return;
      end

      t_solve = tic;

      results = OSQP.empty_results(obj.n, obj.m);
      s = obj.settings;

      % ---- Apply scaling to initial iterates ----
      xs = obj.scl.Dinv .* obj.x;
      zs = obj.scl.E    .* obj.z;
      ys = obj.scl.Einv .* obj.y * obj.scl.c;

      % ---- Main ADMM loop ----
      status_val = OSQP.STATUS_UNSOLVED;
      rho_updates = 0;
      adaptive_rho_interval = 0;

      if s.adaptive_rho && s.adaptive_rho_interval > 0
        adaptive_rho_interval = s.adaptive_rho_interval;
      end

      t_iter_start = tic;

      % Cold start if not warm start
      if ~s.warm_start
        xs = zeros(obj.n, 1);
        zs = zeros(obj.m, 1);
        ys = zeros(obj.m, 1);
      end

      for iter = 1:s.max_iter
        % ---- Store previous iterates ----
        xs_prev = xs;
        zs_prev = zs;
        ys_prev = ys;

        % ---- Step 1: Solve KKT system for xz_tilde ----
        rhs_x = s.sigma * xs_prev - obj.qs;
        if obj.m > 0
          rhs_z = zs_prev - obj.rho_inv_vec .* ys;
          rhs = [rhs_x; rhs_z];
        else
          rhs = rhs_x;
        end

        sol = obj.kkt_factor \ rhs;
        xtilde = sol(1:obj.n);

        if obj.m > 0
          ztilde = zs_prev + obj.rho_inv_vec .* (sol(obj.n+1:end) - ys);
        else
          ztilde = zeros(0, 1);
        end

        % ---- Step 2: Update x and z with relaxation ----
        xs = s.alpha * xtilde + (1 - s.alpha) * xs_prev;

        if obj.m > 0
          zs_relaxed = s.alpha * ztilde + (1 - s.alpha) * zs_prev;
          zs = OSQP.project_box( ...
            zs_relaxed + obj.rho_inv_vec .* ys, obj.ls, obj.us);
        end

        % ---- Step 3: Update dual variable ----
        if obj.m > 0
          ys = ys + obj.rho_vec .* ( ...
            s.alpha * ztilde + (1 - s.alpha) * zs_prev - zs);
        end

        % ---- Check termination ----
        if s.check_termination > 0 && mod(iter, s.check_termination) == 0
          [prim_res, dual_res] = OSQP.compute_residuals( ...
            xs, zs, ys, obj.Ps, obj.qs, obj.As, ...
            obj.n, obj.m, s.scaled_termination, obj.scl);

          [converged, status_val] = OSQP.check_convergence( ...
            prim_res, dual_res, xs, zs, ys, xs_prev, ys_prev, ...
            obj.Ps, obj.qs, obj.As, obj.ls, obj.us, ...
            obj.n, obj.m, s);

          if converged
            break;
          end
        end

        % ---- Check time limit ----
        if s.time_limit > 0 && toc(t_solve) >= s.time_limit
          status_val = OSQP.STATUS_TIME_LIMIT_REACHED;
          break;
        end

        % ---- Adaptive rho ----
        if s.adaptive_rho
          if adaptive_rho_interval == 0
            if iter == 1
              t_iter_start = tic;
            elseif iter == 2
              t_one_iter = toc(t_iter_start);
              t_setup = max(obj.setup_time, 1e-10);
              if t_one_iter > 0
                adaptive_rho_interval = max(1, ...
                  round(s.adaptive_rho_fraction * t_setup / t_one_iter));
              else
                adaptive_rho_interval = 25;
              end
            end
          end
          if adaptive_rho_interval > 0 && mod(iter, adaptive_rho_interval) == 0
            [prim_res2, dual_res2] = OSQP.compute_residuals_scaled( ...
              xs, zs, ys, obj.Ps, obj.qs, obj.As, obj.n, obj.m);
            new_rho = OSQP.compute_new_rho( ...
              prim_res2, dual_res2, obj.settings.rho, ...
              s.adaptive_rho_tolerance);
            if new_rho ~= obj.settings.rho
              obj.settings.rho = new_rho;
              [obj.rho_vec, obj.rho_inv_vec] = OSQP.make_rho_vec( ...
                obj.constr_type, new_rho, obj.m);
              obj.kkt_factor = OSQP.factorize_kkt( ...
                obj.Ps, obj.As, obj.rho_vec, s.sigma, obj.n, obj.m, ...
                obj.settings.linear_solver);
              rho_updates = rho_updates + 1;
            end
          end
        end
      end % for iter

      obj.solve_time = toc(t_solve);

      % If loop ended without check, do final convergence test
      if status_val == OSQP.STATUS_UNSOLVED
        if ~exist('xs_prev', 'var')
          xs_prev = xs; ys_prev = ys;
        end
        [prim_res, dual_res] = OSQP.compute_residuals( ...
          xs, zs, ys, obj.Ps, obj.qs, obj.As, ...
          obj.n, obj.m, s.scaled_termination, obj.scl);
        [~, status_val] = OSQP.check_convergence( ...
          prim_res, dual_res, xs, zs, ys, xs_prev, ys_prev, ...
          obj.Ps, obj.qs, obj.As, obj.ls, obj.us, ...
          obj.n, obj.m, s);
        if status_val == OSQP.STATUS_UNSOLVED
          status_val = OSQP.STATUS_MAX_ITER_REACHED;
        end
      end

      % ---- Unscale solution ----
      x_unscaled = obj.scl.D    .* xs;
      z_unscaled = obj.scl.Einv .* zs;
      y_unscaled = obj.scl.E    .* ys * obj.scl.cinv;

      % Store warm-start iterates (unscaled)
      obj.x = x_unscaled;
      obj.z = z_unscaled;
      obj.y = y_unscaled;

      % ---- Fill results ----
      results.info.iter        = iter;
      results.info.status_val  = status_val;
      results.info.status      = OSQP.status_val_to_str(status_val);
      results.info.rho_updates = rho_updates;
      results.info.rho_estimate = obj.settings.rho;
      results.info.setup_time  = obj.setup_time;
      results.info.solve_time  = obj.solve_time;
      results.info.update_time = obj.update_time;

      solution_present = ismember(status_val, [ ...
        OSQP.STATUS_SOLVED, ...
        OSQP.STATUS_SOLVED_INACCURATE, ...
        OSQP.STATUS_MAX_ITER_REACHED]);

      if solution_present
        results.x = x_unscaled;
        results.y = y_unscaled;
        results.prim_inf_cert = nan(obj.m, 1);
        results.dual_inf_cert = nan(obj.n, 1);
        Pfull = obj.P_triu + obj.P_triu' - diag(diag(obj.P_triu));
        results.info.obj_val = 0.5 * (x_unscaled' * (Pfull * x_unscaled)) + ...
          obj.q' * x_unscaled;
        results.info.pri_res = norm(obj.A * x_unscaled - z_unscaled, inf);
        results.info.dua_res = norm(Pfull * x_unscaled + obj.q + obj.A' * y_unscaled, inf);
      elseif status_val == OSQP.STATUS_PRIMAL_INFEASIBLE || ...
          status_val == OSQP.STATUS_PRIMAL_INFEASIBLE_INACCURATE
        results.x = nan(obj.n, 1);
        results.y = nan(obj.m, 1);
        % Certificate: delta_y = unscaled y difference (normalized)
        delta_y = obj.scl.E .* (ys - ys_prev);
        if norm(delta_y, inf) > 0
          delta_y = delta_y / norm(delta_y, inf);
        end
        results.prim_inf_cert = delta_y;
        results.dual_inf_cert = nan(obj.n, 1);
        results.info.obj_val = inf;
        results.info.pri_res = inf;
        results.info.dua_res = inf;
      elseif status_val == OSQP.STATUS_DUAL_INFEASIBLE || ...
          status_val == OSQP.STATUS_DUAL_INFEASIBLE_INACCURATE
        results.x = nan(obj.n, 1);
        results.y = nan(obj.m, 1);
        results.prim_inf_cert = nan(obj.m, 1);
        % Certificate: delta_x = unscaled x difference (normalized)
        delta_x = obj.scl.D .* (xs - xs_prev);
        if norm(delta_x, inf) > 0
          delta_x = delta_x / norm(delta_x, inf);
        end
        results.dual_inf_cert = delta_x;
        results.info.obj_val = -inf;
        results.info.pri_res = inf;
        results.info.dua_res = inf;
      elseif status_val == OSQP.STATUS_NON_CONVEX
        results.x = nan(obj.n, 1);
        results.y = nan(obj.m, 1);
        results.prim_inf_cert = nan(obj.m, 1);
        results.dual_inf_cert = nan(obj.n, 1);
        results.info.obj_val = nan;
        results.info.pri_res = inf;
        results.info.dua_res = inf;
      else
        results.x = nan(obj.n, 1);
        results.y = nan(obj.m, 1);
        results.prim_inf_cert = nan(obj.m, 1);
        results.dual_inf_cert = nan(obj.n, 1);
        results.info.obj_val = nan;
        results.info.pri_res = inf;
        results.info.dua_res = inf;
      end

      % ---- Polishing ----
      results.info.status_polish = 0;
      if s.polish && solution_present
        t_polish = tic;
        results = OSQP.polish_solution(results, ...
          obj.P_triu, obj.q, obj.A, obj.l, obj.u, s);
        obj.polish_time = toc(t_polish);
      end
      results.info.polish_time = obj.polish_time;
      results.info.run_time = obj.setup_time + obj.update_time + ...
        obj.solve_time + obj.polish_time;

      if s.verbose
        fprintf('OSQP: %s | iter=%d | obj=%.4e | pri_res=%.2e | dua_res=%.2e\n', ...
          results.info.status, results.info.iter, ...
          results.info.obj_val, results.info.pri_res, results.info.dua_res);
      end
    end

    function update(obj, varargin)
      % UPDATE  Update problem vectors and/or matrices.
      %
      %   solver.update('q', q_new)
      %   solver.update('l', l_new, 'u', u_new)
      %   solver.update('Px', Px_new)
      %   solver.update('Px', Px_new, 'Px_idx', idx)
      %   solver.update('Ax', Ax_new)

      if ~obj.isSetup
        error('OSQP:update', 'Call setup() before update().');
      end
      t_update = tic;

      if mod(numel(varargin), 2) ~= 0
        error('OSQP:update', 'Arguments must be name/value pairs.');
      end

      args = struct();
      for k = 1:2:numel(varargin)
        args.(varargin{k}) = varargin{k+1};
      end

      refactor = false;

      % q
      if isfield(args, 'q')
        q_new = double(args.q(:));
        if numel(q_new) ~= obj.n
          error('OSQP:update', 'q must have length n=%d', obj.n);
        end
        obj.q = q_new;
        refactor = false; % q doesn't need refactorize
      end

      % l and u
      if isfield(args, 'l') || isfield(args, 'u')
        if isfield(args, 'l')
          l_new = double(args.l(:));
          if numel(l_new) ~= obj.m
            error('OSQP:update', 'l must have length m=%d', obj.m);
          end
          obj.l = max(l_new, -OSQP.OSQP_INFTY);
        end
        if isfield(args, 'u')
          u_new = double(args.u(:));
          if numel(u_new) ~= obj.m
            error('OSQP:update', 'u must have length m=%d', obj.m);
          end
          obj.u = min(u_new, OSQP.OSQP_INFTY);
        end
        % Reclassify constraints and rebuild rho vec
        obj.constr_type = OSQP.classify_constraints(obj.l, obj.u);
        [obj.rho_vec, obj.rho_inv_vec] = OSQP.make_rho_vec( ...
          obj.constr_type, obj.settings.rho, obj.m);
        refactor = true;
      end

      % Px (update nonzero values of P)
      if isfield(args, 'Px')
        Px_new = double(args.Px(:));
        [ri, ci, ~] = find(obj.P_triu);
        nz_orig = nnz(obj.P_triu);
        if isfield(args, 'Px_idx')
          idx = args.Px_idx(:);
          if numel(Px_new) ~= numel(idx)
            error('OSQP:update', 'Px and Px_idx must have same length');
          end
          old_vals = nonzeros(obj.P_triu);
          old_vals(idx) = Px_new;
          obj.P_triu = sparse(ri, ci, old_vals, obj.n, obj.n);
        else
          if numel(Px_new) ~= nz_orig
            error('OSQP:update', 'Px must have %d elements', nz_orig);
          end
          obj.P_triu = sparse(ri, ci, Px_new, obj.n, obj.n);
        end
        refactor = true;
      end

      % Ax (update nonzero values of A)
      if isfield(args, 'Ax')
        Ax_new = double(args.Ax(:));
        if isfield(args, 'Ax_idx')
          idx = args.Ax_idx;
          if numel(Ax_new) ~= numel(idx)
            error('OSQP:update', 'Ax and Ax_idx must have same length');
          end
          [ri, ci, vals] = find(obj.A);
          vals(idx) = Ax_new;
          obj.A = sparse(ri, ci, vals, obj.m, obj.n);
        else
          nz_orig = nnz(obj.A);
          if numel(Ax_new) ~= nz_orig
            error('OSQP:update', 'Ax must have %d elements', nz_orig);
          end
          [ri, ci] = find(obj.A);
          obj.A = sparse(ri, ci, Ax_new, obj.m, obj.n);
        end
        refactor = true;
      end

      % Rescale and refactorize if needed
      if refactor
        [obj.scl, obj.Ps, obj.qs, obj.As, obj.ls, obj.us] = ...
          OSQP.scale_problem(obj.P_triu, obj.q, obj.A, obj.l, obj.u, obj.settings);
        obj.kkt_factor = OSQP.factorize_kkt( ...
          obj.Ps, obj.As, obj.rho_vec, obj.settings.sigma, obj.n, obj.m, ...
          obj.settings.linear_solver);
      else
        % Only q changed; just recompute scaled q: qs = c * D * q
        obj.qs = obj.scl.c * (obj.scl.D .* obj.q);
      end

      obj.update_time = obj.update_time + toc(t_update);
    end

    function update_settings(obj, varargin)
      % UPDATE_SETTINGS  Update solver settings.
      %
      %   solver.update_settings('max_iter', 1000, ...)
      %   solver.update_settings(settings_struct)

      if ~obj.isSetup
        error('OSQP:update_settings', 'Call setup() before update_settings().');
      end

      updatable = {'max_iter','eps_abs','eps_rel','eps_prim_inf','eps_dual_inf', ...
        'time_limit','rho','alpha','delta','polish','polish_refine_iter', ...
        'verbose','check_termination','warm_start','scaled_termination'};

      rho_updated = false;
      if isscalar(varargin) && isstruct(varargin{1})
        s = varargin{1};
        fnames = fieldnames(s);
        for k = 1:numel(fnames)
          if ~ismember(fnames{k}, updatable)
            error('OSQP:update_settings', ...
              'Setting ''%s'' cannot be updated.', fnames{k});
          end
          obj.settings.(fnames{k}) = s.(fnames{k});
        end
        rho_updated = isfield(s, 'rho');
      else
        if mod(numel(varargin), 2) ~= 0
          error('OSQP:update_settings', 'Arguments must be name/value pairs.');
        end
        for k = 1:2:numel(varargin)
          name = varargin{k};
          if ~ismember(name, updatable)
            error('OSQP:update_settings', ...
              'Setting ''%s'' cannot be updated.', name);
          end
          obj.settings.(name) = varargin{k+1};
          if strcmp(name, 'rho'); rho_updated = true; end
        end
      end

      % If rho was updated, rebuild rho vectors and refactorize
      if rho_updated
        [obj.rho_vec, obj.rho_inv_vec] = OSQP.make_rho_vec( ...
          obj.constr_type, obj.settings.rho, obj.m);
        obj.kkt_factor = OSQP.factorize_kkt( ...
          obj.Ps, obj.As, obj.rho_vec, obj.settings.sigma, obj.n, obj.m, ...
          obj.settings.linear_solver);
      end
    end

    function warm_start(obj, varargin)
      % WARM_START  Set warm-start values for primal/dual variables.
      %
      %   solver.warm_start('x', x0)
      %   solver.warm_start('y', y0)
      %   solver.warm_start('x', x0, 'y', y0)

      if ~obj.isSetup
        error('OSQP:warm_start', 'Call setup() before warm_start().');
      end
      if mod(numel(varargin), 2) ~= 0
        error('OSQP:warm_start', 'Arguments must be name/value pairs.');
      end
      x_updated = false;
      y_updated = false;
      for k = 1:2:numel(varargin)
        switch varargin{k}
          case 'x'
            x0 = double(varargin{k+1}(:));
            if numel(x0) ~= obj.n
              error('OSQP:warm_start', 'x0 must have length n=%d', obj.n);
            end
            obj.x = x0;
            x_updated = true;
          case 'y'
            y0 = double(varargin{k+1}(:));
            if numel(y0) ~= obj.m
              error('OSQP:warm_start', 'y0 must have length m=%d', obj.m);
            end
            obj.y = y0;
            y_updated = true;
          otherwise
            error('OSQP:warm_start', 'Unknown argument ''%s''.', varargin{k});
        end
      end
      if x_updated
        % Update z = A*x (unscaled); cold-start y if not provided
        obj.z = obj.A * obj.x;
        if ~y_updated
          obj.y = zeros(obj.m, 1);
        end
      elseif y_updated
        % Only y provided: cold-start x and z
        obj.x = zeros(obj.n, 1);
        obj.z = zeros(obj.m, 1);
      end
    end

    function [n, m] = get_dimensions(obj)
      % GET_DIMENSIONS  Return number of variables and constraints.
      n = obj.n;
      m = obj.m;
    end

  end % methods (public)

  % =====================================================================
  % Static API
  % =====================================================================
  methods (Static)

    function s = default_settings()
      % DEFAULT_SETTINGS  Return a struct of default solver settings.
      s.rho                   = 0.1;
      s.sigma                 = 1e-6;
      s.scaling               = 10;
      s.adaptive_rho          = true;
      s.adaptive_rho_interval = 0;
      s.adaptive_rho_tolerance = 5;
      s.adaptive_rho_fraction = 0.4;
      s.max_iter              = 4000;
      s.eps_abs               = 1e-3;
      s.eps_rel               = 1e-3;
      s.eps_prim_inf          = 1e-4;
      s.eps_dual_inf          = 1e-4;
      s.alpha                 = 1.6;
      s.delta                 = 1e-6;
      s.polish                = false;
      s.polish_refine_iter    = 3;
      s.verbose               = true;
      s.scaled_termination    = false;
      s.check_termination     = 25;
      s.warm_start            = true;
      s.time_limit            = 1e10;
      s.linear_solver         = 'matlab_ldl'; % 'qdldl' | 'qdldl_c' | 'pardiso_mkl' | 'matlab_ldl'
    end

    function v = version()
      % VERSION  Return solver version string.
      v = '0.1.0-matlab';
    end

    function c = constant(name)
      % CONSTANT  Return solver status constant by name.
      switch upper(name)
        case 'OSQP_SOLVED'
          c = OSQP.STATUS_SOLVED;
        case 'OSQP_SOLVED_INACCURATE'
          c = OSQP.STATUS_SOLVED_INACCURATE;
        case 'OSQP_MAX_ITER_REACHED'
          c = OSQP.STATUS_MAX_ITER_REACHED;
        case 'OSQP_PRIMAL_INFEASIBLE'
          c = OSQP.STATUS_PRIMAL_INFEASIBLE;
        case 'OSQP_DUAL_INFEASIBLE'
          c = OSQP.STATUS_DUAL_INFEASIBLE;
        case 'OSQP_TIME_LIMIT_REACHED'
          c = OSQP.STATUS_TIME_LIMIT_REACHED;
        case 'OSQP_NON_CONVEX'
          c = OSQP.STATUS_NON_CONVEX;
        case 'OSQP_UNSOLVED'
          c = OSQP.STATUS_UNSOLVED;
        otherwise
          error('OSQP:constant', 'Unknown constant: %s', name);
      end
    end

  end % methods (static public)

  % =====================================================================
  % Status constants
  % =====================================================================
  properties (Constant)
    OSQP_INFTY                         = 1e30
    STATUS_DUAL_INFEASIBLE_INACCURATE  = 4
    STATUS_PRIMAL_INFEASIBLE_INACCURATE = 3
    STATUS_SOLVED_INACCURATE           = 2
    STATUS_SOLVED                      = 1
    STATUS_MAX_ITER_REACHED            = -2
    STATUS_PRIMAL_INFEASIBLE           = -3
    STATUS_DUAL_INFEASIBLE             = -4
    STATUS_INTERRUPTED                 = -5
    STATUS_TIME_LIMIT_REACHED          = -6
    STATUS_NON_CONVEX                  = -7
    STATUS_UNSOLVED                    = -10
  end

  % =====================================================================
  % Internal static helpers
  % =====================================================================
  methods (Static, Access = private)

    function str = status_val_to_str(val)
      switch val
        case  4,  str = 'dual_infeasible_inaccurate';
        case  3,  str = 'primal_infeasible_inaccurate';
        case  2,  str = 'solved_inaccurate';
        case  1,  str = 'solved';
        case -2,  str = 'maximum_iterations_reached';
        case -3,  str = 'primal_infeasible';
        case -4,  str = 'dual_infeasible';
        case -5,  str = 'interrupted';
        case -6,  str = 'time_limit_reached';
        case -7,  str = 'non_convex';
        otherwise, str = 'unsolved';
      end
    end

    function r = empty_results(n, m)
      r.x = nan(n, 1);
      r.y = nan(m, 1);
      r.prim_inf_cert = nan(m, 1);
      r.dual_inf_cert = nan(n, 1);
      r.info.iter          = 0;
      r.info.status        = 'unsolved';
      r.info.status_val    = OSQP.STATUS_UNSOLVED;
      r.info.status_polish = 0;
      r.info.obj_val       = nan;
      r.info.pri_res       = nan;
      r.info.dua_res       = nan;
      r.info.setup_time    = 0;
      r.info.solve_time    = 0;
      r.info.update_time   = 0;
      r.info.polish_time   = 0;
      r.info.run_time      = 0;
      r.info.rho_updates   = 0;
      r.info.rho_estimate  = 0;
    end

    function ctype = classify_constraints(l, u)
      % 1 = equality, 0 = inequality (box)
      m = numel(l);
      ctype = zeros(m, 1);
      for i = 1:m
        if l(i) == u(i)
          ctype(i) = 1;  % equality
        end
      end
    end

    function [rho_vec, rho_inv_vec] = make_rho_vec(ctype, rho, m)
      if m == 0
        rho_vec     = zeros(0, 1);
        rho_inv_vec = zeros(0, 1);
        return;
      end
      rho_vec     = rho * ones(m, 1);
      % Equality constraints: multiply rho by 1000
      rho_vec(ctype == 1) = rho * 1e3;
      rho_inv_vec = 1 ./ rho_vec;
    end

    function [scl, Ps, qs, As, ls, us] = scale_problem(P_triu, q, A, l, u, settings)
      % SCALE_PROBLEM  Apply Ruiz equilibration scaling.
      n = size(P_triu, 1);
      m = size(A, 1);

      num_iter = settings.scaling;

      D    = ones(n, 1);
      E    = ones(m, 1);
      c    = 1.0;

      if num_iter == 0
        % No scaling
        Ps = P_triu;
        qs = q;
        As = A;
        ls = l;
        us = u;
        scl.D    = D;
        scl.E    = E;
        scl.Dinv = 1 ./ D;
        scl.Einv = 1 ./ E;
        scl.c    = c;
        scl.cinv = 1 / c;
        return;
      end

      % Symmetrize P for norm computations
      Pfull = P_triu + P_triu' - diag(diag(P_triu));

      for iter_s = 1:num_iter
        % Compute column norms of [P; A] and row norms of [A]
        % For symmetric P we use the max of row/col norms
        if n > 0
          % Vectorised inf-norms of each column of D*P*D and A*D
          Dsp = spdiags(D, 0, n, n);
          PDE = Dsp * Pfull * Dsp;              % symmetric D*P*D scaling
          AD  = A * Dsp;                         % A*D column scaling
          combined = max(full(max(abs(PDE)))', full(max(abs(AD)))');   % n×1
          pos = combined > 0;
          D(pos) = D(pos) ./ sqrt(combined(pos));
        end
        if m > 0
          % Vectorised inf-norms of each row of E*A*D
          Dsp = spdiags(D, 0, n, n);
          Esp = spdiags(E, 0, m, m);
          EAD = Esp * A * Dsp;
          EAD_norms = full(max(abs(EAD), [], 2));   % m×1
          pos = EAD_norms > 0;
          E(pos) = E(pos) ./ sqrt(EAD_norms(pos));
        end
        % Cost scaling: scale by mean of D'*P_diag*D norms
        DPD = D' .* diag(sparse(Pfull))' .* D';
        mean_norm = mean(abs(DPD));
        q_norms   = abs(q .* D);
        cost_scale = max(mean_norm, mean(q_norms));
        if cost_scale > 0
          c = c / sqrt(cost_scale);
        end
      end

      % Clip scaling vectors
      D = max(D, 1e-6);
      E = max(E, 1e-6);
      c = max(c, 1e-6);

      Dinv = 1 ./ D;
      Einv = 1 ./ E;
      cinv = 1 / c;

      % Apply scaling: hat(P) = c*D*P*D, hat(q) = c*D*q
      %                hat(A) = E*A*D, hat(l) = E*l, hat(u) = E*u
      Dsp = diag(sparse(D));
      Esp = diag(sparse(E));
      Pfull_scaled = c * (Dsp * Pfull * Dsp);
      Ps = triu(Pfull_scaled);
      qs = c * (D .* q);
      As = Esp * A * Dsp;
      % Clip scaled bounds
      ls_raw = E .* l;
      us_raw = E .* u;
      ls = max(ls_raw, -OSQP.OSQP_INFTY);
      us = min(us_raw,  OSQP.OSQP_INFTY);

      scl.D    = D;
      scl.E    = E;
      scl.Dinv = Dinv;
      scl.Einv = Einv;
      scl.c    = c;
      scl.cinv = cinv;
    end

    function F = factorize_kkt(P_triu, A, rho_vec, sigma, n, m, linear_solver)
      % FACTORIZE_KKT  Factor the (n+m) x (n+m) KKT matrix:
      %   K = [P + sigma*I,  A';  A, -diag(1./rho_vec)]
      % This is quasi-definite.
      %
      %   linear_solver: 'qdldl' (pure-MATLAB QDLDL),
      %                  'qdldl_c' (upstream C QDLDL via MEX),
      %                  'pardiso_mkl' (Intel oneMKL Pardiso via MEX),
      %                  'matlab_ldl' (MATLAB built-in ldl, fastest pure MATLAB), or
      %                  'matlab_linsolve' (ldl + linsolve opts — for benchmarking)

      if nargin < 7 || isempty(linear_solver)
        linear_solver = 'matlab_ldl';
      end

      Pfull = P_triu + P_triu' - diag(diag(P_triu));
      Ktl   = Pfull + sigma * speye(n);   % top-left block

      if m > 0
        Kbr = -diag(sparse(1 ./ rho_vec));  % bottom-right block
        K   = [Ktl, A'; A, Kbr];
      else
        K = Ktl;
      end

      K = (K + K') / 2;  % ensure exact symmetry

      if strcmp(linear_solver, 'matlab_ldl')
        F = MATLABLDLFactorization(K);
      elseif strcmp(linear_solver, 'matlab_linsolve')
        F = LinsolveFactorization(K);
      elseif strcmp(linear_solver, 'pardiso_mkl')
        mexName = ['pardiso_mkl_mex.' mexext];
        if exist(mexName, 'file') ~= 3
          build_pardiso_mkl_mex();
        end
        F = PARDISOMKLFactorization(K);
      elseif strcmp(linear_solver, 'qdldl_c')
        mexName = ['qdldl_c_factor_mex.' mexext];
        if exist(mexName, 'file') ~= 3
          build_qdldl_c_mex();
        end
        F = QDLDLCFactorization(triu(K), 'auto');
      else
        F = qdldl(triu(K), 'perm', 'auto');
      end
    end

    function z_proj = project_box(z, l, u)
      z_proj = min(max(z, l), u);
    end

    function [prim_res, dual_res] = compute_residuals(xs, zs, ys, Ps, qs, As, ~, m, ~, ~)
      % Compute primal and dual residuals in original scale
      if m > 0
        Axs = As * xs;
        prim_res = norm(Axs - zs, inf);
      else
        prim_res = 0;
      end
      Pfull = Ps + Ps' - diag(diag(Ps));
      if m > 0
        dual_res = norm(Pfull * xs + qs + As' * ys, inf);
      else
        dual_res = norm(Pfull * xs + qs, inf);
      end
    end

    function [prim_res, dual_res] = compute_residuals_scaled(xs, zs, ys, Ps, qs, As, ~, m)
      if m > 0
        prim_res = norm(As * xs - zs, inf);
      else
        prim_res = 0;
      end
      Pfull = Ps + Ps' - diag(diag(Ps));
      if m > 0
        dual_res = norm(Pfull * xs + qs + As' * ys, inf);
      else
        dual_res = norm(Pfull * xs + qs, inf);
      end
    end

    function [converged, status_val] = check_convergence( ...
        prim_res, dual_res, xs, zs, ys, xs_prev, ys_prev, ...
        Ps, qs, As, ls, us, ~, m, s)
      % CHECK_CONVERGENCE  Test optimality, primal/dual infeasibility.
      %
      % Certificates follow OSQP paper (Stellato et al. 2020):
      %   delta_x = xs - xs_prev  →  dual infeasibility direction
      %   delta_y = ys - ys_prev  →  primal infeasibility direction

      status_val = OSQP.STATUS_UNSOLVED;
      converged  = false;

      Pfull = Ps + Ps' - diag(diag(Ps));

      % ---- Adaptive tolerances ----
      if m > 0
        Axs_norm = norm(As * xs, inf);
        zs_norm  = norm(zs, inf);
        eps_prim = s.eps_abs + s.eps_rel * max(Axs_norm, zs_norm);
      else
        eps_prim = s.eps_abs;
      end
      Pxs_norm  = norm(Pfull * xs, inf);
      Atys_norm = 0;
      if m > 0
        Atys_norm = norm(As' * ys, inf);
      end
      qs_norm  = norm(qs, inf);
      eps_dual = s.eps_abs + s.eps_rel * max([Pxs_norm; Atys_norm; qs_norm]);

      % ---- Optimality ----
      if prim_res <= eps_prim && dual_res <= eps_dual
        status_val = OSQP.STATUS_SOLVED;
        converged  = true;
        return;
      end

      % ---- Primal infeasibility (certificate: delta_y) ----
      % The dual iterates diverge; project direction onto feasible set.
      if m > 0
        dy       = ys - ys_prev;
        norm_dy  = norm(dy, inf);
        if norm_dy > s.eps_prim_inf
          dy_n    = dy / norm_dy;
          ATdy    = norm(As' * dy_n, inf);
          lu_ok   = OSQP.primal_inf_check(dy_n, ls, us, s.eps_prim_inf);
          if ATdy <= s.eps_prim_inf && lu_ok
            status_val = OSQP.STATUS_PRIMAL_INFEASIBLE;
            converged  = true;
            return;
          end
        end
      end

      % ---- Dual infeasibility (certificate: delta_x) ----
      % The primal iterates diverge in an unbounded direction.
      dx      = xs - xs_prev;
      norm_dx = norm(dx, inf);
      if norm_dx > s.eps_dual_inf
        dx_n    = dx / norm_dx;
        Pdx     = norm(Pfull * dx_n, inf);
        qdx     = qs' * dx_n;
        if Pdx <= s.eps_dual_inf && qdx < -s.eps_dual_inf
          if m > 0
            Adx = As * dx_n;
            ok  = OSQP.dual_inf_check(Adx, ls, us, s.eps_dual_inf);
          else
            ok = true;
          end
          if ok
            status_val = OSQP.STATUS_DUAL_INFEASIBLE;
            converged  = true;
            return;
          end
        end
      end
    end

    function ok = primal_inf_check(v, l, u, eps)
      % Check u'*v+ + l'*v- < 0 where v+ = max(v,0), v- = min(v,0)
      % Treat bounds with |b| >= OSQP_INFTY as infinite.
      INF = OSQP.OSQP_INFTY;
      vpos = max(v, 0);
      vneg = min(v, 0);
      val  = 0;
      for i = 1:numel(l)
        if abs(u(i)) < INF, val = val + u(i) * vpos(i); end
        if abs(l(i)) < INF, val = val + l(i) * vneg(i); end
      end
      ok = val < -eps;
    end

    function ok = dual_inf_check(Adx, l, u, eps)
      % Check A*delta_x satisfies signs required by dual infeasibility.
      % Treat bounds with |b| >= OSQP_INFTY as infinite.
      INF = OSQP.OSQP_INFTY;
      ok = true;
      for i = 1:numel(l)
        li = l(i); ui = u(i);
        li_fin = abs(li) < INF;
        ui_fin = abs(ui) < INF;
        ai = Adx(i);
        if li_fin && ui_fin
          % Both finite: must be near zero
          if abs(ai) > eps, ok = false; return; end
        elseif li_fin && ~ui_fin
          % l finite, u = +inf: must be >= -eps
          if ai < -eps, ok = false; return; end
        elseif ~li_fin && ui_fin
          % l = -inf, u finite: must be <= eps
          if ai > eps, ok = false; return; end
        end
        % both infinite: no constraint on Adx(i)
      end
    end

    function new_rho = compute_new_rho(prim_res, dual_res, rho, tol)
      if prim_res == 0 || dual_res == 0
        new_rho = rho;
        return;
      end
      ratio = sqrt(prim_res / dual_res);
      new_rho_candidate = rho * ratio;
      if new_rho_candidate > tol * rho || new_rho_candidate < rho / tol
        new_rho = max(min(new_rho_candidate, 1e6), 1e-6);
      else
        new_rho = rho;
      end
    end

    function results = polish_solution(results, P_triu, q, A, l, u, s)
      % POLISH_SOLUTION  Refine solution via KKT active-set solve.
      %
      % Uses dual variable signs to identify active constraints, which is
      % more robust than primal residual thresholds when the ADMM solution
      % is only approximately feasible.

      n = numel(q);
      m = size(A, 1);
      x  = results.x;
      y  = results.y;
      Pfull = P_triu + P_triu' - diag(diag(P_triu));

      INF = OSQP.OSQP_INFTY;

      % Identify active constraints via dual variable signs.
      % y > 0 ↔ active at upper bound; y < 0 ↔ active at lower bound.
      % Fallback: also mark active if Ax is within eps_abs of a bound.
      Ax = A * x;
      tol_act  = max(s.delta, s.eps_abs);
      act_l = (y < -s.delta & abs(l) < INF) | ...
        (abs(l) < INF & abs(Ax - l) <= tol_act);
      act_u = (y >  s.delta & abs(u) < INF) | ...
        (abs(u) < INF & abs(Ax - u) <= tol_act);
      active_idx = find(act_l | act_u);
      n_act = numel(active_idx);

      if n_act == 0
        % No active constraints: the ADMM solution is already good.
        % Only attempt unconstrained refinement if P is positive definite.
        try
          % Check if P + delta*I is well-conditioned enough
          Preg = Pfull + s.delta * speye(n);
          x_pol = Preg \ (-q);
          Ax_pol = A * x_pol;
          feas_l = all(Ax_pol >= l - tol_act | abs(l) >= INF);
          feas_u = all(Ax_pol <= u + tol_act | abs(u) >= INF);
          if feas_l && feas_u
            obj_pol = 0.5 * x_pol' * Pfull * x_pol + q' * x_pol;
            obj_cur = 0.5 * x' * Pfull * x + q' * x;
            if obj_pol < obj_cur + abs(obj_cur) * 1e-6
              results.x = x_pol;
              results.y = zeros(m, 1);
              results.info.obj_val = obj_pol;
              results.info.status_polish = 1;
            else
              results.info.status_polish = -1;
            end
          else
            results.info.status_polish = -1;
          end
        catch
          results.info.status_polish = -1;
        end
        return;
      end

      % Build active-set KKT system
      A_act = A(active_idx, :);
      b_act = zeros(n_act, 1);
      for k = 1:n_act
        i = active_idx(k);
        if act_l(i) && act_u(i)
          b_act(k) = (l(i) + u(i)) / 2;
        elseif act_l(i)
          b_act(k) = l(i);
        else
          b_act(k) = u(i);
        end
      end

      try
        % KKT: [P + delta*I, A_act'; A_act, -delta*I] [x; nu] = [-q; b_act]
        % Solve directly from scratch for best accuracy.
        Kreg = [Pfull + s.delta*speye(n), A_act'; ...
          A_act, -s.delta*speye(n_act)];
        rhs0 = [-q; b_act];
        sol0 = Kreg \ rhs0;
        x_pol = sol0(1:n);
        y_act = sol0(n+1:end);

        % Iterative refinement around direct solve
        for refine = 1:s.polish_refine_iter
          res = rhs0 - Kreg * [x_pol; y_act];
          delta_sol = Kreg \ res;
          x_pol = x_pol + delta_sol(1:n);
          y_act = y_act + delta_sol(n+1:end);
        end

        y_pol = zeros(m, 1);
        y_pol(active_idx) = y_act;

        % Check feasibility (treating ±INF as unconstrained)
        Ax_pol = A * x_pol;
        feas_l = all(Ax_pol >= l - tol_act | abs(l) >= INF);
        feas_u = all(Ax_pol <= u + tol_act | abs(u) >= INF);
        if feas_l && feas_u
          obj_pol = 0.5 * x_pol' * Pfull * x_pol + q' * x_pol;
          results.x = x_pol;
          results.y = y_pol;
          results.info.obj_val = obj_pol;
          results.info.status_polish = 1;
        else
          results.info.status_polish = -1;
        end
      catch
        results.info.status_polish = -1;
      end
    end

  end % methods (static, private)

end % classdef OSQP
