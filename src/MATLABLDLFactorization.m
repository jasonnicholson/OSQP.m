classdef MATLABLDLFactorization < handle
% MATLABLDLFactorization  Thin wrapper around MATLAB's built-in ldl().
%
%   Provides the same mldivide interface as QDLDLFactorization so it can
%   be used as a drop-in replacement inside OSQP.
%
%   MATLAB's ldl() computes K(p,p) = L * D * L' where L is unit lower
%   triangular (sparse) and D is block-diagonal (1×1 for quasi-definite K).
%   Solves are then performed using MATLAB's fast sparse triangular solver.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties (Access = private)
        L       % unit lower triangular sparse factor
        D       % block-diagonal factor (sparse); 1×1 or 2×2 Bunch-Kaufman blocks
        p       % permutation vector: K(p,p) = L*D*L'
        n       % problem dimension
    end

    methods
        function obj = MATLABLDLFactorization(K)
            % Factorize a symmetric matrix K using MATLAB's built-in ldl().
            %   K(p,p) = L * D * L'
            % D is block-diagonal: 1×1 blocks for positive/negative-definite
            % pivots, 2×2 Bunch-Kaufman blocks for indefinite pairs.
            [L_, D_, p_] = ldl(K, 'vector');
            obj.L = L_;
            obj.D = D_;
            obj.p = p_;
            obj.n = size(K, 1);
        end

        function x = mldivide(obj, b)
            % Solve K*x = b  where  K(p,p) = L * D * L'
            c = b(obj.p);              % permute RHS
            c = obj.L  \ c;            % forward  solve
            c = obj.D  \ c;            % block-diagonal solve (handles 2×2 pivots)
            c = obj.L' \ c;            % backward solve
            x      = b;               % allocate output (same class/size)
            x(obj.p) = c;             % inverse permute
        end
    end
end
