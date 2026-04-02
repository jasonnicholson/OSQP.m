classdef LinsolveFactorization < handle
% LinsolveFactorization  Wrapper using linsolve() with explicit triangular opts.
%
%   Provides the same mldivide interface as MATLABLDLFactorization.
%   Uses linsolve() with opts struct on the L and L' factors to explicitly
%   declare triangular structure, potentially avoiding backslash decision-tree
%   overhead on every solve.
%
%   NOTE: MATLAB documentation states that linsolve opts apply only to
%   dense matrices; for sparse inputs linsolve falls back to the same
%   algorithm as '\'. This class exists to verify that empirically.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties (Access = private)
        L       % unit lower triangular sparse factor
        D       % block-diagonal factor (sparse)
        p       % permutation vector: K(p,p) = L*D*L'
        n       % problem dimension
        optsLT  % linsolve opts: lower triangular, unit diagonal
        optsUT  % linsolve opts: upper triangular, unit diagonal
    end

    methods
        function obj = LinsolveFactorization(K)
            [L_, D_, p_] = ldl(K, 'vector');
            obj.L = L_;
            obj.D = D_;
            obj.p = p_;
            obj.n = size(K, 1);
            obj.optsLT = struct('LT', true, 'UNIT', true);
            obj.optsUT = struct('UT', true, 'UNIT', true);
        end

        function x = mldivide(obj, b)
            % Solve K*x = b  where  K(p,p) = L * D * L'
            c = b(obj.p);
            c = linsolve(obj.L,  c, obj.optsLT);   % forward  solve (LT+UNIT)
            c = obj.D \ c;                          % block-diagonal solve
            c = linsolve(obj.L', c, obj.optsUT);    % backward solve (UT+UNIT)
            x = b;
            x(obj.p) = c;
        end
    end
end
