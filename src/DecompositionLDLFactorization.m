classdef DecompositionLDLFactorization < handle
% DecompositionLDLFactorization  Wrapper using MATLAB's decomposition() object.
%
%   Provides the same mldivide interface as MATLABLDLFactorization so it
%   can be used as a drop-in replacement for benchmarking purposes.
%
%   Uses decomposition(K, 'ldl') which internally stores the LDL
%   factorization and dispatches solves through the same MATLAB sparse
%   triangular solver infrastructure as MATLABLDLFactorization.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties (Access = private)
        dK  % decomposition object (ldl type)
    end

    methods
        function obj = DecompositionLDLFactorization(K)
            % Factorize symmetric K using MATLAB's decomposition(K, 'ldl').
            obj.dK = decomposition(K, 'ldl', 'CheckCondition', false);
        end

        function x = mldivide(obj, b)
            % Solve K*x = b using the stored decomposition.
            x = obj.dK \ b;
        end
    end
end
