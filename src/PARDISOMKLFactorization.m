classdef PARDISOMKLFactorization < handle
% PARDISOMKLFactorization  MKL Pardiso-backed direct factorization wrapper.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties (Access = private)
        handleId
        n
    end

    methods
        function obj = PARDISOMKLFactorization(K)
            if ~issparse(K)
                K = sparse(K);
            end
            [m, n] = size(K);
            if m ~= n
                error('OSQP:PardisoMKL', 'K must be square.');
            end
            obj.n = n;
            obj.handleId = pardiso_mkl_mex('factorize', tril(K));
        end

        function x = mldivide(obj, b)
            if size(b, 1) ~= obj.n
                error('OSQP:PardisoMKL', 'Right hand side has incompatible size.');
            end
            x = pardiso_mkl_mex('solve', obj.handleId, b);
        end

        function delete(obj)
            if ~isempty(obj.handleId)
                try
                    pardiso_mkl_mex('free', obj.handleId);
                catch
                    % Best effort cleanup only.
                end
                obj.handleId = [];
            end
        end
    end
end