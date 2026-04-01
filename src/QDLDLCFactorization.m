classdef QDLDLCFactorization < handle
% QDLDLCFactorization  QDLDL factorization backed by the upstream C code.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    properties (Access = private)
        perm
        L
        L_unit
        Dinv_vec
    end

    methods
        function obj = QDLDLCFactorization(A, perm)
            if nargin < 2
                perm = 'auto';
            end

            if ~issparse(A)
                A = sparse(A);
            end

            [m, n] = size(A);
            if m ~= n
                error('QDLDL:DimensionMismatch', 'Input matrix must be square.');
            end

            if ~istriu(A)
                A = triu(A);
            else
                A = sparse(A);
            end

            if ischar(perm) || (isstring(perm) && isscalar(perm))
                if ~strcmp(char(perm), 'auto')
                    error('QDLDL:InvalidPermutation', 'perm must be [], ''auto'', or a permutation vector.');
                end
                perm = symamd(A);
                perm = perm(:);
            elseif isempty(perm)
                perm = [];
            else
                perm = perm(:);
                if numel(perm) ~= n || ~isequal(sort(double(perm(:))), (1:n)')
                    error('QDLDL:InvalidPermutation', 'perm must be a permutation of 1:n.');
                end
            end

            if isempty(perm)
                factor_input = A;
            else
                Afull = A + A' - spdiags(diag(A), 0, n, n);
                factor_input = triu(Afull(perm, perm));
            end

            [obj.L, obj.Dinv_vec] = qdldl_c_factor_mex(factor_input);
            obj.perm = perm;
            obj.L_unit = obj.L + speye(n);
        end

        function x = mldivide(obj, b)
            if isempty(obj.perm)
                tmp = b;
            else
                tmp = b(obj.perm);
            end

            tmp = obj.L_unit \ tmp;
            tmp = tmp .* obj.Dinv_vec;
            tmp = obj.L_unit' \ tmp;

            if isempty(obj.perm)
                x = tmp;
            else
                x = b;
                x(obj.perm) = tmp;
            end
        end
    end
end