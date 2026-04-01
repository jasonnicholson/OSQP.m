function build_qdldl_c_mex(force)
% BUILD_QDLDL_C_MEX  Build the upstream QDLDL C MEX factorization wrapper.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    if nargin < 1
        force = false;
    end

    srcDir = fileparts(mfilename('fullpath'));
    mexTarget = fullfile(srcDir, ['qdldl_c_factor_mex.' mexext]);
    if ~force && exist(mexTarget, 'file') == 3
        return;
    end

    repoRoot = fileparts(srcDir);
    candidates = {
        fullfile(repoRoot, '..', 'QDLDL')
        fullfile(repoRoot, '..', 'qdldl')
    };

    qdldlRoot = '';
    for idx = 1:numel(candidates)
        if isfolder(candidates{idx})
            qdldlRoot = candidates{idx};
            break;
        end
    end

    if isempty(qdldlRoot)
        error('OSQP:QDLDLCBuild', 'Could not locate the upstream QDLDL C source directory.');
    end

    includeDir = fullfile(qdldlRoot, 'include');
    sourceFile = fullfile(qdldlRoot, 'src', 'qdldl.c');
    headerDir = fullfile(srcDir, 'qdldl_c_headers');
    mexSource = fullfile(srcDir, 'qdldl_c_factor_mex.c');

    if ~isfile(sourceFile) || ~isfolder(includeDir)
        error('OSQP:QDLDLCBuild', 'QDLDL source tree is incomplete: %s', qdldlRoot);
    end

    mex('-R2018a', ...
        ['-I' headerDir], ['-I' includeDir], ...
        '-output', mexTarget, ...
        mexSource, sourceFile);
end