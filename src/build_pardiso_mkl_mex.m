function build_pardiso_mkl_mex(force)
% BUILD_PARDISO_MKL_MEX  Build the MKL Pardiso MEX wrapper.

% Copyright (c) 2026 Jason H. Nicholson
% SPDX-License-Identifier: Apache-2.0

    if nargin < 1
        force = false;
    end

    srcDir = fileparts(mfilename('fullpath'));
    mexTarget = fullfile(srcDir, ['pardiso_mkl_mex.' mexext]);
    if ~force && exist(mexTarget, 'file') == 3
        return;
    end

    mklRoot = '/opt/intel/oneapi/mkl/latest';
    if ~isfolder(mklRoot)
        error('OSQP:PardisoMKLBuild', 'oneMKL not found at %s. Install intel-oneapi-mkl-devel first.', mklRoot);
    end

    includeDir = fullfile(mklRoot, 'include');
    libRt = fullfile(mklRoot, 'lib', 'libmkl_rt.so');
    if ~isfile(libRt)
        altRt = fullfile(mklRoot, 'lib', 'intel64', 'libmkl_rt.so');
        if isfile(altRt)
            libRt = altRt;
        else
            error('OSQP:PardisoMKLBuild', 'Could not locate libmkl_rt.so under %s.', mklRoot);
        end
    end

    mklLibDir = fileparts(libRt);
    rpathFlag = ['LDFLAGS=$LDFLAGS -Wl,-rpath,' mklLibDir];

    mex('-R2018a', ...
        ['-I' includeDir], ...
        rpathFlag, ...
        fullfile(srcDir, 'pardiso_mkl_mex.c'), ...
        libRt, ...
        '-ldl', ...
        '-output', mexTarget);
end