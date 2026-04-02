#include "mex.h"

#include <dlfcn.h>

static const char* default_symbols[] = {
    "pardiso",
    "PARDISO",
    "pardiso_64",
    "PARDISO_64",
    "mkl_pardiso",
    "MKL_PARDISO",
    "pardiso_init",
    "pardisoinit",
    "mkl_pds_lp64",
    "mkl_pds_ilp64"
};

static mxArray* make_symbol_cell_default(void) {
    mwSize i;
    mwSize n = sizeof(default_symbols) / sizeof(default_symbols[0]);
    mxArray* out = mxCreateCellMatrix((mwSize)n, 1);
    for (i = 0; i < n; ++i) {
        mxSetCell(out, i, mxCreateString(default_symbols[i]));
    }
    return out;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    const char* default_lib = "/home/jason/Programs/MATLAB/2025b/bin/glnxa64/mkl.so";
    char* libpath = NULL;
    const mxArray* symbols_in = NULL;
    mxArray* symbols_cell = NULL;
    mxArray* names = NULL;
    mxArray* found = NULL;
    mxArray* addresses = NULL;
    mxArray* open_error = NULL;
    mxArray* out = NULL;
    void* handle = NULL;
    const char* fields[] = {
        "library",
        "opened",
        "open_error",
        "symbols",
        "found",
        "addresses"
    };
    mwSize i;
    mwSize nsyms;

    if (nrhs > 2) {
        mexErrMsgIdAndTxt("OSQP:ProbeMKL:InvalidInput", "Usage: probe_mkl_symbols_mex([library_path], [symbol_cellstr]).");
    }

    if (nrhs >= 1 && !mxIsEmpty(prhs[0])) {
        if (!mxIsChar(prhs[0])) {
            mexErrMsgIdAndTxt("OSQP:ProbeMKL:InvalidLibrary", "library_path must be a char array.");
        }
        libpath = mxArrayToString(prhs[0]);
        if (!libpath) {
            mexErrMsgIdAndTxt("OSQP:ProbeMKL:Allocation", "Failed to read library_path string.");
        }
    } else {
        libpath = mxArrayToString(mxCreateString(default_lib));
    }

    if (nrhs >= 2 && !mxIsEmpty(prhs[1])) {
        symbols_in = prhs[1];
        if (!mxIsCell(symbols_in)) {
            mxFree(libpath);
            mexErrMsgIdAndTxt("OSQP:ProbeMKL:InvalidSymbols", "symbol_cellstr must be a cell array of char strings.");
        }
        symbols_cell = mxDuplicateArray(symbols_in);
    } else {
        symbols_cell = make_symbol_cell_default();
    }

    nsyms = mxGetNumberOfElements(symbols_cell);
    names = mxCreateCellMatrix(nsyms, 1);
    found = mxCreateLogicalMatrix(nsyms, 1);
    addresses = mxCreateDoubleMatrix(nsyms, 1, mxREAL);
    open_error = mxCreateString("");

    dlerror();
    handle = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        if (err) {
            mxDestroyArray(open_error);
            open_error = mxCreateString(err);
        }
    }

    for (i = 0; i < nsyms; ++i) {
        const mxArray* c = mxGetCell(symbols_cell, i);
        char* s = NULL;
        void* addr = NULL;

        if (!c || !mxIsChar(c)) {
            mxSetCell(names, i, mxCreateString(""));
            mxGetLogicals(found)[i] = false;
            mxGetDoubles(addresses)[i] = 0.0;
            continue;
        }

        s = mxArrayToString(c);
        if (!s) {
            mxSetCell(names, i, mxCreateString(""));
            mxGetLogicals(found)[i] = false;
            mxGetDoubles(addresses)[i] = 0.0;
            continue;
        }

        mxSetCell(names, i, mxCreateString(s));

        if (handle) {
            dlerror();
            addr = dlsym(handle, s);
            if (addr && dlerror() == NULL) {
                mxGetLogicals(found)[i] = true;
                mxGetDoubles(addresses)[i] = (double)(unsigned long long)addr;
            } else {
                mxGetLogicals(found)[i] = false;
                mxGetDoubles(addresses)[i] = 0.0;
            }
        } else {
            mxGetLogicals(found)[i] = false;
            mxGetDoubles(addresses)[i] = 0.0;
        }

        mxFree(s);
    }

    out = mxCreateStructMatrix(1, 1, 6, fields);
    mxSetField(out, 0, "library", mxCreateString(libpath ? libpath : ""));
    mxSetField(out, 0, "opened", mxCreateLogicalScalar(handle != NULL));
    mxSetField(out, 0, "open_error", open_error);
    mxSetField(out, 0, "symbols", names);
    mxSetField(out, 0, "found", found);
    mxSetField(out, 0, "addresses", addresses);

    if (handle) {
        dlclose(handle);
    }
    if (libpath) {
        mxFree(libpath);
    }
    if (symbols_cell) {
        mxDestroyArray(symbols_cell);
    }

    if (nlhs > 0) {
        plhs[0] = out;
    } else {
        mxDestroyArray(out);
    }
}