#include "mex.h"
#include "matrix.h"

#include "qdldl.h"

#include <limits.h>

static void copy_mwindex_to_qdldl(const mwIndex* src, QDLDL_int* dst, mwSize count) {
    mwSize idx;
    for (idx = 0; idx < count; ++idx) {
        if ((unsigned long long)src[idx] > (unsigned long long)LLONG_MAX) {
            mexErrMsgIdAndTxt("OSQP:QDLDLCMex:IndexOverflow", "Sparse index exceeds QDLDL_int range.");
        }
        dst[idx] = (QDLDL_int)src[idx];
    }
}

static void copy_qdldl_to_mwindex(const QDLDL_int* src, mwIndex* dst, mwSize count) {
    mwSize idx;
    for (idx = 0; idx < count; ++idx) {
        if (src[idx] < 0) {
            mexErrMsgIdAndTxt("OSQP:QDLDLCMex:NegativeIndex", "QDLDL returned a negative sparse index.");
        }
        dst[idx] = (mwIndex)src[idx];
    }
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    const mxArray* A;
    mwSize n;
    mwSize nnzA;
    const mwIndex* Ap_mw;
    const mwIndex* Ai_mw;
    const double* Ax;
    QDLDL_int* Ap;
    QDLDL_int* Ai;
    QDLDL_int* Lp;
    QDLDL_int* Li;
    QDLDL_float* Lx;
    QDLDL_float* D;
    QDLDL_float* Dinv;
    QDLDL_int* etree;
    QDLDL_int* Lnz;
    QDLDL_int* iwork;
    QDLDL_bool* bwork;
    QDLDL_float* fwork;
    QDLDL_int sumLnz;
    QDLDL_int positiveInertia;
    mwIndex* Lp_out;
    mwIndex* Li_out;
    double* Lx_out;
    double* Dinv_out;
    mwSize idx;

    if (nrhs != 1) {
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:InvalidNumInputs", "Expected one sparse matrix input.");
    }
    if (nlhs < 2 || nlhs > 3) {
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:InvalidNumOutputs", "Expected outputs [L, Dinv] or [L, Dinv, positive_inertia].");
    }

    A = prhs[0];
    if (!mxIsSparse(A) || !mxIsDouble(A) || mxIsComplex(A)) {
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:InvalidInput", "Input must be a real sparse double matrix.");
    }

    if (mxGetM(A) != mxGetN(A)) {
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:DimensionMismatch", "Input matrix must be square.");
    }

    n = mxGetN(A);
    Ap_mw = mxGetJc(A);
    Ai_mw = mxGetIr(A);
    Ax = mxGetDoubles(A);
    nnzA = Ap_mw[n];

    Ap = (QDLDL_int*)mxMalloc((n + 1) * sizeof(QDLDL_int));
    Ai = (QDLDL_int*)mxMalloc(nnzA * sizeof(QDLDL_int));
    copy_mwindex_to_qdldl(Ap_mw, Ap, n + 1);
    copy_mwindex_to_qdldl(Ai_mw, Ai, nnzA);

    etree = (QDLDL_int*)mxMalloc(n * sizeof(QDLDL_int));
    Lnz = (QDLDL_int*)mxMalloc(n * sizeof(QDLDL_int));
    iwork = (QDLDL_int*)mxMalloc(3 * n * sizeof(QDLDL_int));

    sumLnz = QDLDL_etree((QDLDL_int)n, Ap, Ai, iwork, Lnz, etree);
    if (sumLnz < 0) {
        mxFree(iwork);
        mxFree(Lnz);
        mxFree(etree);
        mxFree(Ai);
        mxFree(Ap);
        if (sumLnz == -1) {
            mexErrMsgIdAndTxt("OSQP:QDLDLCMex:InvalidPattern", "Matrix must contain only the upper triangle with no empty columns.");
        }
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:Overflow", "QDLDL etree overflowed its integer type.");
    }

    Lp = (QDLDL_int*)mxMalloc((n + 1) * sizeof(QDLDL_int));
    Li = (QDLDL_int*)mxMalloc((mwSize)sumLnz * sizeof(QDLDL_int));
    Lx = (QDLDL_float*)mxMalloc((mwSize)sumLnz * sizeof(QDLDL_float));
    D = (QDLDL_float*)mxMalloc(n * sizeof(QDLDL_float));
    Dinv = (QDLDL_float*)mxMalloc(n * sizeof(QDLDL_float));
    bwork = (QDLDL_bool*)mxMalloc(n * sizeof(QDLDL_bool));
    fwork = (QDLDL_float*)mxMalloc(n * sizeof(QDLDL_float));

    positiveInertia = QDLDL_factor((QDLDL_int)n, Ap, Ai, Ax, Lp, Li, Lx, D, Dinv,
        Lnz, etree, bwork, iwork, fwork);
    if (positiveInertia < 0) {
        mxFree(fwork);
        mxFree(bwork);
        mxFree(Dinv);
        mxFree(D);
        mxFree(Lx);
        mxFree(Li);
        mxFree(Lp);
        mxFree(iwork);
        mxFree(Lnz);
        mxFree(etree);
        mxFree(Ai);
        mxFree(Ap);
        mexErrMsgIdAndTxt("OSQP:QDLDLCMex:FactorizationFailed", "QDLDL factorization failed; matrix is not quasi-definite.");
    }

    plhs[0] = mxCreateSparse(n, n, (mwSize)sumLnz, mxREAL);
    Lp_out = mxGetJc(plhs[0]);
    Li_out = mxGetIr(plhs[0]);
    Lx_out = mxGetDoubles(plhs[0]);
    copy_qdldl_to_mwindex(Lp, Lp_out, n + 1);
    copy_qdldl_to_mwindex(Li, Li_out, (mwSize)sumLnz);
    for (idx = 0; idx < (mwSize)sumLnz; ++idx) {
        Lx_out[idx] = Lx[idx];
    }

    plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
    Dinv_out = mxGetDoubles(plhs[1]);
    for (idx = 0; idx < n; ++idx) {
        Dinv_out[idx] = Dinv[idx];
    }

    if (nlhs >= 3) {
        plhs[2] = mxCreateDoubleScalar((double)positiveInertia);
    }

    mxFree(fwork);
    mxFree(bwork);
    mxFree(Dinv);
    mxFree(D);
    mxFree(Lx);
    mxFree(Li);
    mxFree(Lp);
    mxFree(iwork);
    mxFree(Lnz);
    mxFree(etree);
    mxFree(Ai);
    mxFree(Ap);
}