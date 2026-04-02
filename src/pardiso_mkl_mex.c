/*
 * pardiso_mkl_mex.c — MKL Pardiso direct solver MEX interface for OSQP.m
 *
 * Commands:
 *   h = pardiso_mkl_mex('factorize', tril_K)   % factor symmetric KKT
 *   x = pardiso_mkl_mex('solve', h, b)         % triangular solve
 *       pardiso_mkl_mex('free', h)              % release factorization
 *
 * Uses LP64 interface (MKL_INT = int, 32-bit indices) matching the
 * reference OSQP MKL backend.
 *
 * Input: lower triangle of symmetric K in MATLAB sparse CSC format.
 * Internally reinterprets CSC(tril) as CSR(triu) for Pardiso — the two
 * representations are identical for symmetric matrices (just swap the
 * meaning of row/column) so only index-base adjustment (+1) is needed.
 *
 * Copyright (c) 2025 Jason H. Nicholson
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mex.h"
#include "matrix.h"
#include "mkl_pardiso.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Handle storage — static array lives in BSS, always valid.          */
/* Heap arrays (ia, ja, a) use stdlib malloc/free so they persist     */
/* across MEX calls (mxCalloc/mxMalloc are auto-freed by MATLAB).    */
/* ------------------------------------------------------------------ */

typedef struct {
    int      active;
    MKL_INT  n;
    MKL_INT  nnz;
    MKL_INT *ia;          /* 1-based CSR row pointers  (n+1) — stdlib heap */
    MKL_INT *ja;          /* 1-based CSR column indices (nnz) — stdlib heap */
    double  *a;           /* CSR values                (nnz) — stdlib heap */
    void    *pt[64];      /* Pardiso internal state (inline, zero-init)    */
    MKL_INT  iparm[64];   /* Pardiso parameters    (inline)                */
    MKL_INT  maxfct, mnum, mtype, nrhs, msglvl;
} Handle;

#define MAX_HANDLES 64
static Handle g_handles[MAX_HANDLES]; /* BSS — zero-initialised */
static int    g_init = 0;

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static void release_handle(Handle *h) {
    MKL_INT phase = -1, err = 0, idum = 0;
    double  ddum  = 0;

    if (!h || !h->active) return;

    /* Phase -1: release Pardiso internal memory */
    pardiso(h->pt, &h->maxfct, &h->mnum, &h->mtype, &phase,
            &h->n, &ddum, h->ia, h->ja, &idum, &h->nrhs,
            h->iparm, &h->msglvl, &ddum, &ddum, &err);

    free(h->ia);
    free(h->ja);
    free(h->a);
    memset(h, 0, sizeof(*h));   /* zeros active, pt[], etc. */
}

static void cleanup_all(void) {
    int i;
    for (i = 0; i < MAX_HANDLES; i++)
        release_handle(&g_handles[i]);
    g_init = 0;
}

static int alloc_slot(void) {
    int i;
    for (i = 0; i < MAX_HANDLES; i++)
        if (!g_handles[i].active) return i;
    mexErrMsgIdAndTxt("OSQP:PardisoMKL:Full",
                      "All %d Pardiso handle slots are in use.", MAX_HANDLES);
    return -1; /* unreachable */
}

static Handle *get_handle(const mxArray *mx) {
    int id;
    if (!mxIsDouble(mx) || mxGetNumberOfElements(mx) != 1)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Handle",
                          "Handle must be a scalar double.");
    id = (int)mxGetScalar(mx);
    if (id < 1 || id > MAX_HANDLES || !g_handles[id - 1].active)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Handle",
                          "Invalid or freed handle %d.", id);
    return &g_handles[id - 1];
}

/*
 * Reinterpret MATLAB CSC(tril(K)) as 1-based CSR(triu(K)).
 *
 * For a symmetric matrix K, CSC of the lower triangle and CSR of the
 * upper triangle contain identical data — only the interpretation of
 * row vs. column changes.  So the conversion is a trivial copy with
 * a +1 index-base adjustment (MATLAB 0-based → Pardiso 1-based).
 *
 * Outputs are heap-allocated with stdlib malloc (caller must free).
 */
static void csc_lower_to_csr_upper(const mxArray *L,
                                   MKL_INT **ia_p, MKL_INT **ja_p,
                                   double **a_p,
                                   MKL_INT *n_p, MKL_INT *nnz_p) {
    mwSize   n   = mxGetN(L);
    mwIndex *jc  = mxGetJc(L);
    mwIndex *ir  = mxGetIr(L);
    double  *pr  = mxGetDoubles(L);
    mwSize   nnz = jc[n];
    MKL_INT *ia, *ja;
    double  *av;
    mwSize   i;

    if ((long long)n > (long long)INT_MAX ||
        (long long)nnz > (long long)INT_MAX)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:TooLarge",
                          "Matrix exceeds LP64 index range (n=%lld, nnz=%lld).",
                          (long long)n, (long long)nnz);

    ia = (MKL_INT *)malloc((n + 1) * sizeof(MKL_INT));
    ja = (MKL_INT *)malloc(nnz     * sizeof(MKL_INT));
    av = (double  *)malloc(nnz     * sizeof(double));
    if (!ia || !ja || !av) {
        free(ia); free(ja); free(av);
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:OOM",
                          "malloc failed in CSR conversion.");
    }

    /* jc → ia  (column pointers become row pointers, +1 for base) */
    for (i = 0; i <= n; i++)
        ia[i] = (MKL_INT)(jc[i] + 1);

    /* ir → ja,  pr → a  (row indices become column indices, +1) */
    for (i = 0; i < nnz; i++) {
        ja[i] = (MKL_INT)(ir[i] + 1);
        av[i] = pr[i];
    }

    *ia_p  = ia;
    *ja_p  = ja;
    *a_p   = av;
    *n_p   = (MKL_INT)n;
    *nnz_p = (MKL_INT)nnz;
}

/* ------------------------------------------------------------------ */
/* Commands                                                           */
/* ------------------------------------------------------------------ */

static void cmd_factorize(int nlhs, mxArray *plhs[],
                          int nrhs, const mxArray *prhs[]) {
    int      slot;
    Handle  *h;
    MKL_INT  phase, err = 0, idum = 0;
    double   ddum = 0;

    if (nrhs != 2 || nlhs != 1)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Usage",
                          "h = pardiso_mkl_mex('factorize', tril_K)");

    if (!mxIsSparse(prhs[1]) || mxIsComplex(prhs[1]) || !mxIsDouble(prhs[1]))
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Input",
                          "Input must be a real sparse double matrix.");
    if (mxGetM(prhs[1]) != mxGetN(prhs[1]))
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Input",
                          "Matrix must be square.");

    slot = alloc_slot();
    h    = &g_handles[slot];
    memset(h, 0, sizeof(*h));

    csc_lower_to_csr_upper(prhs[1], &h->ia, &h->ja, &h->a, &h->n, &h->nnz);

    h->active = 1;
    h->maxfct = 1;
    h->mnum   = 1;
    h->mtype  = -2;      /* real symmetric indefinite */
    h->nrhs   = 1;
    h->msglvl = 0;

    /* Pardiso control parameters — matches reference OSQP MKL backend */
    h->iparm[0]  = 1;    /* use non-default values */
    h->iparm[1]  = 3;    /* parallel nested dissection (OpenMP) */
    h->iparm[5]  = 0;    /* write solution into x (not b) */
    h->iparm[7]  = 0;    /* iterative refinement steps (auto) */
    h->iparm[9]  = 13;   /* pivot perturbation 1e-13 */
    h->iparm[27] = 0;    /* double precision */
    h->iparm[34] = 0;    /* Fortran-style 1-based indexing */

    /* Phase 11: symbolic factorization */
    phase = 11;
    pardiso(h->pt, &h->maxfct, &h->mnum, &h->mtype, &phase,
            &h->n, h->a, h->ia, h->ja, &idum, &h->nrhs,
            h->iparm, &h->msglvl, &ddum, &ddum, &err);
    if (err) {
        release_handle(h);
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Symbolic",
                          "Symbolic factorization failed (error=%d).", (int)err);
    }

    /* Phase 22: numerical factorization */
    phase = 22;
    pardiso(h->pt, &h->maxfct, &h->mnum, &h->mtype, &phase,
            &h->n, h->a, h->ia, h->ja, &idum, &h->nrhs,
            h->iparm, &h->msglvl, &ddum, &ddum, &err);
    if (err) {
        release_handle(h);
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Numeric",
                          "Numeric factorization failed (error=%d).", (int)err);
    }

    plhs[0] = mxCreateDoubleScalar((double)(slot + 1));
}

static void cmd_solve(int nlhs, mxArray *plhs[],
                      int nrhs, const mxArray *prhs[]) {
    Handle  *h;
    MKL_INT  phase = 33, err = 0, idum = 0, c_nrhs;
    double  *bdata, *x;
    mwSize   nb, ncols;

    if (nrhs != 3 || nlhs != 1)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Usage",
                          "x = pardiso_mkl_mex('solve', h, b)");

    h = get_handle(prhs[1]);

    if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]))
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Input",
                          "b must be a real double array.");
    nb    = mxGetM(prhs[2]);
    ncols = mxGetN(prhs[2]);
    if ((MKL_INT)nb != h->n)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Size",
                          "b has %d rows, expected %d.",
                          (int)nb, (int)h->n);

    plhs[0] = mxCreateDoubleMatrix(nb, ncols, mxREAL);
    x     = mxGetDoubles(plhs[0]);
    bdata = mxGetDoubles(prhs[2]);
    c_nrhs = (MKL_INT)ncols;

    pardiso(h->pt, &h->maxfct, &h->mnum, &h->mtype, &phase,
            &h->n, h->a, h->ia, h->ja, &idum, &c_nrhs,
            h->iparm, &h->msglvl, bdata, x, &err);
    if (err)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Solve",
                          "Pardiso solve failed (error=%d).", (int)err);
}

static void cmd_free(int nlhs, mxArray *plhs[],
                     int nrhs, const mxArray *prhs[]) {
    (void)nlhs; (void)plhs;
    if (nrhs != 2)
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Usage",
                          "pardiso_mkl_mex('free', h)");
    release_handle(get_handle(prhs[1]));
}

/* ------------------------------------------------------------------ */
/* Entry point                                                        */
/* ------------------------------------------------------------------ */

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    char cmd[16];

    if (!g_init) {
        memset(g_handles, 0, sizeof(g_handles));
        mexAtExit(cleanup_all);
        g_init = 1;
    }

    if (nrhs < 1 || !mxIsChar(prhs[0]) ||
        mxGetString(prhs[0], cmd, sizeof(cmd)))
        mexErrMsgIdAndTxt("OSQP:PardisoMKL:Usage",
                          "First arg must be 'factorize', 'solve', or 'free'.");

    if      (strcmp(cmd, "factorize") == 0) cmd_factorize(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "solve")     == 0) cmd_solve(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "free")      == 0) cmd_free(nlhs, plhs, nrhs, prhs);
    else mexErrMsgIdAndTxt("OSQP:PardisoMKL:Usage",
                           "Unknown command '%s'.", cmd);
}