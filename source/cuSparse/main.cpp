/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <typeinfo> // for usage of C++ typeid
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <vector>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "helper_cusolver.h"
#include "helper_cuda.h"
#include "cusolverSp.h"
#include <chrono>
#include "../core/sparse_matrix.h"
using namespace Core;
//profiling the code
#define TIME_INDIVIDUAL_LIBRARY_CALLS

#define DBICGSTAB_MAX_ULP_ERR   100
#define DBICGSTAB_EPS           1.E-14f //9e-2

#define CLEANUP()                       \
do {                                    \
    if (x)          free (x);           \
    if (f)          free (f);           \
    if (r)          free (r);           \
    if (rw)         free (rw);          \
    if (p)          free (p);           \
    if (pw)         free (pw);          \
    if (s)          free (s);           \
    if (t)          free (t);           \
    if (v)          free (v);           \
    if (tx)         free (tx);          \
    if (Aval)       free(Aval);         \
    if (AcolsIndex) free(AcolsIndex);   \
    if (ArowsIndex) free(ArowsIndex);   \
    if (Mval)       free(Mval);         \
    if (devPtrX)    checkCudaErrors(cudaFree (devPtrX));                    \
    if (devPtrF)    checkCudaErrors(cudaFree (devPtrF));                    \
    if (devPtrR)    checkCudaErrors(cudaFree (devPtrR));                    \
    if (devPtrRW)   checkCudaErrors(cudaFree (devPtrRW));                   \
    if (devPtrP)    checkCudaErrors(cudaFree (devPtrP));                    \
    if (devPtrS)    checkCudaErrors(cudaFree (devPtrS));                    \
    if (devPtrT)    checkCudaErrors(cudaFree (devPtrT));                    \
    if (devPtrV)    checkCudaErrors(cudaFree (devPtrV));                    \
    if (devPtrAval) checkCudaErrors(cudaFree (devPtrAval));                 \
    if (devPtrAcolsIndex) checkCudaErrors(cudaFree (devPtrAcolsIndex));     \
    if (devPtrArowsIndex) checkCudaErrors(cudaFree (devPtrArowsIndex));     \
    if (devPtrMval)       checkCudaErrors(cudaFree (devPtrMval));           \
    if (stream)           checkCudaErrors(cudaStreamDestroy(stream));       \
    if (cublasHandle)     checkCudaErrors(cublasDestroy(cublasHandle));     \
    if (cusparseHandle)   checkCudaErrors(cusparseDestroy(cusparseHandle)); \
    fflush (stdout);                                    \
} while (0)



//////////////////////////////////////////////////////////////////////////
static void gpu_pbicgstab(cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle, int m, int n, int nnz,
	const cusparseMatDescr_t descra, /* the coefficient matrix in CSR format */
	double *a, int *ia, int *ja,
	const cusparseMatDescr_t descrm, /* the preconditioner in CSR format, lower & upper triangular factor */
	double *vm, int *im, int *jm,
	cusparseSolveAnalysisInfo_t info_l, cusparseSolveAnalysisInfo_t info_u, /* the analysis of the lower and upper triangular parts */
	double *f, double *r, double *rw, double *p, double *pw, double *s, double *t, double *v, double *x,
	int maxit, double tol, double ttt_sv)
{
	double rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
	double nrmr, nrmr0;
	rho = 0.0;
	double zero = 0.0;
	double one = 1.0;
	double mone = -1.0;
	int i = 0;
	int j = 0;
	double ttl, ttl2, ttu, ttu2, ttm, ttm2;
	double ttt_mv = 0.0;

	//WARNING: Analysis is done outside of the function (and the time taken by it is passed to the function in variable ttt_sv)

	//compute initial residual r0=b-Ax0 (using initial guess in x)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	checkCudaErrors(cudaDeviceSynchronize());
	ttm = second();
#endif

	checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, x, &zero, r));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	cudaDeviceSynchronize();
	ttm2 = second();
	ttt_mv += (ttm2 - ttm);
	//printf("matvec %f (s)\n",ttm2-ttm);
#endif
	checkCudaErrors(cublasDscal(cublasHandle, n, &mone, r, 1));
	checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, f, 1, r, 1));
	//copy residual r into r^{\hat} and p
	checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, rw, 1));
	checkCudaErrors(cublasDcopy(cublasHandle, n, r, 1, p, 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr0));
	//printf("gpu, init residual:norm %20.16f\n",nrmr0); 

	for (i = 0; i<maxit; ) {
		rhop = rho;
		checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, r, 1, &rho));

		if (i > 0) {
			beta = (rho / rhop) * (alpha / omega);
			negomega = -omega;
			checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, v, 1, p, 1));
			checkCudaErrors(cublasDscal(cublasHandle, n, &beta, p, 1));
			checkCudaErrors(cublasDaxpy(cublasHandle, n, &one, r, 1, p, 1));
		}
		//preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_l, p, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl2 = second();
		ttu = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_u, t, pw));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttu2 = second();
		ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
		//printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif

		//matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm = second();
#endif

		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, pw, &zero, v));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm2 = second();
		ttt_mv += (ttm2 - ttm);
		//printf("matvec %f (s)\n",ttm2-ttm);
#endif

		checkCudaErrors(cublasDdot(cublasHandle, n, rw, 1, v, 1, &temp));
		alpha = rho / temp;
		negalpha = -(alpha);
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &negalpha, v, 1, r, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &alpha, pw, 1, x, 1));
		checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

		if (nrmr < tol*nrmr0) {
			j = 5;
			break;
		}

		//preconditioning step (lower and upper triangular solve)
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_l, r, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttl2 = second();
		ttu = second();
#endif
		checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
		checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
		checkCudaErrors(cusparseDcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, &one, descrm, vm, im, jm, info_u, t, s));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttu2 = second();
		ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
		//printf("solve lower %f (s), upper %f (s) \n",ttl2-ttl,ttu2-ttu);
#endif
		//matrix-vector multiplication
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm = second();
#endif

		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descra, a, ia, ja, s, &zero, t));
#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
		checkCudaErrors(cudaDeviceSynchronize());
		ttm2 = second();
		ttt_mv += (ttm2 - ttm);
		//printf("matvec %f (s)\n",ttm2-ttm);
#endif

		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, r, 1, &temp));
		checkCudaErrors(cublasDdot(cublasHandle, n, t, 1, t, 1, &temp2));
		omega = temp / temp2;
		negomega = -(omega);
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &omega, s, 1, x, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, n, &negomega, t, 1, r, 1));

		checkCudaErrors(cublasDnrm2(cublasHandle, n, r, 1, &nrmr));

		if (nrmr < tol*nrmr0) {
			i++;
			j = 0;
			break;
		}
		i++;
	}

#ifdef TIME_INDIVIDUAL_LIBRARY_CALLS
	printf("gpu total solve time %f (s), matvec time %f (s)\n", ttt_sv, ttt_mv);
#endif
}


int test_bicgstab(const MatrixCRS& matr, const std::vector<double>& _b, std::vector<double>&_x, 
	int symmetrize, int debug, double damping, int maxit, double tol,
	float err, float eps) 
{
	cublasHandle_t cublasHandle = 0;
	cusparseHandle_t cusparseHandle = 0;
	cusparseMatDescr_t descra = 0;
	cusparseMatDescr_t descrm = 0;
	cudaStream_t stream = 0;
	cusparseSolveAnalysisInfo_t info_l = 0;
	cusparseSolveAnalysisInfo_t info_u = 0;
	cusparseStatus_t status1, status2, status3;
	double *devPtrAval = 0;
	int    *devPtrAcolsIndex = 0;
	int    *devPtrArowsIndex = 0;
	double *devPtrMval = 0;
	int    *devPtrMcolsIndex = 0;
	int    *devPtrMrowsIndex = 0;
	double *devPtrX = 0;
	double *devPtrF = 0;
	double *devPtrR = 0;
	double *devPtrRW = 0;
	double *devPtrP = 0;
	double *devPtrPW = 0;
	double *devPtrS = 0;
	double *devPtrT = 0;
	double *devPtrV = 0;
	double *Aval = 0;
	int    *AcolsIndex = 0;
	int    *ArowsIndex = 0;
	double *Mval = 0;
	int    *MrowsIndex = 0;
	int    *McolsIndex = 0;
	double *x = 0;
	double *tx = 0;
	double *f = 0;
	double *r = 0;
	double *rw = 0;
	double *p = 0;
	double *pw = 0;
	double *s = 0;
	double *t = 0;
	double *v = 0;
	int matrixM;
	int matrixN;
	int matrixSizeAval, matrixSizeAcolsIndex, matrixSizeArowsIndex, mSizeAval, mSizeAcolsIndex, mSizeArowsIndex;
	int arraySizeX, arraySizeF, arraySizeR, arraySizeRW, arraySizeP, arraySizePW, arraySizeS, arraySizeT, arraySizeV, nnz, mNNZ;
	long long flops;
	double start, stop;
	int num_iterations, nbrTests, count, base, mbase;
	cusparseOperation_t trans;
	double alpha;
	double ttt_sv = 0.0;
	

	//printf("Testing %cbicgstab\n", *element_type);

	alpha = damping;
	trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

	/* load the coefficient matrix */
	//if (loadMMSparseMatrix(matrix_filename, *element_type, true, &matrixM, &matrixN, &nnz, &Aval, &ArowsIndex, &AcolsIndex, symmetrize)) 
	{
		Aval = matr.data;
		ArowsIndex = (int*)matr.rowPtr;
		AcolsIndex = (int*)matr.colInd;
		matrixM = matr.rowPtrSize;
		matrixN = matr.rowPtrSize;
		nnz = matr.dataSize;
	}

	matrixSizeAval = nnz;
	matrixSizeAcolsIndex = matrixSizeAval;
	matrixSizeArowsIndex = matrixM + 1;
	base = ArowsIndex[0];
	if (matrixM != matrixN) {
		fprintf(stderr, "!!!! matrix MUST be square, error: m=%d != n=%d\n", matrixM, matrixN);
		return EXIT_FAILURE;
	}
	printf("^^^^ M=%d, N=%d, nnz=%d\n", matrixM, matrixN, nnz);

	/* set some extra parameters for lower triangular factor */
	mNNZ = ArowsIndex[matrixM] - ArowsIndex[0];
	mSizeAval = mNNZ;
	mSizeAcolsIndex = mSizeAval;
	mSizeArowsIndex = matrixM + 1;
	mbase = ArowsIndex[0];

	/* compressed sparse row */
	arraySizeX = matrixN;
	arraySizeF = matrixM;
	arraySizeR = matrixM;
	arraySizeRW = matrixM;
	arraySizeP = matrixN;
	arraySizePW = matrixN;
	arraySizeS = matrixM;
	arraySizeT = matrixM;
	arraySizeV = matrixM;

	/* initialize cublas */
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	/* initialize cusparse */
	status1 = cusparseCreate(&cusparseHandle);
	if (status1 != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "!!!! CUSPARSE initialization error\n");
		return EXIT_FAILURE;
	}
	/* create three matrix descriptors */
	status1 = cusparseCreateMatDescr(&descra);
	status2 = cusparseCreateMatDescr(&descrm);
	if ((status1 != CUSPARSE_STATUS_SUCCESS) ||
		(status2 != CUSPARSE_STATUS_SUCCESS)) {
		fprintf(stderr, "!!!! CUSPARSE cusparseCreateMatDescr (coefficient matrix or preconditioner) error\n");
		return EXIT_FAILURE;
	}

	/* allocate device memory for csr matrix and vectors */
	checkCudaErrors(cudaMalloc((void**)&devPtrX, sizeof(devPtrX[0]) * arraySizeX));
	checkCudaErrors(cudaMalloc((void**)&devPtrF, sizeof(devPtrF[0]) * arraySizeF));
	checkCudaErrors(cudaMalloc((void**)&devPtrR, sizeof(devPtrR[0]) * arraySizeR));
	checkCudaErrors(cudaMalloc((void**)&devPtrRW, sizeof(devPtrRW[0])* arraySizeRW));
	checkCudaErrors(cudaMalloc((void**)&devPtrP, sizeof(devPtrP[0]) * arraySizeP));
	checkCudaErrors(cudaMalloc((void**)&devPtrPW, sizeof(devPtrPW[0])* arraySizePW));
	checkCudaErrors(cudaMalloc((void**)&devPtrS, sizeof(devPtrS[0]) * arraySizeS));
	checkCudaErrors(cudaMalloc((void**)&devPtrT, sizeof(devPtrT[0]) * arraySizeT));
	checkCudaErrors(cudaMalloc((void**)&devPtrV, sizeof(devPtrV[0]) * arraySizeV));
	checkCudaErrors(cudaMalloc((void**)&devPtrAval, sizeof(devPtrAval[0]) * matrixSizeAval));
	checkCudaErrors(cudaMalloc((void**)&devPtrAcolsIndex, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
	checkCudaErrors(cudaMalloc((void**)&devPtrArowsIndex, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
	checkCudaErrors(cudaMalloc((void**)&devPtrMval, sizeof(devPtrMval[0]) * mSizeAval));

	/* allocate host memory for  vectors */
	x = (double *)malloc(arraySizeX * sizeof(x[0]));
	f = (double *)malloc(arraySizeF * sizeof(f[0]));
	r = (double *)malloc(arraySizeR * sizeof(r[0]));
	rw = (double *)malloc(arraySizeRW * sizeof(rw[0]));
	p = (double *)malloc(arraySizeP * sizeof(p[0]));
	pw = (double *)malloc(arraySizePW * sizeof(pw[0]));
	s = (double *)malloc(arraySizeS * sizeof(s[0]));
	t = (double *)malloc(arraySizeT * sizeof(t[0]));
	v = (double *)malloc(arraySizeV * sizeof(v[0]));
	tx = (double *)malloc(arraySizeX * sizeof(tx[0]));
	Mval = (double *)malloc(matrixSizeAval * sizeof(Mval[0]));
	if ((!Aval) || (!AcolsIndex) || (!ArowsIndex) || (!Mval) ||
		(!x) || (!f) || (!r) || (!rw) || (!p) || (!pw) || (!s) || (!t) || (!v) || (!tx)) {
		CLEANUP();
		fprintf(stderr, "!!!! memory allocation error\n");
		return EXIT_FAILURE;
	}
	/* use streams */
	int useStream = 0;
	if (useStream) {

		checkCudaErrors(cudaStreamCreate(&stream));

		if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS) {
			CLEANUP();
			fprintf(stderr, "!!!! cannot set CUBLAS stream\n");
			return EXIT_FAILURE;
		}
		status1 = cusparseSetStream(cusparseHandle, stream);
		if (status1 != CUSPARSE_STATUS_SUCCESS) {
			CLEANUP();
			fprintf(stderr, "!!!! cannot set CUSPARSE stream\n");
			return EXIT_FAILURE;
		}
	}

	/* clean memory */
	checkCudaErrors(cudaMemset((void *)devPtrX, 0, sizeof(devPtrX[0])          * arraySizeX));
	checkCudaErrors(cudaMemset((void *)devPtrF, 0, sizeof(devPtrF[0])          * arraySizeF));
	checkCudaErrors(cudaMemset((void *)devPtrR, 0, sizeof(devPtrR[0])          * arraySizeR));
	checkCudaErrors(cudaMemset((void *)devPtrRW, 0, sizeof(devPtrRW[0])         * arraySizeRW));
	checkCudaErrors(cudaMemset((void *)devPtrP, 0, sizeof(devPtrP[0])          * arraySizeP));
	checkCudaErrors(cudaMemset((void *)devPtrPW, 0, sizeof(devPtrPW[0])         * arraySizePW));
	checkCudaErrors(cudaMemset((void *)devPtrS, 0, sizeof(devPtrS[0])          * arraySizeS));
	checkCudaErrors(cudaMemset((void *)devPtrT, 0, sizeof(devPtrT[0])          * arraySizeT));
	checkCudaErrors(cudaMemset((void *)devPtrV, 0, sizeof(devPtrV[0])          * arraySizeV));
	checkCudaErrors(cudaMemset((void *)devPtrAval, 0, sizeof(devPtrAval[0])       * matrixSizeAval));
	checkCudaErrors(cudaMemset((void *)devPtrAcolsIndex, 0, sizeof(devPtrAcolsIndex[0]) * matrixSizeAcolsIndex));
	checkCudaErrors(cudaMemset((void *)devPtrArowsIndex, 0, sizeof(devPtrArowsIndex[0]) * matrixSizeArowsIndex));
	checkCudaErrors(cudaMemset((void *)devPtrMval, 0, sizeof(devPtrMval[0])       * mSizeAval));

	memset(x, 0, arraySizeX * sizeof(x[0]));
	memcpy(f, &_b[0], sizeof(_b[0])*_b.size());
	//memset(f, 0, arraySizeF * sizeof(f[0]));
	memset(r, 0, arraySizeR * sizeof(r[0]));
	memset(rw, 0, arraySizeRW * sizeof(rw[0]));
	memset(p, 0, arraySizeP * sizeof(p[0]));
	memset(pw, 0, arraySizePW * sizeof(pw[0]));
	memset(s, 0, arraySizeS * sizeof(s[0]));
	memset(t, 0, arraySizeT * sizeof(t[0]));
	memset(v, 0, arraySizeV * sizeof(v[0]));
	memset(tx, 0, arraySizeX * sizeof(tx[0]));

	/* create the test matrix and vectors on the host */
	checkCudaErrors(cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL));
	if (base) {
		checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ONE));
	}
	else {
		checkCudaErrors(cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO));
	}
	checkCudaErrors(cusparseSetMatType(descrm, CUSPARSE_MATRIX_TYPE_GENERAL));
	if (mbase) {
		checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ONE));
	}
	else {
		checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ZERO));
	}

	//compute the right-hand-side f=A*e, where e=[1, ..., 1]'
	for (int i = 0; i<arraySizeP; i++) {
		p[i] = 1.0;
	}

	/* copy the csr matrix and vectors into device memory */
	double start_matrix_copy, stop_matrix_copy, start_preconditioner_copy, stop_preconditioner_copy;

	start_matrix_copy = second();

	checkCudaErrors(cudaMemcpy(devPtrAval, Aval, (size_t)(matrixSizeAval * sizeof(Aval[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrAcolsIndex, AcolsIndex, (size_t)(matrixSizeAcolsIndex * sizeof(AcolsIndex[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrArowsIndex, ArowsIndex, (size_t)(matrixSizeArowsIndex * sizeof(ArowsIndex[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrMval, devPtrAval, (size_t)(matrixSizeAval * sizeof(devPtrMval[0])), cudaMemcpyDeviceToDevice));

	stop_matrix_copy = second();

	fprintf(stdout, "Copy matrix from CPU to GPU, time(s) = %10.8f\n", stop_matrix_copy - start_matrix_copy);

	checkCudaErrors(cudaMemcpy(devPtrX, x, (size_t)(arraySizeX * sizeof(devPtrX[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrF, f, (size_t)(arraySizeF * sizeof(devPtrF[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrR, r, (size_t)(arraySizeR * sizeof(devPtrR[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrRW, rw, (size_t)(arraySizeRW * sizeof(devPtrRW[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrP, p, (size_t)(arraySizeP * sizeof(devPtrP[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrPW, pw, (size_t)(arraySizePW * sizeof(devPtrPW[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrS, s, (size_t)(arraySizeS * sizeof(devPtrS[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrT, t, (size_t)(arraySizeT * sizeof(devPtrT[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(devPtrV, v, (size_t)(arraySizeV * sizeof(devPtrV[0])), cudaMemcpyHostToDevice));

	/* --- GPU --- */
	/* create the analysis info (for lower and upper triangular factors) */
	checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
	checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));

	/* analyse the lower and upper triangular factors */
	double ttl = second();
	checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_LOWER));
	checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_UNIT));
	checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_l));
	checkCudaErrors(cudaDeviceSynchronize());
	double ttl2 = second();

	double ttu = second();
	checkCudaErrors(cusparseSetMatFillMode(descrm, CUSPARSE_FILL_MODE_UPPER));
	checkCudaErrors(cusparseSetMatDiagType(descrm, CUSPARSE_DIAG_TYPE_NON_UNIT));
	checkCudaErrors(cusparseDcsrsv_analysis(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrm, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex, info_u));
	checkCudaErrors(cudaDeviceSynchronize());
	double ttu2 = second();
	ttt_sv += (ttl2 - ttl) + (ttu2 - ttu);
	printf("analysis lower %f (s), upper %f (s) \n", ttl2 - ttl, ttu2 - ttu);

	/* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
	double start_ilu, stop_ilu;
	printf("CUSPARSE csrilu0 ");
	start_ilu = second();
	devPtrMrowsIndex = devPtrArowsIndex;
	devPtrMcolsIndex = devPtrAcolsIndex;
	checkCudaErrors(cusparseDcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, descra, devPtrMval, devPtrArowsIndex, devPtrAcolsIndex, info_l));
	checkCudaErrors(cudaDeviceSynchronize());
	stop_ilu = second();
	fprintf(stdout, "time(s) = %10.8f \n", stop_ilu - start_ilu);

	/* run the test */
	num_iterations = 1; //10; 
	start = second() / num_iterations;
	for (count = 0; count<num_iterations; count++) {

		gpu_pbicgstab(cublasHandle, cusparseHandle, matrixM, matrixN, nnz,
			descra, devPtrAval, devPtrArowsIndex, devPtrAcolsIndex,
			descrm, devPtrMval, devPtrMrowsIndex, devPtrMcolsIndex,
			info_l, info_u,
			devPtrF, devPtrR, devPtrRW, devPtrP, devPtrPW, devPtrS, devPtrT, devPtrV, devPtrX, maxit, tol, ttt_sv);

		checkCudaErrors(cudaDeviceSynchronize());
	}
	stop = second() / num_iterations;

	/* destroy the analysis info (for lower and upper triangular factors) */
	checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l));
	checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u));

	/* copy the result into host memory */
	checkCudaErrors(cudaMemcpy(tx, devPtrX, (size_t)(arraySizeX * sizeof(tx[0])), cudaMemcpyDeviceToHost));

	return EXIT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////
void testCG(const MatrixCRS& matr, const std::vector<double>& _b, std::vector<double>& _x, const double tol, const int max_iter)
{
	int M = matr.rowPtrSize;
	int N = matr.rowPtrSize;
	int nz = matr.dataSize;
	int *I = (int*)matr.rowPtr;
	int *J = (int*)matr.colInd;
	double *val = matr.data;
	double *x;
	double *rhs;
	double a, b, na, r0, r1;
	int *d_col, *d_row;
	double *d_val, *d_x, dot;
	double *d_r, *d_p, *d_Ax;
	int k;
	double alpha, beta, alpham1;

	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	int devID = findCudaDevice(0, nullptr);
	if (devID < 0)
	{
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// Statistics about the GPU device
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
		deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

	/* Generate a random tridiagonal symmetric matrix in CSR format */
	x   = (double *)malloc(sizeof(double)*N);
	rhs = (double *)malloc(sizeof(double)*N);

	for (int i = 0; i < N; i++)
	{
		rhs[i] = _b[i];
		x[i] = 0.0;
	}

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	checkCudaErrors(cudaMalloc((void **)&d_col, nz * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N * sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N * sizeof(double)));

	cudaMemcpy(d_col, J, nz * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N * sizeof(double), cudaMemcpyHostToDevice);

	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;
	auto start = std::chrono::system_clock::now();
	cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);

	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

	k = 1;

	while (r1 > tol*tol && k <= max_iter)
	{
		if (k > 1)
		{
			b = r1 / r0;
			cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
			cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
		}
		else
		{
			cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
		}

		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
		cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
		a = r1 / dot;

		cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
		na = -a;
		cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);

		r0 = r1;
		cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		cudaDeviceSynchronize();
		k++;
	}
	auto end = std::chrono::system_clock::now();
	auto time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
	printf("iteration = %3d, residual = %e\n, time = %f\n", k, sqrt(r1), time);
	cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

	float rsum, diff, err = 0.0;

	for (int i = 0; i < N; i++)
	{
		rsum = 0.0;

		for (int j = I[i]; j < I[i + 1]; j++)
		{
			rsum += val[j] * x[J[j]];
		}

		diff = fabs(rsum - rhs[i]);

		if (diff > err)
		{
			err = diff;
		}
	}

	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	free(I);
	free(J);
	free(val);
	free(x);
	free(rhs);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);
	printf("Test Summary:  Error amount = %f\n", err);
}
#define CUSOLVER_COMMON_H_
//////////////////////////////////////////////////////////////////////////
void testSolverSP(const MatrixCRS& matr, const std::vector<double>& _b, std::vector<double>& _x, const double tol, const int max_iter)
{
	struct testOpts opts;
	opts.reorder = NULL;
	opts.testFunc = "chol";
	cusolverSpHandle_t handle = NULL;
	cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
	cudaStream_t stream = NULL;
	cusparseMatDescr_t descrA = NULL;

	int rowsA = matr.rowPtrSize; /* number of rows of A */
	int colsA = matr.rowPtrSize; /* number of columns of A */
	int nnzA = matr.dataSize; /* number of nonzeros of A */
	int baseA = 0; /* base index in CSR format */

				   /* CSR(A) from I/O */
	int *h_csrRowPtrA = (int*)matr.rowPtr;
	int *h_csrColIndA = (int*)matr.colInd;
	double *h_csrValA = matr.data;

	double *h_z = NULL; /* z = B \ (Q*b) */
	double *h_x = NULL; /* x = A \ b */
	double *h_b = NULL; /* b = ones(n,1) */
	double *h_Qb = NULL; /* Q*b */
	double *h_r = NULL; /* r = b - A*x */

	int *h_Q = NULL; /* <int> n */
					 /* reorder to reduce zero fill-in */
					 /* Q = symrcm(A) or Q = symamd(A) */
					 /* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
	int *h_csrRowPtrB = NULL; /* <int> n+1 */
	int *h_csrColIndB = NULL; /* <int> nnzA */
	double *h_csrValB = NULL; /* <double> nnzA */
	int *h_mapBfromA = NULL;  /* <int> nnzA */

	size_t size_perm = 0;
	void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

							 /* device copy of A: used in residual evaluation */
	int *d_csrRowPtrA = NULL;
	int *d_csrColIndA = NULL;
	double *d_csrValA = NULL;

	/* device copy of B: used in B*z = Q*b */
	int *d_csrRowPtrB = NULL;
	int *d_csrColIndB = NULL;
	double *d_csrValB = NULL;

	int *d_Q = NULL; /* device copy of h_Q */
	double *d_z = NULL; /* z = B \ Q*b */
	double *d_x = NULL; /* x = A \ b */
	double *d_b = NULL; /* a copy of h_b */
	double *d_Qb = NULL; /* a copy of h_Qb */
	double *d_r = NULL; /* r = b - A*x */

	const int reorder = 0; /* no reordering */
	int singularity = 0; /* -1 if A is invertible under tol. */

						 /* the constants are used in residual evaluation, r = b - A*x */
	const double minus_one = -1.0;
	const double one = 1.0;

	double b_inf = 0.0;
	double x_inf = 0.0;
	double r_inf = 0.0;
	double A_inf = 0.0;
	int errors = 0;
	int issym = 0;

	double start, stop;
	double time_solve_cpu;
	double time_solve_gpu;

	printf("step 1: read matrix market format\n");
	baseA = h_csrRowPtrA[0]; // baseA = {0,1}
	printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

	cusolverSpCreate(&handle);
 	cusparseCreate(&cusparseHandle);

	(cudaStreamCreate(&stream));
	/* bind stream to cusparse and cusolver*/
	(cusolverSpSetStream(handle, stream));
	(cusparseSetStream(cusparseHandle, stream));

	/* configure matrix descriptor*/
	(cusparseCreateMatDescr(&descrA));
	(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	if (baseA)
	{
		(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
	}
	else
	{
		(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
	}

	h_z = (double*)malloc(sizeof(double)*colsA);
	h_x = (double*)malloc(sizeof(double)*colsA);
	h_b = (double*)malloc(sizeof(double)*rowsA);
	h_Qb = (double*)malloc(sizeof(double)*rowsA);
	h_r = (double*)malloc(sizeof(double)*rowsA);
	for (int i = 0; i < rowsA; i++)
	{
		h_b[i] = _b[i];
	}

	h_Q = (int*)malloc(sizeof(int)*colsA);
	h_csrRowPtrB = (int*)malloc(sizeof(int)*(rowsA + 1));
	h_csrColIndB = (int*)malloc(sizeof(int)*nnzA);
	h_csrValB = (double*)malloc(sizeof(double)*nnzA);
	h_mapBfromA = (int*)malloc(sizeof(int)*nnzA);

	(cudaMalloc((void **)&d_csrRowPtrA, sizeof(int)*(rowsA + 1)));
	(cudaMalloc((void **)&d_csrColIndA, sizeof(int)*nnzA));
	(cudaMalloc((void **)&d_csrValA, sizeof(double)*nnzA));
	(cudaMalloc((void **)&d_csrRowPtrB, sizeof(int)*(rowsA + 1)));
	(cudaMalloc((void **)&d_csrColIndB, sizeof(int)*nnzA));
	(cudaMalloc((void **)&d_csrValB, sizeof(double)*nnzA));
	(cudaMalloc((void **)&d_Q, sizeof(int)*colsA));
	(cudaMalloc((void **)&d_z, sizeof(double)*colsA));
	(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
	(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
	(cudaMalloc((void **)&d_Qb, sizeof(double)*rowsA));
	(cudaMalloc((void **)&d_r, sizeof(double)*rowsA));

	/* verify if A has symmetric pattern or not */
// 	(cusolverSpXcsrissymHost(
// 		handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrRowPtrA + 1, h_csrColIndA, &issym));

	issym = true;
	if (!issym)
	{
		printf("Error: A has no symmetric pattern, please use LU or QR \n");
	}
	
	printf("step 2: reorder the matrix A to minimize zero fill-in\n");
	printf("        if the user choose a reordering by -P=symrcm or -P=symamd\n");

	if (NULL != opts.reorder)
	{
		if (0 == strcmp(opts.reorder, "symrcm"))
		{
			printf("step 2.1: Q = symrcm(A) \n");
			(cusolverSpXcsrsymrcmHost(
				handle, rowsA, nnzA,
				descrA, h_csrRowPtrA, h_csrColIndA,
				h_Q));
		}
		else if (0 == strcmp(opts.reorder, "symamd"))
		{
			printf("step 2.1: Q = symamd(A) \n");
			(cusolverSpXcsrsymamdHost(
				handle, rowsA, nnzA,
				descrA, h_csrRowPtrA, h_csrColIndA,
				h_Q));
		}
		else
		{
			fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
			return;
		}
	}
	else {
		printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
		for (int j = 0; j < rowsA; j++) {
			h_Q[j] = j;
		}
	}

	printf("step 2.2: B = A(Q,Q) \n");

	memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int)*(rowsA + 1));
	memcpy(h_csrColIndB, h_csrColIndA, sizeof(int)*nnzA);

	(cusolverSpXcsrperm_bufferSizeHost(
		handle, rowsA, colsA, nnzA,
		descrA, h_csrRowPtrB, h_csrColIndB,
		h_Q, h_Q,
		&size_perm));

	if (buffer_cpu)
	{
		free(buffer_cpu);
	}
	buffer_cpu = (void*)malloc(sizeof(char)*size_perm);
	//assert(NULL != buffer_cpu);

	/* h_mapBfromA = Identity */
	for (int j = 0; j < nnzA; j++)
	{
		h_mapBfromA[j] = j;
	}
	(cusolverSpXcsrpermHost(
		handle, rowsA, colsA, nnzA,
		descrA, h_csrRowPtrB, h_csrColIndB,
		h_Q, h_Q,
		h_mapBfromA,
		buffer_cpu));

	/* B = A( mapBfromA ) */
	for (int j = 0; j < nnzA; j++)
	{
		h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
	}

//	printf("step 3: b(j) = 1 + j/n \n");
// 	for (int row = 0; row < rowsA; row++)
// 	{
// 		h_b[row] = 1.0 + ((double)row) / ((double)rowsA);
// 	}

	/* h_Qb = b(Q) */
	for (int row = 0; row < rowsA; row++) {
		h_Qb[row] = h_b[h_Q[row]];
	}

	printf("step 4: prepare data on device\n");
	(cudaMemcpyAsync(d_csrRowPtrA, h_csrRowPtrA, sizeof(int)*(rowsA + 1), cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_csrColIndA, h_csrColIndA, sizeof(int)*nnzA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_csrValA, h_csrValA, sizeof(double)*nnzA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_csrRowPtrB, h_csrRowPtrB, sizeof(int)*(rowsA + 1), cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_csrColIndB, h_csrColIndB, sizeof(int)*nnzA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_csrValB, h_csrValB, sizeof(double)*nnzA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_Qb, h_Qb, sizeof(double)*rowsA, cudaMemcpyHostToDevice, stream));
	(cudaMemcpyAsync(d_Q, h_Q, sizeof(int)*rowsA, cudaMemcpyHostToDevice, stream));

	printf("step 5: solve A*x = b on CPU \n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol"))
	{
		(cusolverSpDcsrlsvcholHost(
			handle, rowsA, nnzA,
			descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else if (0 == strcmp(opts.testFunc, "lu"))
	{
		(cusolverSpDcsrlsvluHost(
			handle, rowsA, nnzA,
			descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));

	}
	else if (0 == strcmp(opts.testFunc, "qr"))
	{
		(cusolverSpDcsrlsvqrHost(
			handle, rowsA, nnzA,
			descrA, h_csrValB, h_csrRowPtrB, h_csrColIndB,
			h_Qb, tol, reorder, h_z, &singularity));
	}
	else
	{
		fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
		return ;
	}

	/* Q*x = z */
	for (int row = 0; row < rowsA; row++) {
		h_x[h_Q[row]] = h_z[row];
	}

	if (0 <= singularity)
	{
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
	}

	stop = second();
	time_solve_cpu = stop - start;

	printf("step 6: evaluate residual r = b - A*x (result on CPU)\n");
	(cudaMemcpyAsync(d_r, d_b, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice, stream));
	(cudaMemcpyAsync(d_x, h_x, sizeof(double)*colsA, cudaMemcpyHostToDevice, stream));
	(cusparseDcsrmv(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		rowsA,
		colsA,
		nnzA,
		&minus_one,
		descrA,
		d_csrValA,
		d_csrRowPtrA,
		d_csrColIndA,
		d_x,
		&one,
		d_r));
	(cudaMemcpyAsync(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost, stream));
	/* wait until h_r is ready */
	(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);
	A_inf = csr_mat_norminf(rowsA, colsA, nnzA, descrA, h_csrValA, h_csrRowPtrA, h_csrColIndA);

	printf("(CPU) |b - A*x| = %E \n", r_inf);
	printf("(CPU) |A| = %E \n", A_inf);
	printf("(CPU) |x| = %E \n", x_inf);
	printf("(CPU) |b| = %E \n", b_inf);
	printf("(CPU) |b - A*x|/(|A|*|x| + |b|) = %E \n", r_inf / (A_inf * x_inf + b_inf));

	printf("step 7: solve A*x = b on GPU\n");
	start = second();

	/* solve B*z = Q*b */
	if (0 == strcmp(opts.testFunc, "chol"))
	{
		(cusolverSpDcsrlsvchol(
			handle, rowsA, nnzA,
			descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
			d_Qb, tol, reorder, d_z, &singularity));

	}
	else if (0 == strcmp(opts.testFunc, "lu"))
	{
		printf("WARNING: no LU available on GPU \n");
	}
	else if (0 == strcmp(opts.testFunc, "qr"))
	{
		(cusolverSpDcsrlsvqr(
			handle, rowsA, nnzA,
			descrA, d_csrValB, d_csrRowPtrB, d_csrColIndB,
			d_Qb, tol, reorder, d_z, &singularity));
	}
	else
	{
		fprintf(stderr, "Error: %s is unknow function\n", opts.testFunc);
		return ;
	}
	(cudaDeviceSynchronize());
	if (0 <= singularity)
	{
		printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
	}
	/* Q*x = z */
	(cusparseDsctr(cusparseHandle,
		rowsA,
		d_z,
		d_Q,
		d_x,
		CUSPARSE_INDEX_BASE_ZERO));
	(cudaDeviceSynchronize());

	stop = second();
	time_solve_gpu = stop - start;

	printf("step 8: evaluate residual r = b - A*x (result on GPU)\n");
	(cudaMemcpyAsync(d_r, d_b, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice, stream));
	(cusparseDcsrmv(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		rowsA,
		colsA,
		nnzA,
		&minus_one,
		descrA,
		d_csrValA,
		d_csrRowPtrA,
		d_csrColIndA,
		d_x,
		&one,
		d_r));
	(cudaMemcpyAsync(h_x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost, stream));
	(cudaMemcpyAsync(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost, stream));
	/* wait until h_x and h_r are ready */
	(cudaDeviceSynchronize());

	b_inf = vec_norminf(rowsA, h_b);
	x_inf = vec_norminf(colsA, h_x);
	r_inf = vec_norminf(rowsA, h_r);

	if (0 != strcmp(opts.testFunc, "lu"))
	{
		// only cholesky and qr have GPU version
		printf("(GPU) |b - A*x| = %E \n", r_inf);
		printf("(GPU) |A| = %E \n", A_inf);
		printf("(GPU) |x| = %E \n", x_inf);
		printf("(GPU) |b| = %E \n", b_inf);
		printf("(GPU) |b - A*x|/(|A|*|x| + |b|) = %E \n", r_inf / (A_inf * x_inf + b_inf));
	}

	fprintf(stdout, "timing %s: CPU = %10.6f sec , GPU = %10.6f sec\n", opts.testFunc, time_solve_cpu, time_solve_gpu);

	if (0 != strcmp(opts.testFunc, "lu")) {
		printf("show last 10 elements of solution vector (GPU) \n");
		printf("consistent result for different reordering and solver \n");
		for (int j = rowsA - 10; j < rowsA; j++) {
			printf("x[%d] = %E\n", j, h_x[j]);
		}
	}

	if (handle) { (cusolverSpDestroy(handle)); }
	if (cusparseHandle) { (cusparseDestroy(cusparseHandle)); }
	if (stream) { (cudaStreamDestroy(stream)); }
	if (descrA) { (cusparseDestroyMatDescr(descrA)); }

	if (h_csrValA) { free(h_csrValA); }
	if (h_csrRowPtrA) { free(h_csrRowPtrA); }
	if (h_csrColIndA) { free(h_csrColIndA); }
	if (h_z) { free(h_z); }
	if (h_x) { free(h_x); }
	if (h_b) { free(h_b); }
	if (h_Qb) { free(h_Qb); }
	if (h_r) { free(h_r); }

	if (h_Q) { free(h_Q); }

	if (h_csrRowPtrB) { free(h_csrRowPtrB); }
	if (h_csrColIndB) { free(h_csrColIndB); }
	if (h_csrValB) { free(h_csrValB); }
	if (h_mapBfromA) { free(h_mapBfromA); }

	if (buffer_cpu) { free(buffer_cpu); }

	if (d_csrValA) { (cudaFree(d_csrValA)); }
	if (d_csrRowPtrA) { (cudaFree(d_csrRowPtrA)); }
	if (d_csrColIndA) { (cudaFree(d_csrColIndA)); }
	if (d_csrValB) { (cudaFree(d_csrValB)); }
	if (d_csrRowPtrB) { (cudaFree(d_csrRowPtrB)); }
	if (d_csrColIndB) { (cudaFree(d_csrColIndB)); }
	if (d_Q) { (cudaFree(d_Q)); }
	if (d_z) { (cudaFree(d_z)); }
	if (d_x) { (cudaFree(d_x)); }
	if (d_b) { (cudaFree(d_b)); }
	if (d_Qb) { (cudaFree(d_Qb)); }
	if (d_r) { (cudaFree(d_r)); }
}

int main(int argc, char *argv[]) {
	int status = EXIT_FAILURE;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	ReadResVector(res, "res.crs");
	ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());

	int symmetrize = 0;
	int debug = 0;
	int maxit = 2000; //5; //2000; //1000;  //50; //5; //50; //100; //500; //10000;
	double tol = 0.0001; //0.000001; //0.00001; //0.00000001; //0.0001; //0.001; //0.00000001; //0.1; //0.001; //0.00000001;
	double damping = 0.75;

	/* WARNING: it is assumed that the matrices are stores in Matrix Market format */
	printf("-----------BICGSTAB-------------");
  	status = test_bicgstab(matr, res, x, symmetrize, debug, damping, maxit, tol,
   		DBICGSTAB_MAX_ULP_ERR, DBICGSTAB_EPS);
	printf("------------CG------------------");
	testCG(matr, res, x, tol, maxit);
//	testSolverSP(matr, res, x, tol, maxit);
	return status;
}

