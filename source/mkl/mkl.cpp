// mkl.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
/*******************************************************************************
* Copyright 2005-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!
!  Content: Intel(R) MKL RCI CG (Conjugate Gradient method) C example without
!           both preconditioner and user-defined stopping criteria
!
!*******************************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include "mkl.h"
#include "mkl_rci.h"
#include "mkl_blas.h"
#include "mkl_spblas.h"
#include "mkl_service.h"
#include "mkl_dss.h"

struct MatrixCRS
{
	double* data;
	unsigned int* colInd;
	unsigned int* rowPtr;
	unsigned int  dataSize;
	unsigned int  rowPtrSize;
};

//////////////////////////////////////////////////////////////////////////
static void ReadCSRMatrixFromBinary(MatrixCRS& matr, const std::string& in_file_name)
{
	std::ifstream file_out_data, file_out_cols, file_out_rows;
	file_out_data.open(std::string(in_file_name + "data").c_str(), std::ifstream::binary);
	file_out_cols.open(std::string(in_file_name + "cols").c_str(), std::ifstream::binary);
	file_out_rows.open(std::string(in_file_name + "rows").c_str(), std::ifstream::binary);

	file_out_data.seekg(0, file_out_data.end);
	long dataSize = file_out_data.tellg() / sizeof(double);
	file_out_data.seekg(0);
	matr.data = new double[dataSize];
	matr.dataSize = dataSize;
	file_out_data.read((char*)matr.data, dataSize * sizeof(double));

	file_out_rows.seekg(0, file_out_rows.end);
	long rowSize = file_out_rows.tellg() / sizeof(unsigned int);
	file_out_rows.seekg(0);
	matr.rowPtr = new unsigned int[rowSize];
	matr.rowPtrSize = rowSize - 1;
	file_out_rows.read((char*)matr.rowPtr, rowSize * sizeof(unsigned int));

	file_out_cols.seekg(0, file_out_cols.end);
	long colSize = file_out_cols.tellg() / sizeof(unsigned int);
	file_out_cols.seekg(0);
	matr.colInd = new unsigned int[colSize];
	file_out_cols.read((char*)matr.colInd, colSize * sizeof(unsigned int));
	file_out_data.close();
	file_out_rows.close();
	file_out_cols.close();
}

//////////////////////////////////////////////////////////////////////////
static void ReadResVector(std::vector<double>& vec, const std::string& file_name)
{
	std::ifstream file;
	file.open(file_name.c_str(), std::ifstream::binary);
	file.seekg(0, file.end);
	long dataSize = file.tellg() / sizeof(double);
	file.seekg(0);
	vec.resize(dataSize);
	file.read((char*)&vec[0], dataSize * sizeof(double));
}

//////////////////////////////////////////////////////////////////////////
static void CG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT rci_request, itercount, expected_itercount = 8, i;
	double* rhs = (double*)malloc(n*sizeof(double));
	/* Fill all arrays containing matrix data. */
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;
	
	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	MKL_INT length = 128;
	double* expected_sol = (double*)malloc(n*sizeof(double));
	
	MKL_INT ipar[128];
	double euclidean_norm;
	double dpar[128];
	double* tmp = (double*)malloc(4 * n * sizeof(double));
	double eone = -1.E0;
	MKL_INT ione = 1;
	double* A = matrix.data;
	/* Some additional variables to use with the RCI (P)CG solver                */
	/*---------------------------------------------------------------------------*/
	double* solution = &x[0];
	struct matrix_descr descrA;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, A);
	double all_time = 0.0;
	
	/*---------------------------------------------------------------------------*/
	/* Initialize the solver                                                     */
	/*---------------------------------------------------------------------------*/
	dcg_init(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
	assert(rci_request == 0);
	i = 0;
	double init_sqrt_res_norm = dnrm2(&n, &res[0], &i);
	const int numIterates = 1;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		memcpy(rhs, &res[0], n * sizeof(double));
		memset(expected_sol, 0, n * sizeof(double));
		/*---------------------------------------------------------------------------*/
		
		/* Initialize the initial guess                                              */
		/*---------------------------------------------------------------------------*/
		for (i = 0; i < n; i++)
		{
			expected_sol[i] = expected_x[i];
			solution[i] = 0.0;
		}
		for (i = 0; i < 4*n; i++)
		{
			tmp[i] = 0.0;
		}
		
		/*---------------------------------------------------------------------------*/
		/* Set the desired parameters:                                               */
		/* LOGICAL parameters:                                                       */
		/* do residual stopping test                                                 */
		/* do not request for the user defined stopping test                         */
		/* DOUBLE parameters                                                         */
		/* set the relative tolerance to 1.0D-5 instead of default value 1.0D-6      */
		/*---------------------------------------------------------------------------*/
		ipar[0] = n;
		ipar[2] = 1;
		ipar[3] = 0; //current it
		ipar[4] = max_it;
		ipar[5] = 1;
		ipar[6] = 1;
		ipar[7] = 1; //STOp test if (it >= max_it)

		ipar[8] = 1; //RES(i-1) <= RES(i) - stop factor
		ipar[9] = 0; //user stop test/ In CUSP: ||res|| <= abs_tol + rel_tol*||b||;
		ipar[10] = 0;//no preconditioner
		
		dpar[0] = tolerance; //relative tolerance
		dpar[1] = 0.0;//absolute tolerance
		dpar[2] = init_sqrt_res_norm;//
		dpar[3] = 0.0;//init = abs_tol + rel_tol*||b||
		dpar[4] = 0.0;//current ||res||
		dpar[5] = 0.0;//prev ||res||
		dpar[6] = 0.0;//alpha for CG
		dpar[7] = 0.0;//beta for CG


		/*---------------------------------------------------------------------------*/
		/* Check the correctness and consistency of the newly set parameters         */
		/*---------------------------------------------------------------------------*/
		dcg_check(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
		assert(rci_request == 0);
		/*---------------------------------------------------------------------------*/
		/* Compute the solution by RCI (P)CG solver without preconditioning          */
		/* Reverse Communications starts here                                        */
		/*---------------------------------------------------------------------------*/
		do
		{
			dcg(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
			/*---------------------------------------------------------------------------*/
			/* If rci_request=0, then the solution was found with the required precision */
			/*---------------------------------------------------------------------------*/
			if (rci_request == 1)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, tmp, 0.0, &tmp[n]);
			}
			if (rci_request == 2) //control test
			{
				if (sqrt(dpar[4]) <= dpar[1] + dpar[0] * init_sqrt_res_norm)
				{
					break;
				}
			}
			if (rci_request == -1)
			{
				printf("Solver reached max_iterates %i, with current norm = %2.6f\n", max_it, dpar[4]);
				break;
			}
		} while (rci_request != 0);
		assert(rci_request == 0 || rci_request == 2);
		dcg_get(&n, solution, rhs, &rci_request, ipar, dpar, tmp, &itercount);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	/*---------------------------------------------------------------------------*/
	/* Print solution vector: solution[n] and number of iterations: itercount    */
	/*---------------------------------------------------------------------------*/
	printf("----------------CG--------------------\n");
	printf("The system has been solved\n");
	printf("\nExpected solution have error\n");
	double expect_rate = 0.0;
	for (i = 0; i < n; i++)
	{
		expect_rate += 100.0*abs(expected_sol[i] - solution[i]) / abs(expected_sol[i]);
	}
	expect_rate /= (double)n;
	printf(" %6.6f \%", expect_rate);
	printf("\n");

	printf("\nNumber of iterations: %d\n", itercount);
	//i = 0;
	//euclidean_norm = ;
	double final_tol = 0.0;
	i = 0;
	final_tol = dnrm2(&n, &res[0], &i)*dpar[0];
	printf("Norm of residual = %E, final tolerance =%E \n", dpar[4], final_tol);
	printf("Average time = %f\n", all_time);
	printf("----------------------------------------\n");
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(rhs);
	free(expected_sol);
	free(tmp);
}



//////////////////////////////////////////////////////////////////////////
static void CG_UserStop(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT rci_request, itercount, expected_itercount = 8, i;
	double* rhs = (double*)malloc(n * sizeof(double));
	/* Fill all arrays containing matrix data. */
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	MKL_INT length = 128;
	double* expected_sol = (double*)malloc(n * sizeof(double));

	MKL_INT ipar[128];
	double euclidean_norm;
	double dpar[128];
	double* tmp = (double*)malloc(4 * n * sizeof(double));
	double* temp = (double*)malloc(n * sizeof(double));
	double eone = -1.E0;
	MKL_INT ione = 1;
	double* A = matrix.data;
	/* Some additional variables to use with the RCI (P)CG solver                */
	/*---------------------------------------------------------------------------*/
	double* solution = &x[0];
	struct matrix_descr descrA;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, A);
	double all_time = 0.0;

	/*---------------------------------------------------------------------------*/
	/* Initialize the solver                                                     */
	/*---------------------------------------------------------------------------*/
	dcg_init(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
	assert(rci_request == 0);
	i = 0;
	double init_sqrt_res_norm = dnrm2(&n, &res[0], &i);
	const int numIterates = 1;
	double final_tol = init_sqrt_res_norm*dpar[0] + dpar[1];
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		memcpy(rhs, &res[0], n * sizeof(double));
		memset(expected_sol, 0, n * sizeof(double));
		/*---------------------------------------------------------------------------*/

		/* Initialize the initial guess                                              */
		/*---------------------------------------------------------------------------*/
		for (i = 0; i < n; i++)
		{
			expected_sol[i] = expected_x[i];
			solution[i] = 0.0;
		}
		for (i = 0; i < 4 * n; i++)
		{
			tmp[i] = 0.0;
		}

		/*---------------------------------------------------------------------------*/
		/* Set the desired parameters:                                               */
		/* LOGICAL parameters:                                                       */
		/* do residual stopping test                                                 */
		/* do not request for the user defined stopping test                         */
		/* DOUBLE parameters                                                         */
		/* set the relative tolerance to 1.0D-5 instead of default value 1.0D-6      */
		/*---------------------------------------------------------------------------*/
		ipar[0] = n;
		ipar[2] = 1;
		ipar[3] = 0; //current it
		ipar[4] = max_it;
		ipar[5] = 1;
		ipar[6] = 1;
		ipar[7] = 1; //STOp test if (it >= max_it)

		ipar[8] = 0; //RES(i-1) <= RES(i) - stop factor
		ipar[9] = 1; //user stop test/ In CUSP: ||res|| <= abs_tol + rel_tol*||b||;
		ipar[10] = 0;//no preconditioner

		dpar[0] = tolerance; //relative tolerance
		dpar[1] = 0.0;//absolute tolerance
		dpar[2] = init_sqrt_res_norm;//
		dpar[3] = 0.0;//init = abs_tol + rel_tol*||b||
		dpar[4] = 0.0;//current ||res||
		dpar[5] = 0.0;//prev ||res||
		dpar[6] = 0.0;//alpha for CG
		dpar[7] = 0.0;//beta for CG


					  /*---------------------------------------------------------------------------*/
					  /* Check the correctness and consistency of the newly set parameters         */
					  /*---------------------------------------------------------------------------*/
		dcg_check(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
		assert(rci_request == 0);
		/*---------------------------------------------------------------------------*/
		/* Compute the solution by RCI (P)CG solver without preconditioning          */
		/* Reverse Communications starts here                                        */
		/*---------------------------------------------------------------------------*/
 		mkl_sparse_d_mv(transA, 1.0, csrA, descrA, solution, 0.0, temp);
 		daxpy(&n, &eone, rhs, &ione, temp, &ione);
 		euclidean_norm = dnrm2(&n, temp, &ione);
		do
		{
			dcg(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
			/*---------------------------------------------------------------------------*/
			/* If rci_request=0, then the solution was found with the required precision */
			/*---------------------------------------------------------------------------*/
			if (rci_request == 1)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, tmp, 0.0, &tmp[n]);
			}
			if (rci_request == 2) //control test
			{
  				//mkl_sparse_d_mv(transA, 1.0, csrA, descrA, solution, 0.0, temp);
  				//daxpy(&n, &eone, rhs, &ione, temp, &ione);
//  				euclidean_norm = dnrm2(&n, temp, &ione);
// 				if (euclidean_norm < final_tol) break;
				if (sqrt(dpar[4]) < final_tol) break;
				//if ((dpar[4]/ euclidean_norm) < (tolerance*tolerance)) break;
			}
			if (rci_request == -1)
			{
				printf("Solver reached max_iterates %i, with current norm = %2.6f\n", max_it, dpar[4]);
				break;
			}
		} while (rci_request != 0);
		assert(rci_request == 0 || rci_request == 2);
		dcg_get(&n, solution, rhs, &rci_request, ipar, dpar, tmp, &itercount);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	/*---------------------------------------------------------------------------*/
	/* Print solution vector: solution[n] and number of iterations: itercount    */
	/*---------------------------------------------------------------------------*/
	printf("----------------CG UserStop-----------------\n");
	printf("The system has been solved\n");
	printf("\nExpected solution have error\n");
	double expect_rate = 0.0;
	for (i = 0; i < n; i++)
	{
		expect_rate += 100.0*abs(expected_sol[i] - solution[i]) / abs(expected_sol[i]);
	}
	expect_rate /= (double)n;
	printf(" %6.6f \%", expect_rate);
	printf("\n");

	printf("\nNumber of iterations: %d\n", itercount);
	//i = 0;
	//euclidean_norm = ;
	
	printf("Norm of residual = %E, final tolerance =%E \n", dpar[4], final_tol);
	printf("Average time = %f\n", all_time);
	printf("----------------------------------------\n");
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(rhs);
	free(expected_sol);
	free(tmp);
	free(temp);
}

//////////////////////////////////////////////////////////////////////////
static void CG_LikeCUSP(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	double* y = (double*)malloc(n * sizeof(double));
	//double* z = (double*)malloc(n * sizeof(double));
	double* r = (double*)malloc(n * sizeof(double));
	double* p = (double*)malloc(n * sizeof(double));
	double* solution = &x[0];
	for (int i = 0; i < x.size(); i++)
	{
		solution[i] = 0.0;
	}
	struct matrix_descr descrA;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, matrix.data);
	double all_time = 0.0;
	const int numIterates = 1;
	const double one = 1.0;
	const double zero = 0.0;
	const double eone = -1.0;
	const int ione = 1;
	const int izero = 0;
	int iterates = 0;
	double norm_residual = 0.0;
	double final_tolerance = 0.0;
	bool converged = false;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		iterates = 0;
		//r <= b
		memcpy(r, &res[0], res.size() * sizeof(double));
		//y <= A*x
		mkl_sparse_d_mv(transA, 1.0, csrA, descrA, solution, 0.0, y);
		//r <= r - y
		daxpy(&n, &eone, y, &ione, r, &ione);
		// p <= r
		dcopy(&n, r, &ione, p, &ione);
		//rz = <r, r>
		double rz = ddot(&n, r, &ione, r, &ione);
		double b_norm = dnrm2(&n, &res[0], &ione);
		final_tolerance = b_norm*tolerance;
		norm_residual = dnrm2(&n, r, &ione);
		bool cond = norm_residual <= final_tolerance;
		while (!cond && iterates < max_it)
		{
			//y <= A*p
			mkl_sparse_d_mv(transA, 1.0, csrA, descrA, p, 0.0, y);
			//alpha <= <r,z>/<y,p>
			double alpha = rz / ddot(&n, y, &ione, p, &ione);
			//x <= x +alpha*p 
			daxpy(&n, &alpha, p, &ione, solution, &ione);
			//r <= r - alpha*y
			double ialpha = -alpha;
			daxpy(&n, &ialpha, y, &ione, r, &ione);

			//z <= r
			//dcopy(&n, r, &ione, z, &ione);
			double rz_old = rz;
			rz = ddot(&n, r, &ione, r, &ione);

			//beta <= rz / rz_old
			double beta = rz / rz_old;

			//p <= r + beta*p
			daxpby(&n, &one, r, &ione, &beta, p, &ione);

			final_tolerance = b_norm*tolerance;
			norm_residual = dnrm2(&n, r, &ione);
			cond = norm_residual <= final_tolerance;
			iterates++;
		}
		converged = cond && (iterates <= max_it);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	printf("----------------CG_LikeCUSP----------------\n");
	if (converged)
	{
		printf("The system has been solved\n");
		printf("\nNumber of iterations: %d\n", iterates);
		printf("Norm of residual = %E, final tolerance =%E \n", norm_residual, final_tolerance);
		printf("Average time = %f\n", all_time);
		printf("----------------------------------------\n");
	}


	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(r);
	free(y);
	free(p);
	//free(z);
}

//////////////////////////////////////////////////////////////////////////
static void GmRes(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT rci_request, itercount, expected_itercount = 8, i;
	double* rhs = (double*)malloc(n * sizeof(double));
	/* Fill all arrays containing matrix data. */
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	MKL_INT length = 128;
	double* expected_sol = (double*)malloc(n * sizeof(double));

	MKL_INT ipar[128];
	double euclidean_norm;
	double dpar[128];
	const int non_restarted_it = 150;
	const int tmp_size = n * (2 * non_restarted_it+ 1) + (non_restarted_it * (non_restarted_it + 9)) / 2 + 1;
	double* tmp = (double*)malloc(tmp_size * sizeof(double));
	double one = 1.0;
	double eone = -1.E0;
	double zero = 0.0;
	MKL_INT ione = 1;
	MKL_INT izero = 0;
	double* A = matrix.data;
	/* Some additional variables to use with the RCI (P)CG solver                */
	/*---------------------------------------------------------------------------*/
	double* solution = &x[0];
	struct matrix_descr descrA;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, A);
	double all_time = 0.0;

	/*---------------------------------------------------------------------------*/
	/* Initialize the solver                                                     */
	/*---------------------------------------------------------------------------*/
	dfgmres_init(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
	assert(rci_request == 0);
	i = 0;
	double init_sqrt_res_norm = dnrm2(&n, &res[0], &i);
	const int numIterates = 1;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		memcpy(rhs, &res[0], n * sizeof(double));
		memset(expected_sol, 0, n * sizeof(double));
		/*---------------------------------------------------------------------------*/

		/* Initialize the initial guess                                              */
		/*---------------------------------------------------------------------------*/
		for (i = 0; i < n; i++)
		{
			expected_sol[i] = expected_x[i];
			solution[i] = 0.0;
		}
		for (i = 0; i < tmp_size; i++)
		{
			tmp[i] = 0.0;
		}

		/*---------------------------------------------------------------------------*/
		/* Set the desired parameters:                                               */
		/* LOGICAL parameters:                                                       */
		/* do residual stopping test                                                 */
		/* do not request for the user defined stopping test                         */
		/* DOUBLE parameters                                                         */
		/* set the relative tolerance to 1.0D-5 instead of default value 1.0D-6      */
		/*---------------------------------------------------------------------------*/
		ipar[0] = n;
		ipar[2] = 1;
		ipar[3] = 0; //current it
		ipar[4] = max_it;
		ipar[5] = 1;
		ipar[6] = 1;
		ipar[7] = 1; //STOp test if (it >= max_it)

		ipar[8] = 1; //RES(i-1) <= RES(i) - stop factor
		ipar[9] = 0; //user stop test/ In CUSP: ||res|| <= abs_tol + rel_tol*||b||;
		ipar[10] = 0;//no preconditioner
		ipar[11] = 1;
		dpar[0] = tolerance; //relative tolerance
		dpar[1] = 0.0;//absolute tolerance
		dpar[2] = init_sqrt_res_norm;//
		dpar[3] = 0.0;//init = abs_tol + rel_tol*||b||
		dpar[4] = 0.0;//current ||res||
		dpar[5] = 0.0;//prev ||res||
		dpar[6] = 0.0;
		dpar[7] = 1.0e-12;


		/*---------------------------------------------------------------------------*/
		/* Check the correctness and consistency of the newly set parameters         */
		/*---------------------------------------------------------------------------*/
		dfgmres_check(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
		assert(rci_request == 0);
		/*---------------------------------------------------------------------------*/
		/* Compute the solution by RCI (P)CG solver without preconditioning          */
		/* Reverse Communications starts here                                        */
		/*---------------------------------------------------------------------------*/
		do
		{
			dfgmres(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
			/*---------------------------------------------------------------------------*/
			/* If rci_request=0, then the solution was found with the required precision */
			/*---------------------------------------------------------------------------*/
			if (rci_request == 1)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, &tmp[ipar[21] - 1], 0.0, &tmp[ipar[22] - 1]);
			}
			if (rci_request == -1)
			{
				printf("Solver reached max_iterates %i, with current norm = %2.6f\n", max_it, dpar[4]);
				return;
			}
		} while (rci_request != 0);
		assert(rci_request == 0 || rci_request == 2);
		dfgmres_get(&n, solution, rhs, &rci_request, ipar, dpar, tmp, &itercount);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	/*---------------------------------------------------------------------------*/
	/* Print solution vector: solution[n] and number of iterations: itercount    */
	/*---------------------------------------------------------------------------*/
	printf("----------------GmRes--------------------\n");
	printf("The system has been solved\n");
	printf("\nNumber of iterations: %d\n", itercount);
	//i = 0;
	//euclidean_norm = ;
	double final_tol = 0.0;
	i = 0;
	final_tol = dnrm2(&n, &res[0], &i)*dpar[0];
	printf("Norm of residual = %E, final tolerance =%E \n", sqrt(dpar[4]), final_tol);
	printf("Average time = %f\n", all_time);
	printf("----------------------------------------\n");
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(rhs);
	free(expected_sol);
	free(tmp);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static void CGPrec(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT rci_request, itercount, expected_itercount = 8, i;
	double* rhs = (double*)malloc(n * sizeof(double));
	/* Fill all arrays containing matrix data. */
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	MKL_INT length = 128;
	double* expected_sol = (double*)malloc(n * sizeof(double));

	MKL_INT ipar[128];
	double euclidean_norm;
	double dpar[128];
	double* tmp = (double*)malloc(4 * n * sizeof(double));
	double* ilu0 = (double*)malloc(matrix.dataSize * sizeof(double));
	double* trvec = (double*)malloc(n * sizeof(double));
	double eone = -1.E0;
	MKL_INT ione = 1;
	double* A = matrix.data;
	/* Some additional variables to use with the RCI (P)CG solver                */
	/*---------------------------------------------------------------------------*/
	double* solution = &x[0];
	struct matrix_descr descrA, descrL;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA, csrL;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, A);
	double all_time = 0.0;

	/*---------------------------------------------------------------------------*/
	/* Initialize the solver                                                     */
	/*---------------------------------------------------------------------------*/
	dcg_init(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
	assert(rci_request == 0);
	i = 0;
	double init_sqrt_res_norm = dnrm2(&n, &res[0], &i);
	const int numIterates = 1;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		memcpy(rhs, &res[0], n * sizeof(double));
		memset(expected_sol, 0, n * sizeof(double));
		memset(trvec, 0, n * sizeof(double));
		/*---------------------------------------------------------------------------*/

		/* Initialize the initial guess                                              */
		/*---------------------------------------------------------------------------*/
		for (i = 0; i < n; i++)
		{
			expected_sol[i] = expected_x[i];
			solution[i] = 0.0;
		}
		for (i = 0; i < 4 * n; i++)
		{
			tmp[i] = 0.0;
		}

		/*---------------------------------------------------------------------------*/
		/* Set the desired parameters:                                               */
		/* LOGICAL parameters:                                                       */
		/* do residual stopping test                                                 */
		/* do not request for the user defined stopping test                         */
		/* DOUBLE parameters                                                         */
		/* set the relative tolerance to 1.0D-5 instead of default value 1.0D-6      */
		/*---------------------------------------------------------------------------*/
		ipar[0] = n;
		ipar[1] = 6;//error on screen
		ipar[2] = 1;
		ipar[3] = 0; //current it
		ipar[4] = max_it;
		ipar[5] = 1;
		ipar[6] = 1;
		ipar[7] = 1; //STOp test if (it >= max_it)

		ipar[8] = 0; //RES(i-1) <= RES(i) - stop factor
		ipar[9] = 1; //user stop test/ In CUSP: ||res|| <= abs_tol + rel_tol*||b||;
		ipar[10] = 0;//preconditioner

		dpar[0] = tolerance; //relative tolerance
		dpar[1] = 0.0;//absolute tolerance
		dpar[2] = init_sqrt_res_norm;//
		dpar[3] = 0.0;//init = abs_tol + rel_tol*||b||
		dpar[4] = 0.0;//current ||res||
		dpar[5] = 0.0;//prev ||res||
		dpar[6] = 0.0;//alpha for CG
		dpar[7] = 0.0;//beta for CG

		int error;
		dcsrilu0(&n, A, rows, cols, ilu0, ipar, dpar, &error);
		assert(error == 0);
		double nrm2 = dnrm2(&n, ilu0, &ione);
		mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, ilu0);
		/*---------------------------------------------------------------------------*/
		/* Check the correctness and consistency of the newly set parameters         */
		/*---------------------------------------------------------------------------*/
		dcg_check(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
		assert(rci_request == 0);
		/*---------------------------------------------------------------------------*/
		/* Compute the solution by RCI (P)CG solver without preconditioning          */
		/* Reverse Communications starts here                                        */
		/*---------------------------------------------------------------------------*/

		do
		{
			dcg(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
			/*---------------------------------------------------------------------------*/
			/* If rci_request=0, then the solution was found with the required precision */
			/*---------------------------------------------------------------------------*/
			if (rci_request == 1)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, tmp, 0.0, &tmp[n]);
			}
			else if (rci_request == 2) //control test
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, solution, 0.0, trvec);
				daxpy(&n, &eone, rhs, &ione, trvec, &ione);
				euclidean_norm = dnrm2(&n, trvec, &ione);
				if (euclidean_norm < dpar[1] + dpar[0] * init_sqrt_res_norm) break;
			}
			else if (rci_request == 3)
			{
				descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
				descrL.mode = SPARSE_FILL_MODE_LOWER;
				descrL.diag = SPARSE_DIAG_UNIT;
				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, &tmp[2*n], trvec);

				descrL.mode = SPARSE_FILL_MODE_UPPER;
				descrL.diag = SPARSE_DIAG_NON_UNIT;
				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, trvec, &tmp[3*n]);
			}
			else if (rci_request == -1)
			{
				printf("Solver reached max_iterates %i, with current norm = %2.6f\n", max_it, dpar[4]);
				return;
			}
			else if (rci_request == 0)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, solution, 0.0, trvec);
				daxpy(&n, &eone, rhs, &ione, trvec, &ione);
				euclidean_norm = dnrm2(&n, trvec, &ione);
				break;
			}
			else
			{
				assert(false);
			}
		} while (rci_request != 0);
		assert(rci_request == 0 || rci_request == 2);
		dcg_get(&n, solution, rhs, &rci_request, ipar, dpar, tmp, &itercount);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	/*---------------------------------------------------------------------------*/
	/* Print solution vector: solution[n] and number of iterations: itercount    */
	/*---------------------------------------------------------------------------*/
	printf("----------------CG Prec MKL--------------------\n");
	printf("The system has been solved\n");
	printf("\nNumber of iterations: %d\n", itercount);
	//i = 0;
	//euclidean_norm = ;
	double final_tol = 0.0;
	i = 0;
	final_tol = dnrm2(&n, &res[0], &i)*dpar[0];
	printf("Norm of residual = %E, final tolerance =%E \n", euclidean_norm, final_tol);
	printf("Average time = %f\n", all_time);
	printf("----------------------------------------\n");
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(rhs);
	free(expected_sol);
	free(tmp);
	free(ilu0);
	free(trvec);
}

//////////////////////////////////////////////////////////////////////////
static void GmResPrec(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT rci_request, itercount, expected_itercount = 8, i;
	double* rhs = (double*)malloc(n * sizeof(double));
	/* Fill all arrays containing matrix data. */
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	MKL_INT length = 128;
	double* expected_sol = (double*)malloc(n * sizeof(double));

	MKL_INT ipar[128];
	double euclidean_norm;
	double dpar[128];
	const int non_restarted_it = 150;
	const int tmp_size = n * (2 * non_restarted_it + 1) + (non_restarted_it * (non_restarted_it + 9)) / 2 + 1;
	double* tmp = (double*)malloc(tmp_size * sizeof(double));
	double* ilu0 = (double*)malloc(matrix.dataSize * sizeof(double));
	double* residual = (double*)malloc(n * sizeof(double));
	double* b = (double*)malloc(n * sizeof(double));
	double* trvec = (double*)malloc(n * sizeof(double));

	double one = 1.0;
	double eone = -1.E0;
	double zero = 0.0;
	MKL_INT ione = 1;
	MKL_INT izero = 0;
	double* A = matrix.data;
	/* Some additional variables to use with the RCI (P)CG solver                */
	/*---------------------------------------------------------------------------*/
	double* solution = &x[0];
	struct matrix_descr descrA, descrL;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA, csrL;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, A);
	
	double all_time = 0.0;
	/*---------------------------------------------------------------------------*/
	/* Initialize the solver                                                     */
	/*---------------------------------------------------------------------------*/
	dfgmres_init(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
	assert(rci_request == 0);
	i = 0;
	double init_sqrt_res_norm = dnrm2(&n, &res[0], &i);
	const int numIterates = 1;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		memcpy(rhs, &res[0], n * sizeof(double));
		memset(expected_sol, 0, n * sizeof(double));
		memset(trvec, 0, n * sizeof(double));
		/*---------------------------------------------------------------------------*/

		/* Initialize the initial guess                                              */
		/*---------------------------------------------------------------------------*/
		for (i = 0; i < n; i++)
		{
			expected_sol[i] = expected_x[i];
			solution[i] = 0.0;
		}
		for (i = 0; i < tmp_size; i++)
		{
			tmp[i] = 0.0;
		}

		/*---------------------------------------------------------------------------*/
		/* Set the desired parameters:                                               */
		/* LOGICAL parameters:                                                       */
		/* do residual stopping test                                                 */
		/* do not request for the user defined stopping test                         */
		/* DOUBLE parameters                                                         */
		/* set the relative tolerance to 1.0D-5 instead of default value 1.0D-6      */
		/*---------------------------------------------------------------------------*/
 		ipar[0] = n;
 		ipar[2] = 1;
 		ipar[3] = 0; //current it
 		ipar[4] = max_it;
 		ipar[5] = 1;
 		ipar[6] = 1;
 		ipar[7] = 1; //STOp test if (it >= max_it)
// 
 		ipar[8] = 1; //RES(i-1) <= RES(i) - stop factor
 		ipar[9] = 0; //user stop test/ In CUSP: ||res|| <= abs_tol + rel_tol*||b||;
 		ipar[10] = 1;//preconditioner
 		ipar[11] = 1;
		ipar[30] = 1;
		

 		dpar[0] = tolerance; //relative tolerance
 		dpar[1] = 0.0;//absolute tolerance
 		dpar[2] = init_sqrt_res_norm;//
 		dpar[3] = 0.0;//init = abs_tol + rel_tol*||b||
 		dpar[4] = 0.0;//current ||res||
 		dpar[5] = 0.0;//prev ||res||
 		dpar[6] = 0.0;
 		dpar[7] = 1.0e-12;
		
 		dpar[30] = 1.0e-20;
 		dpar[31] = 1.0e-16;
 		dcopy(&n, rhs, &ione, b, &ione);
 		int error;
 		dcsrilu0(&n, A, rows, cols, ilu0, ipar, dpar, &error);
 		assert(error == 0);
 		double nrm2 = dnrm2(&n, ilu0, &ione);
 		mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, ilu0);
		
		/*---------------------------------------------------------------------------*/
		/* Check the correctness and consistency of the newly set parameters         */
		/*---------------------------------------------------------------------------*/
		dfgmres_check(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
		assert(rci_request == 0);
		do
		{
			dfgmres(&n, solution, rhs, &rci_request, ipar, dpar, tmp);
			/*---------------------------------------------------------------------------*/
			/* If rci_request=0, then the solution was found with the required precision */
			/*---------------------------------------------------------------------------*/
			if (rci_request == 0)
			{
				break;
			}
			if (rci_request == 1)
			{
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, &tmp[ipar[21] - 1], 0.0, &tmp[ipar[22] - 1]);
			}
			else if (rci_request == 2)
			{
				/* Request to the dfgmres_get routine to put the solution into b[N] via ipar[12]
				--------------------------------------------------------------------------------
				WARNING: beware that the call to dfgmres_get routine with ipar[12]=0 at this
				stage may destroy the convergence of the FGMRES method, therefore, only
				advanced users should exploit this option with care */
				ipar[12] = 1;
				/* Get the current FGMRES solution in the vector b[N] */
				int inner_rci_request;
				dfgmres_get(&n, solution, b, &inner_rci_request, ipar, dpar, tmp, &itercount);
				/* Compute the current true residual via Intel(R) MKL (Sparse) BLAS routines */
				mkl_sparse_d_mv(transA, 1.0, csrA, descrA, b, 0.0, residual);
				double dvar = -1.0E0;
				daxpy(&n, &dvar, rhs, &ione, residual, &ione);
				dvar = dnrm2(&n, residual, &ione);
				double rhs_norm = dnrm2(&n, rhs, &ione);
				if (dvar < tolerance*rhs_norm) break;
			}
			else if (rci_request == 3)
			{
 				descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
 				descrL.mode = SPARSE_FILL_MODE_LOWER;
 				descrL.diag = SPARSE_DIAG_UNIT;
 				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, &tmp[ipar[21] - 1], trvec);
 
 				descrL.mode = SPARSE_FILL_MODE_UPPER;
 				descrL.diag = SPARSE_DIAG_NON_UNIT;
 				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, trvec, &tmp[ipar[22] - 1]);
			}
			else if (rci_request == -1)
			{
				printf("Solver reached max_iterates %i, with current norm = %2.6f\n", max_it, dpar[4]);
				return;
			}
			else if (rci_request == 4)
			{
				if (dpar[6] < 1.0E-12)
					break;
			}
			else
			{
				assert(false);
			}
		} while (true);
		assert(rci_request == 0 || rci_request == 2);
		dfgmres_get(&n, solution, rhs, &rci_request, ipar, dpar, tmp, &itercount);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	/*---------------------------------------------------------------------------*/
	/* Print solution vector: solution[n] and number of iterations: itercount    */
	/*---------------------------------------------------------------------------*/
	printf("----------------GmRes Prec--------------------\n");
	printf("The system has been solved\n");
	printf("\nNumber of iterations: %d\n", itercount);
	//i = 0;
	//euclidean_norm = ;
	double final_tol = 0.0;
	i = 0;
	final_tol = dnrm2(&n, &res[0], &i)*dpar[0];
	printf("Norm of residual = %E, final tolerance =%E \n", sqrt(dpar[4]), final_tol);
	printf("Average time = %f\n", all_time);
	printf("----------------------------------------\n");
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	//mkl_sparse_destroy(csrL);
	MKL_Free_Buffers();
	free(rhs);
	free(expected_sol);
	free(tmp);
	free(ilu0);
	free(residual);
	free(b);
	free(trvec);
}

//////////////////////////////////////////////////////////////////////////
static void CG_LikeDelFEM(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;

	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	double* y = (double*)malloc(n * sizeof(double));
	double* z = (double*)malloc(n * sizeof(double));
	double* r = (double*)malloc(n * sizeof(double));
	double* p = (double*)malloc(n * sizeof(double));
	double* solution = &_x[0];
	struct matrix_descr descrA;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, matrix.data);
	double all_time = 0.0;
	const int numIterates = 1;
	const double one = 1.0;
	const double zero = 0.0;
	const double eone = -1.0;
	const int ione = 1;
	const int izero = 0;
	int iterates = 0;
	double norm_residual = 0.0;
	double final_tolerance = dnrm2(&n, &res[0], &ione)*tolerance;
	bool converged = false;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		iterates = 0;
		dscal(&n, &zero, solution, &ione);
		//r <= b
		dcopy(&n, &res[0],&ione, r, &ione);
		double sq_norm_res = ddot(&n, r, &ione, r, &ione);
		const double sq_norm_res_inv = 1.0 / sq_norm_res;
		//p <= r
		dcopy(&n, r, &ione, p, &ione);

		do
		{
			// y <= A*p
			mkl_sparse_d_mv(transA, 1.0, csrA, descrA, p, 0.0, y);
			double alpha = sq_norm_res / ddot(&n, p, &ione, y, &ione);
			double ealpha = -alpha;
			// x <= x + alpha*p
			daxpy(&n, &alpha, p, &ione, solution, &ione);
			//r <= r - alpha*y
			daxpy(&n, &ealpha, y, &ione, r, &ione);

			double sq_norm_res_new = ddot(&n, r, &ione, r, &ione);
			if (sq_norm_res_new * sq_norm_res_inv < tolerance*tolerance)
			{
				norm_residual = sqrt(sq_norm_res_new);
				converged = true;
				break;
			}

			double beta = sq_norm_res_new / sq_norm_res;
			sq_norm_res = sq_norm_res_new;
			daxpby(&n, &one, r, &ione, &beta, p, &ione);
			iterates++;
		} while (iterates < max_it);
		converged = converged && (iterates <= max_it);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	printf("----------------CG_LikeDelFEM----------------\n");
	if (converged)
	{
		printf("The system has been solved\n");
		printf("\nNumber of iterations: %d\n", iterates);
		printf("Norm of residual = %E, final tolerance =%E \n", norm_residual, final_tolerance);
		printf("Average time = %f\n", all_time);
		printf("----------------------------------------\n");
	}
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(r);
	free(y);
	free(p);
	free(z);
}

//////////////////////////////////////////////////////////////////////////
static void PCG_LikeDelFEM(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	/*---------------------------------------------------------------------------*/
	/* Define arrays for the upper triangle of the coefficient matrix and rhs vector */
	/* Compressed sparse row storage is used for sparse representation           */
	/*---------------------------------------------------------------------------*/
	const MKL_INT n = matrix.rowPtrSize;
	MKL_INT* rows = (MKL_INT*)matrix.rowPtr;
	MKL_INT* cols = (MKL_INT*)matrix.colInd;
	MKL_INT ipar[128] = { 0 };
	double dpar[128] = { 0.0 };
	/*---------------------------------------------------------------------------*/
	/* Allocate storage for the solver ?par and temporary storage tmp            */
	/*---------------------------------------------------------------------------*/
	double* y = (double*)malloc(n * sizeof(double));
	double* z = (double*)malloc(n * sizeof(double));
	double* r = (double*)malloc(n * sizeof(double));
	double* p = (double*)malloc(n * sizeof(double));
	double* tmp = (double*)malloc(n * sizeof(double));
	double* ilu0 = (double*)malloc(matrix.dataSize * sizeof(double));
	double* x = &_x[0];

	struct matrix_descr descrA, descrL;
	// Structure with sparse matrix stored in CSR format
	sparse_matrix_t       csrA, csrL;
	sparse_operation_t    transA = SPARSE_OPERATION_NON_TRANSPOSE;
	descrA.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	descrA.mode = SPARSE_FILL_MODE_FULL;
	descrA.diag = SPARSE_DIAG_NON_UNIT;
	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, matrix.data);
	

	double all_time = 0.0;
	const int numIterates = 1;
	const double one = 1.0;
	const double zero = 0.0;
	const double eone = -1.0;
	const int ione = 1;
	const int izero = 0;
	int iterates = 0;
	double norm_residual = 0.0;
	double final_tolerance = 0.0;
	bool converged = false;
	final_tolerance = dnrm2(&n, &res[0], &ione) * tolerance;
	for (int it = 0; it < numIterates; it++)
	{
		auto begin = std::chrono::system_clock::now();
		iterates = 0;
		int error;
		dcsrilu0(&n, matrix.data, rows, cols, ilu0, ipar, dpar, &error);
		assert(error == 0);
		mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ONE, n, n, rows, rows + 1, cols, ilu0);

		//x <= 0;
		dscal(&n, &zero, x, &ione);
		//r <= b
		dcopy(&n, &res[0], &ione, r, &ione);
		double sq_norm_res0 = ddot(&n, r, &ione, r, &ione);
		double sq_norm_res0_inv = 1.0 / sq_norm_res0;
		//z <= r;
		dcopy(&n, r, &ione, z, &ione);
		//prec{
		{
			descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
			descrL.mode = SPARSE_FILL_MODE_LOWER;
			descrL.diag = SPARSE_DIAG_UNIT;
			mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, z, tmp);

			descrL.mode = SPARSE_FILL_MODE_UPPER;
			descrL.diag = SPARSE_DIAG_NON_UNIT;
			mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, tmp, z);
		}
		//}
		//p <= z;
		dcopy(&n, z, &ione, p, &ione);
		double inpro_rz = ddot(&n, r, &ione, z, &ione);
		converged = false;
		do 
		{
			//z <= A*p
			mkl_sparse_d_mv(transA, 1.0, csrA, descrA, p, 0.0, z);
			double alpha = inpro_rz / ddot(&n, p, &ione, z, &ione);
			//r <= r - alpa*z
			double ealpha = -1.0*alpha;
			daxpy(&n, &ealpha, z, &ione, r, &ione);
			//x <= x+ alpa*p
			daxpy(&n, &alpha, p, &ione, x, &ione);
			
			double sq_norm_res = ddot(&n, r, &ione, r, &ione);
			if ((sq_norm_res * sq_norm_res0_inv) < (tolerance*tolerance))
			{
				norm_residual = sqrt(sq_norm_res);
				converged = true;
				break;
			}
			//prec{
			dcopy(&n, r, &ione, z, &ione);
			{
				descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
				descrL.mode = SPARSE_FILL_MODE_LOWER;
				descrL.diag = SPARSE_DIAG_UNIT;
				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, z, tmp);

				descrL.mode = SPARSE_FILL_MODE_UPPER;
				descrL.diag = SPARSE_DIAG_NON_UNIT;
				mkl_sparse_d_trsv(transA, 1.0, csrL, descrL, tmp, z);
			}
			//}
			double inpro_rzx_new = ddot(&n, r, &ione, z, &ione);
			double beta = inpro_rzx_new / inpro_rz;
			inpro_rz = inpro_rzx_new;
			// p <= beta*p
			dscal(&n, &beta, p, &ione);
			// p <= p + z
			daxpy(&n, &one, z, &ione, p, &ione);
			iterates++;
		} while (iterates < max_it);
		converged = converged && (iterates <= max_it);
		auto end = std::chrono::system_clock::now();
		all_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	}
	all_time /= (double)numIterates;
	printf("----------------PCG_LikeDelFEM----------------\n");
	if (converged)
	{
		printf("The system has been solved\n");
		printf("\nNumber of iterations: %d\n", iterates);
		printf("Norm of residual = %E, final tolerance =%E \n", norm_residual, final_tolerance);
		printf("Average time = %f\n", all_time);
		printf("----------------------------------------\n");
	}
	/*-------------------------------------------------------------------------*/
	/* Release internal Intel(R) MKL memory that might be used for computations         */
	/* NOTE: It is important to call the routine below to avoid memory leaks   */
	/* unless you disable Intel(R) MKL Memory Manager                                   */
	/*-------------------------------------------------------------------------*/
	mkl_sparse_destroy(csrA);
	MKL_Free_Buffers();
	free(r);
	free(y);
	free(p);
	free(z);
	free(tmp);
	free(ilu0);
}


/*---------------------------------------------------------------------------*/
int main(void)
{
	const int max_it = 5000;
	const double tol = 1.0e-5;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	ReadResVector(res, "res.crs");
	ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
//	CG(matr, res, x, x_delfem, max_it, tol);
//	CG_UserStop(matr, res, x, x_delfem, max_it, tol);
//	CG_LikeCUSP(matr, res, x, x_delfem, max_it, tol);

	for (int i = 0; i < matr.rowPtrSize; i++)
	{
		for (int j = matr.rowPtr[i]; j < matr.rowPtr[i+1]; j++)
		{
			matr.colInd[j] += 1;
		}
	}
	for (int i = 0; i <= matr.rowPtrSize; i++)
	{
		matr.rowPtr[i] += 1;
	}
	
	CG(matr, res, x, x_delfem, max_it, tol);
	CGPrec(matr, res, x, x_delfem, max_it, tol);
	CG_UserStop(matr, res, x, x_delfem, max_it, tol);
	CG_LikeCUSP(matr, res, x, x_delfem, max_it, tol);
	
	CG_LikeDelFEM(matr, res, x, x_delfem, max_it, tol);
	PCG_LikeDelFEM(matr, res, x, x_delfem, max_it, tol);
	GmRes(matr, res, x, x_delfem, max_it, tol);
	GmResPrec(matr, res, x, x_delfem, max_it, tol);

}
