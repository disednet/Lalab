#include <stdio.h>
#include "iostream"
#include <vector>
#include <chrono>
#include <cuda.h>
#include <thrust/version.h>
#include <cusp/version.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/krylov/cr.h>
#include <cusp/krylov/bicg.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/bicgstab_m.h>
#include <cusp/krylov/gmres.h>
#include <cusp/precond/diagonal.h>
#include <cusp/precond/ainv.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include "../core/sparse_matrix.h"
using namespace Core;
typedef double scalar;
#define TOLERANCE 1.0e-4
#define MAX_ITS 5000
#define NUM_EXPERIMENTS 1
////////////////////////////////////////////////////////////////////////////
extern "C"
int testCuda()
{
	int cuda_major = CUDA_VERSION / 1000;
	int cuda_minor = (CUDA_VERSION % 1000) / 10;
	int thrust_major = THRUST_MAJOR_VERSION;
	int thrust_minor = THRUST_MINOR_VERSION;
	int cusp_major = CUSP_MAJOR_VERSION;
	int cusp_minor = CUSP_MINOR_VERSION;
	std::cout << "CUDA v" << cuda_major << "." << cuda_minor << std::endl;
	std::cout << "THRUST v" << thrust_major << "." << thrust_minor << std::endl;
	std::cout << "CUSP v" << cusp_major << "." << cusp_minor << std::endl;
	return 1;
}


///////////////////////////////////////////////////////////////////////////
extern "C"
double TestBicg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	cusp::identity_operator<scalar, cusp::device_memory> M(A.num_rows, A.num_rows);
	for (unsigned int step = 0; step < numExperiments; step++)
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;
		dev_A = A;
		dev_res = host_res;
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		cusp::krylov::bicg(dev_A, dev_A, dev_x, dev_res, monitor, M, M);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
	}
	avg_time /= (double)numExperiments;
	for (size_t i = 0; i < res.size(); i++)
		x[i] = (double)host_x[i];
	return avg_time;
}

/////////////////////////////////////////////////////////////////////////////
extern "C"
double TestBiCgStab(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	// set preconditioner (identity)
	cusp::identity_operator<scalar, cusp::device_memory> M(A.num_rows, A.num_rows);
	bool converged = false;
	for (unsigned int step = 0; step < numExperiments; step++)
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;
		dev_A = A;
		dev_res = host_res;
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		cusp::krylov::bicgstab(dev_A, dev_x, dev_res, monitor, M);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
		converged = monitor.converged();
		if (!converged) break;

	}
	if (converged)
	{
		avg_time /= (double)numExperiments;
		for (size_t i = 0; i < res.size(); i++)
			x[i] = (double)host_x[i];
	}
	return avg_time;
}


/////////////////////////////////////////////////////////////////////////////
extern "C"
double TestCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	for (unsigned int step = 0; step < numExperiments; step++)
	{
		cusp::hyb_matrix<int, double, cusp::host_memory> hyb_host = A;
		
		std::chrono::time_point<std::chrono::system_clock> start, end;
		cusp::hyb_matrix<int, double, cusp::device_memory> hyb_dev = hyb_host;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;
		dev_A = A;
		dev_res = host_res;
		
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		cusp::krylov::cg(hyb_dev, dev_x, dev_res, monitor);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
	}
	avg_time /= (double)numExperiments;
	for (size_t i = 0; i < res.size(); i++)
		x[i] = (double)host_x[i];
	return avg_time;
}

/////////////////////////////////////////////////////////////////////////////
extern "C"
double TestCr(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	// set preconditioner (identity)
	cusp::identity_operator<scalar, cusp::device_memory> M(A.num_rows, A.num_rows);
	for (unsigned int step = 0; step < numExperiments; step++)
	{

		std::chrono::time_point<std::chrono::system_clock> start, end;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;
		dev_A = A;
		dev_res = host_res;
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		cusp::krylov::cr(dev_A, dev_x, dev_res, monitor, M);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
	}
	avg_time /= (double)numExperiments;
	for (size_t i = 0; i < res.size(); i++)
		x[i] = (double)host_x[i];
	return avg_time;
}


/////////////////////////////////////////////////////////////////////////////
extern "C"
double TestGmRes(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	// set preconditioner (identity)
	cusp::identity_operator<scalar, cusp::device_memory> M(A.num_rows, A.num_rows);
	int restart = 50;
	for (unsigned int step = 0; step < numExperiments; step++)
	{

		std::chrono::time_point<std::chrono::system_clock> start, end;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;

		dev_A = A;
		dev_res = host_res;
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		cusp::krylov::gmres(dev_A, dev_x, dev_res, restart, monitor, M);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
	}
	avg_time /= (double)numExperiments;
	for (size_t i = 0; i < res.size(); i++)
		x[i] = (double)host_x[i];
	return avg_time;
}


/////////////////////////////////////////////////////////////////////////////
extern "C"
double TestPCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x)
{
	double avg_time = 0.0;
	cusp::csr_matrix<int, scalar, cusp::host_memory> A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	for (unsigned int i = 0; i < matrix.rowPtrSize + 1; i++)
	{
		A.row_offsets[i] = matrix.rowPtr[i];
	}
	for (unsigned int i = 0; i < matrix.dataSize; i++)
	{
		A.column_indices[i] = matrix.colInd[i];
		A.values[i] = (scalar)matrix.data[i];
	}
	cusp::array1d<scalar, cusp::host_memory> host_res(res.size());
	for (size_t i = 0; i < res.size(); i++)
		host_res[i] = res[i];
	const unsigned int numExperiments = NUM_EXPERIMENTS;
	cusp::csr_matrix<int, scalar, cusp::device_memory> dev_A(
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.rowPtrSize,
		(size_t)matrix.dataSize);
	cusp::array1d<scalar, cusp::device_memory> dev_res(res.size());
	cusp::array1d<scalar, cusp::device_memory> dev_x(res.size());
	cusp::array1d<scalar, cusp::host_memory> host_x(res.size());
	bool converged = false;
	for (unsigned int step = 0; step < numExperiments; step++)
	{
		std::chrono::time_point<std::chrono::system_clock> start, end;
		for (size_t i = 0; i < host_x.size(); i++)
		{
			host_x[i] = 0.0;
		}
		start = std::chrono::system_clock::now();
		dev_x = host_x;

		dev_A = A;
		dev_res = host_res;
		cusp::monitor<scalar> monitor(dev_res, MAX_ITS, TOLERANCE, 0, false);
		// set preconditioner (identity)
		//cusp::precond::scaled_bridson_ainv<scalar, cusp::device_memory>M(dev_A, 0, 10);
		//cusp::identity_operator<scalar, cusp::device_memory> M(A.num_rows, A.num_rows);
		//cusp::precond::aggregation::smoothed_aggregation<int, scalar, cusp::device_memory> M(dev_A);
		cusp::precond::diagonal<scalar, cusp::device_memory> M(dev_A);
		cusp::krylov::cg(dev_A, dev_x, dev_res, monitor, M);
		host_x = dev_x;
		end = std::chrono::system_clock::now();
		avg_time += (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1000.0;
		if (step == 0)
		{
			monitor.print();
			std::cout << "---------------------------------------------------" << std::endl;
		}
		converged = monitor.converged();
		if (!converged) break;

	}
	if (converged)
	{
		avg_time /= (double)numExperiments;
		for (size_t i = 0; i < res.size(); i++)
			x[i] = (double)host_x[i];
	}
	return avg_time;
}

