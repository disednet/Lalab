// eigen.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <chrono>
#include "../core/sparse_matrix.h"
using namespace Core;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatr;
typedef Eigen::Triplet<double> Tripletd;
typedef Eigen::VectorXd Vector;

//////////////////////////////////////////////////////////////////////////
static void CG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i+1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::ConjugateGradient<SpMatr, Eigen::Lower | Eigen::Upper> cg;
	cg.setMaxIterations(max_it);
	cg.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	cg.compute(A);
	x = cg.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------CG-----------------\n");
	printf("Num iterates = %d\n", cg.iterations());
	printf("Residual = %E\n", cg.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);
	
}

//////////////////////////////////////////////////////////////////////////
static void BiCGStab(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::BiCGSTAB<SpMatr> bicgstab;
	bicgstab.setMaxIterations(max_it);
	bicgstab.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	bicgstab.compute(A);
	x = bicgstab.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------BiCGSTAB-----------------\n");
	printf("Num iterates = %d\n", bicgstab.iterations());
	printf("Residual = %E\n", bicgstab.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);

}

//////////////////////////////////////////////////////////////////////////
static void PCG_ILU0(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::ConjugateGradient<SpMatr, Eigen::Lower | Eigen::Upper, Eigen::IncompleteLUT<double>> cg;
	cg.setMaxIterations(max_it);
	cg.setTolerance(tolerance);
	cg.analyzePattern(A);
	auto begin = std::chrono::system_clock::now();
	cg.factorize(A);
	x = cg.solve(b);
	//x = cg.compute(A).solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------PCG_ILU0-----------------\n");
	printf("Num iterates = %d\n", cg.iterations());
	printf("Residual = %E\n", cg.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);

}

//////////////////////////////////////////////////////////////////////////
static void PBiCGStab_ILU0(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::BiCGSTAB<SpMatr, Eigen::IncompleteLUT<double>> bicgstab;
	bicgstab.setMaxIterations(max_it);
	bicgstab.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	bicgstab.compute(A);
	x = bicgstab.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------PBiCGSTAB_ILU0-----------------\n");
	printf("Num iterates = %d\n", bicgstab.iterations());
	printf("Residual = %E\n", bicgstab.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);
}

//////////////////////////////////////////////////////////////////////////
static void PCG_Diagonal(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::ConjugateGradient<SpMatr, Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
	cg.setMaxIterations(max_it);
	cg.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	cg.compute(A);
	x = cg.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------PCG_Diagonal-----------------\n");
	printf("Num iterates = %d\n", cg.iterations());
	printf("Residual = %E\n", cg.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);

}

//////////////////////////////////////////////////////////////////////////
static void PBiCGStab_Diagonal(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::BiCGSTAB<SpMatr, Eigen::DiagonalPreconditioner<double>> bicgstab;
	bicgstab.setMaxIterations(max_it);
	bicgstab.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	bicgstab.compute(A);
	x = bicgstab.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------PBiCGSTAB_Diagonal-----------------\n");
	printf("Num iterates = %d\n", bicgstab.iterations());
	printf("Residual = %E\n", bicgstab.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);
}

//////////////////////////////////////////////////////////////////////////
static void LSCG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::LeastSquaresConjugateGradient<SpMatr> cg;
	cg.setMaxIterations(max_it);
	cg.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	cg.compute(A);
	x = cg.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------LSCG-----------------\n");
	printf("Num iterates = %d\n", cg.iterations());
	printf("Residual = %E\n", cg.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);

}

//////////////////////////////////////////////////////////////////////////
static void PLSCG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::LeastSquaresConjugateGradient<SpMatr, Eigen::LeastSquareDiagonalPreconditioner<double>> cg;
	cg.setMaxIterations(max_it);
	cg.setTolerance(tolerance);
	auto begin = std::chrono::system_clock::now();
	cg.compute(A);
	x = cg.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	printf("----------------------PLSCG-----------------\n");
	printf("Num iterates = %d\n", cg.iterations());
	printf("Residual = %E\n", cg.error() * sqrt(b.squaredNorm()));
	printf("Duration = %f\n", duration);

}

//////////////////////////////////////////////////////////////////////////
static void SuperLU(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	SpMatr A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::SparseLU<SpMatr> lu;
	auto begin = std::chrono::system_clock::now();
	lu.analyzePattern(A);
	lu.factorize(A);
	x = lu.solve(b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------SuperLU-----------------\n");
	if (lu.info() == Eigen::ComputationInfo::Success)
	{
		printf("System converged;\n");
	}
	else
	{
		printf("System not converged;\n");
	}
	printf("Duration = %f\n", duration);
}

//////////////////////////////////////////////////////////////////////////
static void LLT(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	Eigen::SparseMatrix<double, Eigen::ColMajor> A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double, Eigen::ColMajor>> sol;
	auto begin = std::chrono::system_clock::now();
	x = sol.compute(A).solve(b);
	auto end = std::chrono::system_clock::now();
	Vector residual = A*x;
	residual = residual - b;
	double resNorm = sqrt(residual.squaredNorm());
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------LLT-----------------\n");
	if (sol.info() == Eigen::ComputationInfo::Success)
	{
		printf("System converged;\n");
		printf("Residual norm = %E", resNorm);
	}
	else
	{
		printf("System not converged;\n");
	}
	printf("Duration = %f\n", duration);
}

//////////////////////////////////////////////////////////////////////////
static void LDLT(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const std::vector<double>& expected_x, const int max_it, const double tolerance)
{
	std::vector<Tripletd> tripletList;
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			tripletList.push_back(Tripletd(i, col, data));
		}
	}
	Eigen::SparseMatrix<double, Eigen::ColMajor> A(matrix.rowPtrSize, matrix.rowPtrSize);
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	Vector x(matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		x[i] = 0.0;
		b[i] = res[i];
	}
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::ColMajor>> sol;
	auto begin = std::chrono::system_clock::now();
	x = sol.compute(A).solve(b);
	auto end = std::chrono::system_clock::now();
	Vector residual = A*x;
	residual = residual - b;
	double resNorm = sqrt(residual.squaredNorm());
	double duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;

	printf("----------------------LDLT-----------------\n");
	if (sol.info() == Eigen::ComputationInfo::Success)
	{
		printf("System converged;\n");
		printf("Residual norm = %E", resNorm);
	}
	else
	{
		printf("System not converged;\n");
	}
	printf("Duration = %f\n", duration);
}

int main()
{
	//Eigen::initParallel();
	const int max_it = 5000;
	const double tolerance = 1.e-4;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	ReadResVector(res, "res.crs");
	ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
 	CG(matr, res, x, x_delfem, max_it, tolerance);
 	BiCGStab(matr, res, x, x_delfem, max_it, tolerance);
 	PCG_ILU0(matr, res, x, x_delfem, max_it, tolerance);
 	PBiCGStab_ILU0(matr, res, x, x_delfem, max_it, tolerance);
 	PCG_Diagonal(matr, res, x, x_delfem, max_it, tolerance);
 	PBiCGStab_Diagonal(matr, res, x, x_delfem, max_it, tolerance);
 	LSCG(matr, res, x, x_delfem, max_it, tolerance);
 	PLSCG(matr, res, x, x_delfem, max_it, tolerance);
 	SuperLU(matr, res, x, x_delfem, max_it, tolerance);
	LLT(matr, res, x, x_delfem, max_it, tolerance);
	LDLT(matr, res, x, x_delfem, max_it, tolerance);
	return 0;
}