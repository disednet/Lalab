// Gmm_pp.cpp : Defines the entry point for the console application.
//
#include <gmm.h>
#include "../core/sparse_matrix.h"
#include <chrono>
using namespace Core;
typedef gmm::row_matrix<gmm::wsvector<double>> MatrixCSRW;
typedef gmm::csr_matrix<double> MatrixCSRR;
//////////////////////////////////////////////////////////////////////////
static void CG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance)
	{
		gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
		for (int i = 0; i < matrix.rowPtrSize; i++)
		{
			for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
			{
				int col = matrix.colInd[j];
				double data = matrix.data[j];
				mat[i][col] = data;
			}
		}

		std::vector<double> b(res);
		std::vector<double> x(_x);
		gmm::iteration iter(tolerance);
		iter.set_maxiter(max_it);
		//iter.set_noisy(1);
		gmm::csr_matrix<double> mat_r;
		gmm::copy(mat, mat_r);
		gmm::identity_matrix PS;
		gmm::identity_matrix PR;
		auto begin = std::chrono::system_clock::now();
		gmm::cg(mat_r, x, b, PS, PR, iter);
		auto end = std::chrono::system_clock::now();
		double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
		printf("--------------------------------CG---------------------------\n");
		if (iter.converged())
		{
			printf("System of equation was resolved.\n");
			printf("Num iterates = %d\n", iter.get_iteration());
			printf("Residual = %E\n", iter.get_res());
			printf("Duration = %f\n", time);
		}
		else
		{
			printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
		}
	}

//////////////////////////////////////////////////////////////////////////
static void BICGSTAB(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance)
{
	gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			mat[i][col] = data;
		}
	}
	std::vector<double> x(_x);
	std::vector<double> b(res);
	gmm::iteration iter(tolerance);
	iter.set_maxiter(max_it);
	gmm::csr_matrix<double> mat_r;
	gmm::copy(mat, mat_r);
	gmm::identity_matrix PR;
	auto begin = std::chrono::system_clock::now();
	gmm::bicgstab(mat_r, x, b, PR, iter);
	auto end = std::chrono::system_clock::now();
	double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	printf("--------------------------------BICGSTAB---------------------------\n");
	if (iter.converged())
	{
		printf("System of equation was resolved.\n");
		printf("Num iterates = %d\n", iter.get_iteration());
		printf("Residual = %E\n", iter.get_res());
		printf("Duration = %f\n", time);
	}
	else
	{
		printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
	}
}

//////////////////////////////////////////////////////////////////////////
static void GmRes(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance)
{
	gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			mat[i][col] = data;
		}
	}
	std::vector<double> x(_x);
	std::vector<double> b(res);
	gmm::iteration iter(tolerance);
	iter.set_maxiter(max_it);
	gmm::csr_matrix<double> mat_r;
	gmm::copy(mat, mat_r);
	gmm::identity_matrix PR;
	auto begin = std::chrono::system_clock::now();
	const size_t restart = 50;
	gmm::gmres(mat_r, x, b, PR, restart, iter);
	auto end = std::chrono::system_clock::now();
	double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	printf("--------------------------------GmRes---------------------------\n");
	if (iter.converged())
	{
		printf("System of equation was resolved.\n");
		printf("Num iterates = %d\n", iter.get_iteration());
		printf("Residual = %E\n", iter.get_res());
		printf("Duration = %f\n", time);
	}
	else
	{
		printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
	}
}

//////////////////////////////////////////////////////////////////////////
static void QMR(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance)
{
	gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			mat[i][col] = data;
		}
	}
	std::vector<double> x(_x);
	std::vector<double> b(res);
	gmm::iteration iter(tolerance);
	iter.set_maxiter(max_it);
	gmm::csr_matrix<double> mat_r;
	gmm::copy(mat, mat_r);
	gmm::identity_matrix PR;
	auto begin = std::chrono::system_clock::now();
	gmm::qmr(mat_r, x, b, PR, iter);
	auto end = std::chrono::system_clock::now();
	double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	printf("--------------------------------QMR---------------------------\n");
	if (iter.converged())
	{
		printf("System of equation was resolved.\n");
		printf("Num iterates = %d\n", iter.get_iteration());
		printf("Residual = %E\n", iter.get_res());
		printf("Duration = %f\n", time);
	}
	else
	{
		printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
	}
}

//////////////////////////////////////////////////////////////////////////
static void LSCG(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance)
{
	gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
	for (int i = 0; i < matrix.rowPtrSize; i++)
	{
		for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
		{
			int col = matrix.colInd[j];
			double data = matrix.data[j];
			mat[i][col] = data;
		}
	}
	std::vector<double> x(_x);
	std::vector<double> b(res);
	gmm::iteration iter(tolerance);
	iter.set_maxiter(max_it);
	gmm::csr_matrix<double> mat_r;
	gmm::copy(mat, mat_r);
	gmm::identity_matrix PR;
	auto begin = std::chrono::system_clock::now();
	gmm::least_squares_cg(mat_r, x, b, iter);
	auto end = std::chrono::system_clock::now();
	double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
	printf("--------------------------------Least square CG (LSCG)-------------------------\n");
	if (iter.converged())
	{
		printf("System of equation was resolved.\n");
		printf("Num iterates = %d\n", iter.get_iteration());
		printf("Residual = %E\n", iter.get_res());
		printf("Duration = %f\n", time);
	}
	else
	{
		printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
	}
}

namespace Prec
{
	//////////////////////////////////////////////////////////////////////////
	template <class Prec, typename... PrecArgs>
	static void CG_PREC(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance, const std::string& prec_name,
		PrecArgs... args)
	{
		gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
		for (int i = 0; i < matrix.rowPtrSize; i++)
		{
			for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
			{
				int col = matrix.colInd[j];
				double data = matrix.data[j];
				mat[i][col] = data;
			}
		}
		std::vector<double> b(res);
		std::vector<double> x(_x);
		gmm::iteration iter(tolerance);
		iter.set_maxiter(max_it);
		gmm::identity_matrix PS;
		auto begin = std::chrono::system_clock::now();
		Prec PR(mat, std::forward<PrecArgs>(args)...);
		gmm::cg(mat, x, b, PS, PR, iter);

		auto end = std::chrono::system_clock::now();
		double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
		std::string mes = "--------------------------------CG PREC " + prec_name;
		mes += "------------------\n";
		printf(mes.c_str());
		if (iter.converged())
		{
			printf("System of equation was resolved.\n");
			printf("Num iterates = %d\n", iter.get_iteration());
			printf("Residual = %E\n", iter.get_res());
			printf("Duration = %f\n", time);
		}
		else
		{
			printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template <class Prec, typename... PrecArgs>
	static void BiCGSTAB_PREC(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance, const std::string& prec_name,
		PrecArgs... args)
	{
		gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
		for (int i = 0; i < matrix.rowPtrSize; i++)
		{
			for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
			{
				int col = matrix.colInd[j];
				double data = matrix.data[j];
				mat[i][col] = data;
			}
		}
		std::vector<double> b(res);
		std::vector<double> x(_x);
		gmm::iteration iter(tolerance);
		iter.set_maxiter(max_it);
		gmm::identity_matrix PS;
		auto begin = std::chrono::system_clock::now();

		Prec PR(mat, std::forward<PrecArgs>(args)...);
		gmm::bicgstab(mat, x, b, PR, iter);

		auto end = std::chrono::system_clock::now();
		double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
		std::string mes = "--------------------------------BiCGSTAB PREC " + prec_name;
		mes += "------------------\n";
		printf(mes.c_str());
		if (iter.converged())
		{
			printf("System of equation was resolved.\n");
			printf("Num iterates = %d\n", iter.get_iteration());
			printf("Residual = %E\n", iter.get_res());
			printf("Duration = %f\n", time);
		}
		else
		{
			printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	template <class Prec, typename... PrecArgs>
	static void GmRes_PREC(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance, const std::string& prec_name,
		PrecArgs... args)
	{
		gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
		for (int i = 0; i < matrix.rowPtrSize; i++)
		{
			for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
			{
				int col = matrix.colInd[j];
				double data = matrix.data[j];
				mat[i][col] = data;
			}
		}
		std::vector<double> b(res);
		std::vector<double> x(_x);
		gmm::iteration iter(tolerance);
		iter.set_maxiter(max_it);
		gmm::identity_matrix PS;
		auto begin = std::chrono::system_clock::now();
		Prec PR(mat, std::forward<PrecArgs>(args)...);
		gmm::gmres(mat, x, b, PR, 50, iter);
		auto end = std::chrono::system_clock::now();
		double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
		std::string mes = "--------------------------------GmRes PREC " + prec_name;
		mes += "------------------\n";
		printf(mes.c_str());
		if (iter.converged())
		{
			printf("System of equation was resolved.\n");
			printf("Num iterates = %d\n", iter.get_iteration());
			printf("Residual = %E\n", iter.get_res());
			printf("Duration = %f\n", time);
		}
		else
		{
			printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
		}
	}
	
	//////////////////////////////////////////////////////////////////////////
	template <class Prec, typename... PrecArgs>
	static void QMR_PREC(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& _x, const int max_it, const double tolerance, const std::string& prec_name,
		PrecArgs... args)
	{
		gmm::row_matrix<gmm::wsvector<double>> mat(matrix.rowPtrSize, matrix.rowPtrSize);
		for (int i = 0; i < matrix.rowPtrSize; i++)
		{
			for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; j++)
			{
				int col = matrix.colInd[j];
				double data = matrix.data[j];
				mat[i][col] = data;
			}
		}
		std::vector<double> b(res);
		std::vector<double> x(_x);
		gmm::iteration iter(tolerance);
		iter.set_maxiter(max_it);
		gmm::identity_matrix PS;
		auto begin = std::chrono::system_clock::now();
		Prec PR(mat, std::forward<PrecArgs>(args)...);
		gmm::qmr(mat, x, b, PR, iter);
		auto end = std::chrono::system_clock::now();
		double time = (double)(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0;
		std::string mes = "--------------------------------QMR PREC " + prec_name;
		mes += "------------------\n";
		printf(mes.c_str());
		if (iter.converged())
		{
			printf("System of equation was resolved.\n");
			printf("Num iterates = %d\n", iter.get_iteration());
			printf("Residual = %E\n", iter.get_res());
			printf("Duration = %f\n", time);
		}
		else
		{
			printf("System of equation wasn't resolved. MaxtIt = %d\n", max_it);
		}
	}
}


typedef gmm::diagonal_precond<MatrixCSRW> DiagonalPrec;
typedef gmm::ildlt_precond<MatrixCSRW> ILDLTPrec;
typedef gmm::ildltt_precond<MatrixCSRW> ILDLTTPrec;
typedef gmm::ilu_precond<MatrixCSRW> ILU0Prec;
typedef gmm::ilut_precond<MatrixCSRW> ILUnPrec;
typedef gmm::ildltt_precond<MatrixCSRW> ILUnpPrec;
const std::string d_DIAGONAL = "Diagonal";
const std::string d_ILDLT = "ILdLt";
const std::string d_ILDLTT = "IDdLtt";
const std::string d_ILU0 = "ILU0";
const std::string d_ILUn = "ILUn";
const std::string d_ILUnt = "ILUnt";
int main()
{
	const int max_it = 5000;
	const double tolerance = 1.e-4;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	Core::ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	Core::ReadResVector(res, "res.crs");
	Core::ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
	CG(matr, res, x, max_it, tolerance);
 	BICGSTAB(matr, res, x, max_it, tolerance);
 	GmRes(matr, res, x, max_it, tolerance);
 	QMR(matr, res, x, max_it, tolerance);
 	LSCG(matr, res, x, max_it, tolerance);
	Prec::CG_PREC<DiagonalPrec>(matr, res, x, max_it, tolerance, d_DIAGONAL);
	Prec::CG_PREC<ILDLTPrec>(matr, res, x, max_it, tolerance, d_ILDLT);
	//Prec::CG_PREC<ILDLTTPrec>(matr, res, x, max_it, tolerance, d_ILDLTT, 5, 1.0e-16);
	Prec::CG_PREC<ILU0Prec>(matr, res, x, max_it, tolerance, d_ILU0);
	Prec::CG_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUn, 2, 1.0e-10);
	//Prec::CG_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUnt, 2, 1.0e-20);
	
	Prec::BiCGSTAB_PREC<DiagonalPrec>(matr, res, x, max_it, tolerance, d_DIAGONAL);
	Prec::BiCGSTAB_PREC<ILDLTPrec>(matr, res, x, max_it, tolerance, d_ILDLT);
	//Prec::BiCGSTAB_PREC<ILDLTTPrec>(matr, res, x, max_it, tolerance, d_ILDLTT, 5, 1.0e-16);
	Prec::BiCGSTAB_PREC<ILU0Prec>(matr, res, x, max_it, tolerance, d_ILU0);
	Prec::BiCGSTAB_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUn, 2, 1.0e-10);
	//Prec::BiCGSTAB_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUnt, 2, 1.0e-20);

	Prec::GmRes_PREC<DiagonalPrec>(matr, res, x, max_it, tolerance, d_DIAGONAL);
	Prec::GmRes_PREC<ILDLTPrec>(matr, res, x, max_it, tolerance, d_ILDLT);
	//Prec::GmRes_PREC<ILDLTTPrec>(matr, res, x, max_it, tolerance, d_ILDLTT, 5, 1.0e-16);
	Prec::GmRes_PREC<ILU0Prec>(matr, res, x, max_it, tolerance, d_ILU0);
	Prec::GmRes_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUn, 2, 1.0e-10);
	//Prec::GmRes_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUnt, 2, 1.0e-20);

	Prec::QMR_PREC<DiagonalPrec>(matr, res, x, max_it, tolerance, d_DIAGONAL);
	Prec::QMR_PREC<ILDLTPrec>(matr, res, x, max_it, tolerance, d_ILDLT);
	//Prec::QMR_PREC<ILDLTTPrec>(matr, res, x, max_it, tolerance, d_ILDLTT, 5, 1.0e-16);
	Prec::QMR_PREC<ILU0Prec>(matr, res, x, max_it, tolerance, d_ILU0);
	Prec::QMR_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUn, 2, 1.0e-10);
	//Prec::QMR_PREC<ILUnpPrec>(matr, res, x, max_it, tolerance, d_ILUnt, 2, 1.0e-20);
    return 0;
}

