#include <iostream>
#include <chrono>
#include "../core/sparse_matrix.h"
#include <viennacl/version.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/lu.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/row_scaling.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
using namespace Core;
typedef viennacl::compressed_matrix<double> SparseMatrix;
typedef viennacl::vector<double> DenseVector;
//////////////////////////////////////////////////////////////////////////
template<class Solver>
static void IterativeSolver(const MatrixCRS& A, const std::vector<double>& _b, std::vector<double>&_x, const unsigned int max_it, const double tollerance, const std::string& solverName)
{
	SparseMatrix matr(A.rowPtrSize, A.rowPtrSize);
	matr.set(A.rowPtr, A.colInd, A.data, A.rowPtrSize, A.rowPtrSize, A.dataSize);
 	DenseVector b(_b.size());
 	viennacl::copy(_b.begin(), _b.end(), b.begin());
	std::cout << "---------------" << solverName << "------------------" << std::endl;
	auto begin = std::chrono::system_clock::now();
 	Solver solver(tollerance, max_it);
	DenseVector x = viennacl::linalg::solve(matr, b, solver);
	viennacl::copy(x, _x);
	auto end = std::chrono::system_clock::now();
	DenseVector bn = viennacl::linalg::prod(matr, x);
	DenseVector residual = bn - b;
	double norm_res = viennacl::linalg::norm_2(residual);
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "Duration= " << duration << "ms." << std::endl;
	std::cout << "Iterates = " << solver.iters() << std::endl;
	std::cout << "Estimation error= " << solver.error() << std::endl;
	std::cout << "Final residual norm = " << norm_res << std::endl;
}

//////////////////////////////////////////////////////////////////////////
template <class Solver, class Prec, class Tag,  class... PrecArg>
static void IterativeSolverPrec(const MatrixCRS& A, 
	const std::vector<double>& _b, 
	std::vector<double>&_x, 
	const unsigned int max_it, 
	const double tollerance, 
	const std::string& solverName,
	const std::string& precName,
	PrecArg... args)
{
	SparseMatrix matr(A.rowPtrSize, A.rowPtrSize);
	matr.set(A.rowPtr, A.colInd, A.data, A.rowPtrSize, A.rowPtrSize, A.dataSize);
	DenseVector b(_b.size());
	viennacl::copy(_b.begin(), _b.end(), b.begin());
	std::cout << "---------------" << solverName<< " " << precName << "------------------" << std::endl;
	auto begin = std::chrono::system_clock::now();
	Tag tag_config(std::forward<PrecArg>(args)...);
	Prec ilut(matr, tag_config);
	Solver solver(tollerance, max_it);
	DenseVector x = viennacl::linalg::solve(matr, b, solver, ilut);
	viennacl::copy(x, _x);
	auto end = std::chrono::system_clock::now();
	

	DenseVector bn = viennacl::linalg::prod(matr, x);
	DenseVector residual = bn - b;
	double norm_res = viennacl::linalg::norm_2(residual);
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "Duration= " << duration << "ms." << std::endl;
	std::cout << "Iterates = " << solver.iters() << std::endl;
	std::cout << "Estimation error= " << solver.error() << std::endl;
	std::cout << "Final residual norm = " << norm_res << std::endl;
}


typedef viennacl::linalg::cg_tag CG_Solver;
typedef viennacl::linalg::bicgstab_tag BiCGSTAB_Solver;
typedef viennacl::linalg::gmres_tag GmRes_Solver;

const std::string CG_SOLVER_NAME = "CG";
const std::string BiCGSTAB_SOLVER_NAME = "BiCGSTAB";
const std::string GmRes_SOLVER_NAME = "GmRes";

//////////////////////////////////////////////////////////////////////////
const std::string ILUt_PREC_NAME = "ILUt";
const std::string ILU0_PREC_NAME = "ILU0";
const std::string ICHOL0_PREC_NAME = "ICHOL0";
const std::string BLK_ILU_PREC_NAME = "Block-ILU";
const std::string DIAG_PREC_NAME = "Diagonal";
const std::string JACOBI_PREC_NAME = "Jacobi";

int main()
{
	using namespace viennacl::linalg;
	std::cout << "ViennaCL v." << VIENNACL_MAJOR_VERSION 
		<< "." << VIENNACL_MINOR_VERSION 
		<< "." << VIENNACL_PATCH_VERSION << std::endl;
	const unsigned int max_it = 5000;
	const double tollerance = 1.0e-4;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	Core::ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	Core::ReadResVector(res, "res.crs");
	Core::ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
	IterativeSolver<CG_Solver>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME);
	IterativeSolver<BiCGSTAB_Solver>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME);
	IterativeSolver<GmRes_Solver>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME);

//  	IterativeSolverPrec<CG_Solver, ilu0_precond<SparseMatrix>, ilu0_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, ILU0_PREC_NAME);
//  	IterativeSolverPrec<CG_Solver, ilut_precond<SparseMatrix>, ilut_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, ILUt_PREC_NAME);
//  	IterativeSolverPrec<CG_Solver, ichol0_precond<SparseMatrix>, ichol0_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, ICHOL0_PREC_NAME);
// 	IterativeSolverPrec<CG_Solver, block_ilu_precond<SparseMatrix, ilu0_tag>, ilu0_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, BLK_ILU_PREC_NAME);
// 	IterativeSolverPrec<CG_Solver, row_scaling<SparseMatrix>, row_scaling_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, DIAG_PREC_NAME);
// 	IterativeSolverPrec<CG_Solver, jacobi_precond<SparseMatrix>, jacobi_tag>(matr, res, x, max_it, tollerance, CG_SOLVER_NAME, JACOBI_PREC_NAME);
// 
// 	IterativeSolverPrec<BiCGSTAB_Solver, ilu0_precond<SparseMatrix>, ilu0_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, ILU0_PREC_NAME);
// 	IterativeSolverPrec<BiCGSTAB_Solver, ilut_precond<SparseMatrix>, ilut_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, ILUt_PREC_NAME);
// 	IterativeSolverPrec<BiCGSTAB_Solver, ichol0_precond<SparseMatrix>, ichol0_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, ICHOL0_PREC_NAME);
// 	IterativeSolverPrec<BiCGSTAB_Solver, block_ilu_precond<SparseMatrix, ilu0_tag>, ilu0_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, BLK_ILU_PREC_NAME);
// 	IterativeSolverPrec<BiCGSTAB_Solver, row_scaling<SparseMatrix>, row_scaling_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, DIAG_PREC_NAME);
// 	IterativeSolverPrec<BiCGSTAB_Solver, jacobi_precond<SparseMatrix>, jacobi_tag>(matr, res, x, max_it, tollerance, BiCGSTAB_SOLVER_NAME, JACOBI_PREC_NAME);
// 
// 	IterativeSolverPrec<GmRes_Solver, ilu0_precond<SparseMatrix>, ilu0_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, ILU0_PREC_NAME);
// 	IterativeSolverPrec<GmRes_Solver, ilut_precond<SparseMatrix>, ilut_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, ILUt_PREC_NAME);
// 	IterativeSolverPrec<GmRes_Solver, ichol0_precond<SparseMatrix>, ichol0_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, ICHOL0_PREC_NAME);
// 	IterativeSolverPrec<GmRes_Solver, block_ilu_precond<SparseMatrix, ilu0_tag>, ilu0_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, BLK_ILU_PREC_NAME);
// 	IterativeSolverPrec<GmRes_Solver, row_scaling<SparseMatrix>, row_scaling_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, DIAG_PREC_NAME);
// 	IterativeSolverPrec<GmRes_Solver, jacobi_precond<SparseMatrix>, jacobi_tag>(matr, res, x, max_it, tollerance, GmRes_SOLVER_NAME, JACOBI_PREC_NAME);
	return 1;
}