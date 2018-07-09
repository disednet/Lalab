#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <iostream>
#include "../core/sparse_matrix.h"
#include <chrono>
using namespace Core;

typedef mtl::compressed2D<double> CrsMatrix;
typedef mtl::dense_vector<double> Vector;

//////////////////////////////////////////////////////////////////////////
template <class Solver, class... Args>
static void Solve(MatrixCRS& matrix
	, const std::vector<double>& res
	, std::vector<double>& _x
	, const int max_it
	, const double tolerance
	, const std::string& solverName
	, Args... args)
{
	std::vector<double> data(matrix.dataSize);
	std::vector<int> rows(matrix.rowPtrSize + 1);
	std::vector<int> cols(matrix.dataSize);
	memcpy(&data[0], matrix.data, matrix.dataSize * sizeof(double));
	memcpy(&rows[0], matrix.rowPtr, (matrix.rowPtrSize + 1) * sizeof(int));
	memcpy(&cols[0], matrix.colInd, (matrix.dataSize) * sizeof(int));
	CrsMatrix mat(matrix.rowPtrSize, matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	Vector x(matrix.rowPtrSize, 0.0);
	mat.set_nnz(matrix.dataSize);
	mat.raw_copy(data.begin(), data.end(), rows.begin(), cols.begin());
	for (int i = 0; i < res.size(); i++)
	{
		b[i] = res[i];
	}
	std::cout << "------------------------------" << solverName << "-------------" << std::endl;
	auto begin = std::chrono::system_clock::now();
	itl::basic_iteration<double> iter(b, max_it, tolerance);
	Solver solver(mat, std::forward<Args>(args)...);
	solver.solve(b, x, iter);
	auto end = std::chrono::system_clock::now();
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "Duration = " << duration << "ms" << std::endl;
	if (iter.converged())
	{
		std::cout << "finished! error code = " << iter.error_code() << '\n'
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n"
			<< "Relative tol: " << tolerance << '\n'
			<< "Convergence:  " << pow(iter.relresid(), 1.0 / double(iter.iterations())) << std::endl;
	}
	else
	{
		std::cout << "system wasn't resolved" << "\n"
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n";
	}
		
}

//////////////////////////////////////////////////////////////////////////
template <class Solver, class Prec, class...PrecArgs>
static void SolveWithPrec(MatrixCRS& matrix, const std::vector<double>& res
	, std::vector<double>& _x
	, const int max_it
	, const double tolerance
	, const std::string& solverName
	, const std::string& precName
	, PrecArgs... args)
{
	std::vector<double> data(matrix.dataSize);
	std::vector<int> rows(matrix.rowPtrSize + 1);
	std::vector<int> cols(matrix.dataSize);
	memcpy(&data[0], matrix.data, matrix.dataSize * sizeof(double));
	memcpy(&rows[0], matrix.rowPtr, (matrix.rowPtrSize + 1) * sizeof(int));
	memcpy(&cols[0], matrix.colInd, (matrix.dataSize) * sizeof(int));
	CrsMatrix mat(matrix.rowPtrSize, matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	Vector x(matrix.rowPtrSize, 0.0);
	mat.set_nnz(matrix.dataSize);
	mat.raw_copy(data.begin(), data.end(), rows.begin(), cols.begin());
	for (int i = 0; i < res.size(); i++)
	{
		b[i] = res[i];
	}
	std::cout << "-----------------" << solverName << "  " << precName << "-------------" << std::endl;
	auto begin = std::chrono::system_clock::now();
	itl::basic_iteration<double> iter(b, max_it, tolerance);
	Prec prec(mat, std::forward<PrecArgs>(args)...);
	Solver solver(mat, prec);
	solver.solve(b, x, iter);
	auto end = std::chrono::system_clock::now();
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "Duration = " << duration << "ms" << std::endl;
	if (iter.converged())
	{
		std::cout << "finished! error code = " << iter.error_code() << '\n'
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n"
			<< "Relative tol: " << tolerance << '\n'
			<< "Convergence:  " << pow(iter.relresid(), 1.0 / double(iter.iterations())) << std::endl;
	}
	else
	{
		std::cout << "system wasn't resolved" << "\n"
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n";
	}

}

//////////////////////////////////////////////////////////////////////////
template <class Prec, class...PrecArgs>
static void GmResPrec(MatrixCRS& matrix, const std::vector<double>& res
	, std::vector<double>& _x
	, const int max_it
	, const double tolerance
	, int restart
	, const std::string& precName
	, PrecArgs... args)
{
	std::vector<double> data(matrix.dataSize);
	std::vector<int> rows(matrix.rowPtrSize + 1);
	std::vector<int> cols(matrix.dataSize);
	memcpy(&data[0], matrix.data, matrix.dataSize * sizeof(double));
	memcpy(&rows[0], matrix.rowPtr, (matrix.rowPtrSize + 1) * sizeof(int));
	memcpy(&cols[0], matrix.colInd, (matrix.dataSize) * sizeof(int));
	CrsMatrix mat(matrix.rowPtrSize, matrix.rowPtrSize);
	Vector b(matrix.rowPtrSize);
	Vector x(matrix.rowPtrSize, 0.0);
	mat.set_nnz(matrix.dataSize);
	mat.raw_copy(data.begin(), data.end(), rows.begin(), cols.begin());
	for (int i = 0; i < res.size(); i++)
	{
		b[i] = res[i];
	}
	std::cout << "-----------------GmRes  " << precName << "-------------" << std::endl;
	auto begin = std::chrono::system_clock::now();
	itl::basic_iteration<double> iter(b, max_it, tolerance);
	Prec prec(mat, std::forward<PrecArgs>(args)...);
	itl::gmres_solver<CrsMatrix, Prec> solver(mat, restart, prec);
	solver.solve(b, x, iter);
	auto end = std::chrono::system_clock::now();
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "Duration = " << duration << "ms" << std::endl;
	if (iter.converged())
	{
		std::cout << "finished! error code = " << iter.error_code() << '\n'
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n"
			<< "Relative tol: " << tolerance << '\n'
			<< "Convergence:  " << pow(iter.relresid(), 1.0 / double(iter.iterations())) << std::endl;
	}
	else
	{
		std::cout << "system wasn't resolved" << "\n"
			<< iter.iterations() << " iterations\n"
			<< iter.resid() << " is actual final residual. \n"
			<< iter.relresid() << " is actual relative tolerance achieved. \n";
	}

}

//////////////////////////////////////////////////////////////////////////
typedef itl::cg_solver<CrsMatrix> CGSolver;
typedef itl::bicg_solver<CrsMatrix> BiCGSolver;
typedef itl::bicgstab_solver<CrsMatrix> BiCGSTABSolver;
typedef itl::bicgstab_2_solver<CrsMatrix> BiCGSTAB2Solver;
typedef itl::bicgstab_ell_solver<CrsMatrix> BiCGSTAB_Ell_Solver;
typedef itl::cgs_solver<CrsMatrix> CGSSolver;
typedef itl::gmres_solver<CrsMatrix> GmResSolver;
typedef itl::idr_s_solver<CrsMatrix> IDRSSolver;
typedef itl::qmr_solver<CrsMatrix> QMRSolver;
typedef itl::tfqmr_solver<CrsMatrix> TFQMRSolver;
//////////////////////////////////////////////////////////////////////////
const std::string CG_SOLVER_NAME = "CG";
const std::string BiCG_SOLVER_NAME = "BiCG";
const std::string BiCGSTAB_SOLVER_NAME = "BiCGSTAB";
const std::string BiCGSTAB2_SOLVER_NAME = "BiCGSTAB2";
const std::string BiCGSTAB_ELL_SOLVER_NAME = "BiCGSTAB_ELL";
const std::string CGS_SOLVER_NAME = "CGS(LSCG)";
const std::string GMRES_SOLVER_NAME = "GmRes";
const std::string GMRES_WITH_RESTART_SOLVER_NAME = "GmRes with restart";
const std::string IDRS_SOLVER_NAME = "IDR_S";
const std::string QMR_SOLVER_NAME = "QMR";
const std::string TFQMR_SOLVER_NAME = "TFQMR";
//////////////////////////////////////////////////////////////////////////
typedef itl::pc::diagonal<CrsMatrix> DiagPrec;
typedef itl::pc::ilu_0<CrsMatrix> ILU0Prec;
typedef itl::pc::ilut<CrsMatrix> ILUtPrec;
typedef itl::pc::ic_0<CrsMatrix> ICPrec;

//////////////////////////////////////////////////////////////////////////
const std::string ILU0_PREC = "ILU0";
const std::string ILUt_PREC = "ILUt";
const std::string DIAG_PREC = "Diagonal inversion";
const std::string IC_PREC = "Incmoplete Cholesky";


int main()
{
	const int max_it = 5000;
	const double tol = 1.0e-4;
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	ReadResVector(res, "res.crs");
	ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
	Solve<CGSolver>(matr, res, x, max_it, tol, CG_SOLVER_NAME);
	Solve<BiCGSolver>(matr, res, x, max_it, tol, BiCG_SOLVER_NAME);
	Solve<BiCGSTABSolver>(matr, res, x, max_it, tol, BiCGSTAB_SOLVER_NAME);
	Solve<BiCGSTAB2Solver>(matr, res, x, max_it, tol, BiCGSTAB2_SOLVER_NAME);
	Solve<BiCGSTAB_Ell_Solver>(matr, res, x, max_it, tol, BiCGSTAB_ELL_SOLVER_NAME, 2);
	Solve<CGSSolver>(matr, res, x, max_it, tol, CGS_SOLVER_NAME);
	Solve<GmResSolver>(matr, res, x, max_it, tol, GMRES_SOLVER_NAME);
	Solve<GmResSolver>(matr, res, x, max_it, tol, GMRES_WITH_RESTART_SOLVER_NAME, 150);
	Solve<IDRSSolver>(matr, res, x, max_it, tol, IDRS_SOLVER_NAME, 2);
	Solve<QMRSolver>(matr, res, x, max_it, tol, QMR_SOLVER_NAME);
	Solve<TFQMRSolver>(matr, res, x, max_it, tol, TFQMR_SOLVER_NAME);

	//////////////////////////////////////////////////////////////////////////
	const int t = 1;
	const double ilut_threshold = 1.0e-20;
	const int gmres_restart = 50;
	SolveWithPrec<itl::cg_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, CG_SOLVER_NAME, DIAG_PREC);
 	SolveWithPrec<itl::cg_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, CG_SOLVER_NAME, ILU0_PREC);
// 	SolveWithPrec<itl::cg_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, CG_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
 	SolveWithPrec<itl::cg_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, CG_SOLVER_NAME, IC_PREC);

 	SolveWithPrec<itl::bicg_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, BiCG_SOLVER_NAME, DIAG_PREC);
 	SolveWithPrec<itl::bicg_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, BiCG_SOLVER_NAME, ILU0_PREC);
// 	SolveWithPrec<itl::bicg_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, BiCG_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
 	SolveWithPrec<itl::bicg_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, BiCG_SOLVER_NAME, IC_PREC);
 
	SolveWithPrec<itl::bicgstab_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, BiCGSTAB_SOLVER_NAME, DIAG_PREC);
	SolveWithPrec<itl::bicgstab_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, BiCGSTAB_SOLVER_NAME, ILU0_PREC);
//	SolveWithPrec<itl::bicgstab_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, BiCGSTAB_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
	SolveWithPrec<itl::bicgstab_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, BiCGSTAB_SOLVER_NAME, IC_PREC);

	SolveWithPrec<itl::bicgstab_2_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, BiCGSTAB2_SOLVER_NAME, DIAG_PREC);
	SolveWithPrec<itl::bicgstab_2_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, BiCGSTAB2_SOLVER_NAME, ILU0_PREC);
//	SolveWithPrec<itl::bicgstab_2_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, BiCGSTAB2_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
	SolveWithPrec<itl::bicgstab_2_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, BiCGSTAB2_SOLVER_NAME, IC_PREC);
	
	SolveWithPrec<itl::cgs_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, CGS_SOLVER_NAME, DIAG_PREC);
	SolveWithPrec<itl::cgs_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, CGS_SOLVER_NAME, ILU0_PREC);
//	SolveWithPrec<itl::cgs_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, CGS_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
	SolveWithPrec<itl::cgs_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, CGS_SOLVER_NAME, IC_PREC);
	
	GmResPrec<DiagPrec>(matr, res, x, max_it, tol, gmres_restart, DIAG_PREC);
 	GmResPrec<ILU0Prec>(matr, res, x, max_it, tol, gmres_restart, ILU0_PREC);
// 	GmResPrec<ILUtPrec>(matr, res, x, max_it, tol, gmres_restart, ILUt_PREC, t, ilut_threshold);
 	GmResPrec<ICPrec>(matr, res, x, max_it, tol, gmres_restart, IC_PREC);

	SolveWithPrec<itl::qmr_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, QMR_SOLVER_NAME, DIAG_PREC);
	SolveWithPrec<itl::qmr_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, QMR_SOLVER_NAME, ILU0_PREC);
//	SolveWithPrec<itl::qmr_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, QMR_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
	SolveWithPrec<itl::qmr_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, QMR_SOLVER_NAME, IC_PREC);

	SolveWithPrec<itl::tfqmr_solver<CrsMatrix, DiagPrec>, DiagPrec>(matr, res, x, max_it, tol, TFQMR_SOLVER_NAME, DIAG_PREC);
	SolveWithPrec<itl::tfqmr_solver<CrsMatrix, ILU0Prec>, ILU0Prec>(matr, res, x, max_it, tol, TFQMR_SOLVER_NAME, ILU0_PREC);
//	SolveWithPrec<itl::tfqmr_solver<CrsMatrix, ILUtPrec>, ILUtPrec>(matr, res, x, max_it, tol, TFQMR_SOLVER_NAME, ILUt_PREC, t, ilut_threshold);
	SolveWithPrec<itl::tfqmr_solver<CrsMatrix, ICPrec>, ICPrec>(matr, res, x, max_it, tol, TFQMR_SOLVER_NAME, IC_PREC);
    return 0;
}

