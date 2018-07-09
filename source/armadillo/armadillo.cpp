#include <iostream>
#include "../core/sparse_matrix.h"
#include <armadillo>
#include <assert.h>
#include <chrono>
using namespace Core;

//////////////////////////////////////////////////////////////////////////
static void Solve(const MatrixCCS& matr, const std::vector<double>& _b, std::vector<double>& _x, const unsigned int max_it, const double tollerance)
{
	//arma::SpMat<double> matrix(matr.colPtrSize, matr.colPtrSize);
	std::vector<long long unsigned int> r(matr.dataSize);
	std::vector<long long unsigned int> c(matr.dataSize);
	std::vector<double> data(matr.dataSize);
	unsigned int index = 0;
	for (unsigned int i = 0; i < matr.colPtrSize; i++)
	{
		for (unsigned int j = matr.colPtr[i]; j < matr.colPtr[i+1]; j++)
		{
			//matrix(matr.rowInd[j],i) = matr.data[j];
			r[index] = matr.rowInd[j];
			c[index] = i;
			data[index] = matr.data[j];
			index++;
		}
	}
	arma::umat lr(r);
	arma::umat lc(c);
	arma::umat loc(arma::join_rows(lr, lc).t());
	arma::SpMat<double> matrix(loc, arma::vec(data));
	arma::vec x = arma::zeros<arma::Col<double>>(_x.size());
	arma::vec b(_b);
	auto begin = std::chrono::system_clock::now();
	bool result = arma::spsolve(x, matrix, b);
	auto end = std::chrono::system_clock::now();
	double duration = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
	std::cout << "-------------------------SuperLU------------" << std::endl;
	std::cout << "Duration time = " << duration << std::endl;
	if (result)
	{
		std::cout << " System was resolved." << std::endl;

	}
	else
	{
		std::cout << " System wasn't resolved." << std::endl;
	}
}

int main()
{
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
	MatrixCCS matr_ccs;
	ConverteCRS2CCS(matr, matr_ccs);
	//assert(IsEqual(matr_ccs, matr));
	Solve(matr_ccs, res, x, max_it, tollerance);
	return 1;
}