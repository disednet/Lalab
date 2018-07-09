#include <iostream>
#include <vector>
#include <string>
#include "../core/sparse_matrix.h"
using namespace Core;

extern "C" int testCuda();
extern "C" double TestBicg(MatrixCRS& matrix, const std::vector<double>&res, std::vector<double>& x);
extern "C" double TestBiCgStab(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestCr(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestGmRes(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestPCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);


int main()
{
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	Core::ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	Core::ReadResVector(res, "res.crs");
	Core::ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
	try 
	{
		testCuda();
		{
			double time = TestCg(matr, res, x);
			std::cout << "Processed CG finish! Average time = " << time << "ms." << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestBicg(matr, res, x);
			std::cout << "Processed BiCG finish! Average time = " << time << "ms." << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestBiCgStab(matr, res, x);
		 	std::cout << "Processed BiCGStab finish! Average time = " << time << "ms." << std::endl;
		 	std::cout << "--------------------------------------------------------" << std::endl;
		 	std::cout << std::endl;
		 	std::cout << std::endl;
		}

		{
			double time = TestCr(matr, res, x);
			std::cout << "Processed Cr finish(only for symmetric matrix)! Average time = " << time << "ms." << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestGmRes(matr, res, x);
			std::cout << "Processed GmRes finish(for non symmetric matrix)! Average time = " << time << "ms." << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestPCg(matr, res, x);
			std::cout << "Processed PCG finish! Average time = " << time << "ms." << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}

	}
	catch (std::exception& exc)
	{
		std::cout << exc.what();
	}
	
	return 0;
}