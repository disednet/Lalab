#include "kernel.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
extern "C" int testCuda();
extern "C" double TestBicg(MatrixCRS& matrix, const std::vector<double>&res, std::vector<double>& x);
extern "C" double TestBiCgStab(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestCr(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestGmRes(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
extern "C" double TestPCg(MatrixCRS& matrix, const std::vector<double>& res, std::vector<double>& x);
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
	long colSize = file_out_cols.tellg() / sizeof(unsigned int );
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

int main()
{
	MatrixCRS matr;
	std::vector<double> res;
	std::vector<double> x;
	std::vector<double> x_delfem;
	ReadCSRMatrixFromBinary(matr, "current_matrix.crs");
	ReadResVector(res, "res.crs");
	ReadResVector(x_delfem, "x.crs");
	x.resize(res.size());
	try 
	{
		testCuda();
		{
			double time = TestCg(matr, res, x);
			std::cout << "Processed CG finish! Average time = " << time << "ms." << std::endl;
			double err = 0.0;
			for (unsigned int i = 0; i < x_delfem.size(); i++)
			{
				err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);

			}
			err /= (double)x_delfem.size();
			std::cout << "Error = " << err << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestBicg(matr, res, x);
			std::cout << "Processed BiCG finish! Average time = " << time << "ms." << std::endl;
			double err = 0.0;
			for (unsigned int i = 0; i < x_delfem.size(); i++)
			{
				err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);

			}
			err /= (double)x_delfem.size();
			std::cout << "Error = " << err << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestBiCgStab(matr, res, x);
		 	std::cout << "Processed BiCGStab finish! Average time = " << time << "ms." << std::endl;
		 	double err = 0.0;
		 	for (unsigned int i = 0; i < x_delfem.size(); i++)
		 	{
		 		err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);
		 
		 	}
		 	err /= (double)x_delfem.size();
		 	std::cout << "Error = " << err << std::endl;
		 	std::cout << "--------------------------------------------------------" << std::endl;
		 	std::cout << std::endl;
		 	std::cout << std::endl;
		}

		

		{
			double time = TestPCg(matr, res, x);
			std::cout << "Processed PCG finish! Average time = " << time << "ms." << std::endl;
			double err = 0.0;
			for (unsigned int i = 0; i < x_delfem.size(); i++)
			{
				err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);

			}
			err /= (double)x_delfem.size();
			std::cout << "Error = " << err << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestCr(matr, res, x);
			std::cout << "Processed Cr finish(only for symmetric matrix)! Average time = " << time << "ms." << std::endl;
			double err = 0.0;
			for (unsigned int i = 0; i < x_delfem.size(); i++)
			{
				err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);
			}
			err /= (double)x_delfem.size();
			std::cout << "Error = " << err << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
			std::cout << std::endl;
			std::cout << std::endl;
		}
		{
			double time = TestGmRes(matr, res, x);
			std::cout << "Processed GmRes finish(for non symmetric matrix)! Average time = " << time << "ms." << std::endl;
			double err = 0.0;
			for (unsigned int i = 0; i < x_delfem.size(); i++)
			{
				err += 100.0*abs((x_delfem[i] - x[i]) / x[i]);

			}
			err /= (double)x_delfem.size();
			std::cout << "Error = " << err << std::endl;
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