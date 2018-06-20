#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <fstream>
namespace Core
{
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
}
#endif