#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <fstream>
#include <vector>
#include <map>
namespace Core
{
	struct MatrixCRS
	{
		double* data;
		unsigned int* colInd;
		unsigned int* rowPtr;
		unsigned int  dataSize;
		unsigned int  rowPtrSize;
		MatrixCRS()
			: data(nullptr)
			, rowPtr(nullptr)
			, colInd(nullptr)
			, dataSize(0)
			, rowPtrSize(0)
		{
		}

		~MatrixCRS()
		{
			if (data != nullptr) delete data;
			if (rowPtr != nullptr) delete rowPtr;
			if (colInd != nullptr) delete colInd;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	struct MatrixCCS
	{
		double* data;
		unsigned int* colPtr;
		unsigned int* rowInd;
		unsigned int  dataSize;
		unsigned int  colPtrSize;

		MatrixCCS()
			: data(nullptr)
			, colPtr(nullptr)
			, rowInd(nullptr)
			, dataSize(0)
			, colPtrSize(0)
		{
		}

		~MatrixCCS()
		{
			if (data != nullptr) delete data;
			if (colPtr != nullptr) delete colPtr;
			if (rowInd != nullptr) delete rowInd;
		}
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

	//////////////////////////////////////////////////////////////////////////
	static void ConverteCRS2CCS(const MatrixCRS& src, MatrixCCS& dst)
	{
		std::vector<std::map<int, double>> tmp_matrix;
		if (dst.data != nullptr)
		{
			delete dst.data;
			delete dst.colPtr;
			delete dst.rowInd;
		}
		dst.data = new double[src.dataSize];
		dst.rowInd = new unsigned int[src.dataSize];
		dst.colPtr = new unsigned int[src.rowPtrSize + 1];
		dst.dataSize = src.dataSize;
		dst.colPtrSize = src.rowPtrSize;
		tmp_matrix.resize(src.rowPtrSize);
		for (unsigned int i = 0; i < src.rowPtrSize; i++)
		{
			for (unsigned int j = src.rowPtr[i]; j < src.rowPtr[i+1]; j++)
			{
				unsigned int colInd = src.colInd[j];
				double data = src.data[j];
				tmp_matrix[colInd][i] = data;
			}
		}
		dst.colPtr[0] = 0;
		unsigned int index = 0;
		for (unsigned int i = 0; i < dst.colPtrSize; i++)
		{
			auto& coloumn = tmp_matrix[i];
			dst.colPtr[i + 1] = dst.colPtr[i] + coloumn.size();
			for (auto it = coloumn.begin(); it != coloumn.end(); ++it)
			{
				dst.data[index] = (*it).second;
				dst.rowInd[index] = (*it).first;
				index++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	static void ConverteCCS2CRS(const MatrixCCS& src, MatrixCRS& dst)
	{
		std::vector<std::map<int, double>> tmp_matrix;
		if (dst.data != nullptr)
		{
			delete dst.data;
			delete dst.rowPtr;
			delete dst.colInd;
		}
		dst.data = new double[src.dataSize];
		dst.colInd = new unsigned int[src.dataSize];
		dst.rowPtr = new unsigned int[src.colPtrSize + 1];
		dst.dataSize = src.dataSize;
		dst.rowPtrSize = src.colPtrSize;
		tmp_matrix.resize(src.colPtrSize);
		for (unsigned int i = 0; i < src.colPtrSize; i++)
		{
			for (unsigned int j = src.colPtr[i]; j < src.colPtr[i + 1]; j++)
			{
				unsigned int rowInd = src.rowInd[j];
				double data = src.data[j];
				tmp_matrix[rowInd][i] = data;
			}
		}
		dst.rowPtr[0] = 0;
		unsigned int index = 0;
		for (unsigned int i = 0; i < dst.rowPtrSize; i++)
		{
			auto& row = tmp_matrix[i];
			dst.rowPtr[i + 1] = dst.rowPtr[i] + row.size();
			for (auto it = row.begin(); it != row.end(); ++it)
			{
				dst.data[index] = (*it).second;
				dst.colInd[index] = (*it).first;
				index++;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////
	static bool IsEqual(const MatrixCRS& mat1, const MatrixCRS& mat2)
	{
		if (mat1.dataSize == mat2.dataSize && mat1.rowPtrSize == mat2.rowPtrSize)
		{
			for (unsigned int i = 0; i < mat1.rowPtrSize; i++)
			{
				if ((mat1.rowPtr[i + 1] - mat1.rowPtr[i]) != (mat2.rowPtr[i + 1] - mat2.rowPtr[i])) return false;
				for (unsigned int j = mat1.rowPtr[i]; j < mat1.rowPtr[i + 1]; j++)
				{
					if (mat1.colInd[j] != mat2.colInd[j]) return false;
					if (std::abs(mat1.data[j] - mat2.data[j]) > std::numeric_limits<double>::epsilon()) return false;
				}
			}
		}
		else
			return false;
		return true;
	}

	//////////////////////////////////////////////////////////////////////////
	static bool IsEqual(const MatrixCCS& mat1, const MatrixCCS& mat2)
	{
		if (mat1.dataSize == mat2.dataSize && mat1.colPtrSize == mat2.colPtrSize)
		{
			for (unsigned int i = 0; i < mat1.colPtrSize; i++)
			{
				if ((mat1.colPtr[i + 1] - mat1.colPtr[i]) != (mat2.colPtr[i + 1] - mat2.colPtr[i])) return false;
				for (unsigned int j = mat1.colPtr[i]; j < mat1.colPtr[i + 1]; j++)
				{
					if (mat1.rowInd[j] != mat2.rowInd[j]) return false;
					if (std::abs(mat1.data[j] - mat2.data[j]) > std::numeric_limits<double>::epsilon()) return false;
				}
			}
		}
		else
			return false;
		return true;
	}
	
	//////////////////////////////////////////////////////////////////////////
	static bool IsEqual(const MatrixCCS& mat1, const MatrixCRS& mat2)
	{
		MatrixCRS tmp;
		ConverteCCS2CRS(mat1, tmp);
		return IsEqual(tmp, mat2);
	}

	//////////////////////////////////////////////////////////////////////////
	static bool IsEqual(const MatrixCRS& mat1, const MatrixCCS& mat2)
	{
		return IsEqual(mat2, mat1);
	}
}
#endif