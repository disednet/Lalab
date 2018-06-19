#pragma once

struct MatrixCRS
{
	double* data;
	unsigned int* colInd;
	unsigned int* rowPtr;
	unsigned int  dataSize;
	unsigned int  rowPtrSize;
};


