#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <functional>
#include <algorithm>
#include <array>

#include "Timer.h"
#include "AMPMatrix.h"

using namespace concurrency;

//
// Default Constructor
//
AMPMatrix::AMPMatrix()
	: _row(0)
	, _column(0),
	_array(0)
{}

//
// Copy constructor
//
AMPMatrix::AMPMatrix(const AMPMatrix &rhs)
	: _row(rhs._row)
	, _column(rhs._column)
	, _array(rhs._row, rhs._column, rhs._array)
{};

//
// Constructor with number of row and column and not initialized
//
AMPMatrix::AMPMatrix(const int row, const int column)
	: _row(row)
	, _column(column),
	_array(row, column )
{};

//
// Constructor with number of row and column and initial value
//
static std::vector<double>  makeRandom(int row, int column, double epsilon)
{
	random_device rnd;
	std::vector<double> randoms(row*column);
	for (int i = 0; i < row*column; ++i)
	{
		randoms[i] = ( 2.0 * rnd() - 0xffffffff) / 0xffffffff * epsilon;
	}
	return randoms;
}


AMPMatrix::AMPMatrix(const int row, const int column, const double epsilon)
	: _row(row)
	, _column(column),
	_array(row, column, makeRandom(row, column, epsilon))
{

	// I don't know how to implement random in parallel
	// So I try to do one thread
};

//
// Constructor with array_view
//
AMPMatrix(const concurrency::array_view<> & data)
	:_row( data.extend )


//
// Constructor with number of row and column and initial value of by two dim.
//
AMPMatrix::AMPMatrix(const int row, const int column, const vector<double> & mat)
	:_row( row )
	, _column(column)
	, _array(row, column, mat)
{};


//
// Return row and column as tuple
//
tuple<int, int> AMPMatrix::shape() const
{
	return std::make_tuple(_row, _column);
};

//
// Calculate summation of all value in data
//
double AMPMatrix::sum() const
{
	// It is impossible in parallel
	double result = 0.0;
	for (int i = 0; i < _row; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			result += _array[i][j];
		}
	}
	return result;
};

//
// Calculate summation of all value in data
//
double AMPMatrix::maximum() const
{
	double result = -FLT_MAX;
	for (int i = 0; i < _row; ++i)
	{
		for (int j=0; j < _column; ++j)
		{
			result = _array[i][j] > result ? _array[i][j]: result;
		}
	}
	return result;

};

//
// Slice rows of Matrix
//
AMPMatrix AMPMatrix::slice(int startOfRow, int numOfRows) const
{
/*
	AMPMatrix result(numOfRows, this->getColumn());
	for (int i = 0; i < numOfRows; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			result._data[i][j] = this->_data[startOfRow + i][j];
		}
	}
	return result;
*/
	concurrency::array_view <double, 2>	data(_row, numOfRows);
	parallel_for_each(
		data.extent.tile<2, 2>(),
			[=](tiled_index<2, 2> idx) restrict(amp)
	{
		data[idx.local[1]][idx.local[0]] = _array[idx.global[1] + numOfRows][idx.global[1]];
	});


	return AMPMatrix(numOfRows, this->_column, data);
}

//
// Return the index of max value for each row as vector
//
concurrency::array<uint16_t, 1> AMPMatrix::maxIndex() const
{
	vector<uint16_t> result;
	result.resize(_row);

	for (auto i = 0; i < _row; ++i)
	{
		result[i] = 0;
		auto maxValue = _data[i][0];
		for (auto j = 1; j < _column; ++j)
		{
			if (maxValue < _data[i][j])
			{
				maxValue = _data[i][j];
				result[i] = j;
			}
		}
	}
	return result;
}


//
// *****************************
// static functions
// *****************************
//

//
// Return Transverse matrix 
//
AMPMatrix AMPMatrix::trans(const AMPMatrix & matrix)
{
	AMPMatrix result(matrix._column, matrix._row);

	for (int i = 0; i < matrix._row; ++i)
	{
		for (int j = 0; j < matrix._column; ++j)
		{
			result._data[j][i] = matrix._data[i][j];
		}

	}
	return result;
}

//
// Return inner product of two matrixes
//
AMPMatrix AMPMatrix::dot(const AMPMatrix & lhs, const AMPMatrix & rhs)
{
	if (lhs._column != rhs._row)
	{
		throw new exception("Row of rhs and coloum of rhs mismatch");
	}

	int inner = lhs._column;
	int row = lhs._row;
	int column = rhs._column;

	// Initialize 0
	AMPMatrix result(row, column, 0.0);

	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < column; ++j)
		{
			for (int k = 0; k < inner; ++k)
			{
				result._data[i][j] += lhs._data[i][k] * rhs._data[k][j];
			}
		}
	}

	return result;
}

//
// Calculate sigmoid
//
AMPMatrix AMPMatrix::sigmoid(const AMPMatrix & matrix)
{
	AMPMatrix result(matrix._row, matrix._column);
	for (int i = 0; i < matrix._row; ++i)
	{
		for (int j = 0; j < matrix._column; ++j)
		{
			result._data[i][j] = static_cast<double>(1.0 / (1.0 + exp(-matrix._data[i][j])));
		}
	}
	return result;
}

//
// Calculate relu
//
AMPMatrix AMPMatrix::relu(const AMPMatrix & matrix)
{
	AMPMatrix result(matrix._row, matrix._column);
	for (int i = 0; i < matrix._row; ++i)
	{
		for (int j = 0; j < matrix._column; ++j)
		{
			result._data[i][j] = std::max(0.0, matrix._data[i][j], std::greater<double>());
		}
	}
	return result;
}

//
// Calculate softmax of all value in data
//
AMPMatrix AMPMatrix::softmax(const AMPMatrix & matrix)
{
	AMPMatrix result(matrix._row, matrix._column);
	double fMax = matrix.max();

	for (int i = 0; i < matrix._row; ++i)
	{
		double sumExp = 0.;
		for (int j = 0; j < matrix._column; ++j)
		{
			double exponemt = exp(matrix._data[i][j] - fMax);
			sumExp += exponemt;
			result._data[i][j] = static_cast<double>(exponemt);
		}
		for (int j = 0; j < matrix._column; ++j)
		{
			result._data[i][j] = static_cast<double>(result._data[i][j] / sumExp);
		}

	}

	return result;
};


//
// Output contents of matrix like numpy
//
//	np.array ([
//		[0.779659, -0.755285, 0.225202, 0.326331],
//		[0.129933, -0.653668, -0.540027, 0.449548],
//	])
//
std::ostream& operator<<(ostream& os, const AMPMatrix& matrix)
{
	tuple<int, int> tpl = matrix.shape();
	os << "np.array ([" << endl;
	for (int i = 0; i < matrix.getRow(); ++i)
	{
		os << "  [";
		for (int j = 0; j < matrix.getColumn(); ++j)
		{
			os << " " << fixed << matrix.getData(i, j) << ",";
		}
		os << " ]," << endl;
	}
	os << "])" << endl;
	os << "  Shape " << matrix.shape() << endl;

	return os;
}
