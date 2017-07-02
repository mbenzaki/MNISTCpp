#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <functional>
#include <algorithm>

#include "OneThreadMatrix.h"

using namespace std;

//
// Copy constructor
//
OneThreadMatrix::OneThreadMatrix(const OneThreadMatrix &rhs)
	:OneThreadMatrix(rhs.getRow(), rhs.getColumn())
{
	for (int i = 0; i < rhs.getRow(); ++i)
	{
		for (int j = 0; j < rhs.getColumn(); ++j)
		{
			_data[i][j] = rhs._data[i][j];
		}
	}

};

//
// Constructor with number of row and column and not initialized
//
OneThreadMatrix::OneThreadMatrix(const int row, const int column)
	:_row(row)
	, _column(column)
{
	_data.resize(row);
	for (int i = 0; i < row; ++i)
	{
		_data[i].resize(column);
	}
};

//
// Constructor with number of row and column and initial value
//
OneThreadMatrix::OneThreadMatrix(const int row, const int column, const double epsilon)
	:_row(row)
	, _column(column)
{
	random_device rnd;

	_data.resize(row);
	for (int i = 0; i < row; ++i)
	{
		_data[i].resize(column);
		for (int j = 0; j < column; ++j)
		{
			_data[i][j] = static_cast<double>((2.0 * rnd() - 0xffffffff) / 0xffffffff * epsilon);
		}
	}
};

//
// Constructor with number of row and column and initial value of by one dim.
//
OneThreadMatrix::OneThreadMatrix(const vector<double> & vec)
	:OneThreadMatrix(1, static_cast<int>(vec.size()))
{
	for (int i = 0; i < _column; ++i)
	{
		_data[0][i] = vec[i];
	}
};

//
// Constructor with number of row and column and initial value of by two dim.
//
OneThreadMatrix::OneThreadMatrix(const vector<vector<double>> & mat)
	:OneThreadMatrix(static_cast<int>(mat.size()), static_cast<int>(mat[0].size()))
{
	for (int i = 0; i < _row; ++i)
	{
		if (mat[i].size() != _column)
		{
			throw("Contructor with vec<vec<double>> initializer, it is NOT matrix");
		}
		for (int j = 0; j < _column; ++j)
		{
			_data[i][j] = mat[i][j];
		}
	}
};

//
// Set up matrix with number of row and column and not initialized
//
void OneThreadMatrix::setup(const int row, const int column)
{
	_row = row;
	_column = column;

	_data.resize(row);
	for (int i = 0; i < row; ++i)
	{
		_data[i].resize(column);
	}
};


//
// Set up matrix with number of row and column and initial value
//
void OneThreadMatrix::setup(const int row, const int column, const double epsilon)
{
	random_device rnd;

	_row = row;
	_column = column;

	_data.resize(row);
	for (int i = 0; i < row; ++i)
	{
		_data[i].resize(column);
		for (int j = 0; j < column; ++j)
		{
			_data[i][j] = static_cast<double>((2.0 * rnd() - 0xffffffff) / 0xffffffff * epsilon);
		}
	}
};

//
// Return row and column as tuple
//
tuple<int, int> OneThreadMatrix::shape() const
{
	return std::make_tuple(_row, _column);
};

//
// Calculate summation of all value in data
//
double OneThreadMatrix::sum() const
{
	double result = 0.0;
	for (int i = 0; i < _row; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			result += _data[i][j];
		}
	}
	return result;
};

//
// Calculate summation of all value in data
//
double OneThreadMatrix::max() const
{
	double result = -FLT_MAX;
	for (int i = 0; i < _row; ++i)
	{
		result = std::max(result, *max_element(_data[i].begin(), _data[i].end()));
	}
	return result;
};

//
// Calculate sigmoid
//
void OneThreadMatrix::sigmoid()
{
	for (int i = 0; i < _row; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			_data[i][j] = static_cast<double>(1.0 / (1.0 + exp(-_data[i][j])));
		}
	}
}

//
// Calculate relu
//
void OneThreadMatrix::relu()
{
	for (int i = 0; i < _row; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			_data[i][j] = std::max(0.0, _data[i][j], std::greater<double>());
		}
	}
}

//
// Slice rows of Matrix
//
OneThreadMatrix OneThreadMatrix::slice(int startOfRow, int numOfRows) const
{
	OneThreadMatrix result(numOfRows, this->getColumn());
	for (int i = 0; i < numOfRows; ++i)
	{
		for (int j = 0; j < _column; ++j)
		{
			result._data[i][j] = this->_data[startOfRow + i][j];
		}
	}
	return result;
}

//
// Return the index of max value for each row as vector
//
std::vector<uint16_t> OneThreadMatrix::maxIndex() const
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
OneThreadMatrix OneThreadMatrix::trans(const OneThreadMatrix & matrix)
{
	OneThreadMatrix result(matrix._column, matrix._row);

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
OneThreadMatrix OneThreadMatrix::dot(const OneThreadMatrix & lhs, const OneThreadMatrix & rhs)
{
	if (lhs._column != rhs._row)
	{
		throw new exception("Row of rhs and coloum of rhs mismatch");
	}

	int inner = lhs._column;
	int row = lhs._row;
	int column = rhs._column;

	// Initialize 0
	OneThreadMatrix result(row, column, 0.0);

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
OneThreadMatrix OneThreadMatrix::sigmoid(const OneThreadMatrix & matrix)
{
	OneThreadMatrix result(matrix._row, matrix._column);
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
OneThreadMatrix OneThreadMatrix::relu(const OneThreadMatrix & matrix)
{
	OneThreadMatrix result(matrix._row, matrix._column);
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
OneThreadMatrix OneThreadMatrix::softmax(const OneThreadMatrix & matrix)
{
	OneThreadMatrix result(matrix._row, matrix._column);
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
std::ostream& operator<<(ostream& os, const OneThreadMatrix& matrix)
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

//
// Row and colomn tuple out put ot output stream
//
std::ostream& operator<<(std::ostream& os, const tuple<int, int> & shape)
{
	int row, column;

	std::tie(row, column) = shape;
	os << "(row=" << row << ", column=" << column << ")";

	return os;
}
