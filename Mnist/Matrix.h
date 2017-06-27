#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <algorithm>

using namespace std;

struct Matrix
{

	///// Must be private for object oriented approach:
	vector<vector<float>>	_data;
	std::size_t				_row;
	std::size_t				_column;

public:

	//
	// Default Constructor
	//
	Matrix() {};

	//
	// Copy constructor
	//
	Matrix(const Matrix &rhs);

	//
	// Constructor with number of row and column and not initialized
	//
	Matrix(const std::size_t row, const std::size_t column);

	//
	// Constructor with number of row and column and initial value
	//
	Matrix(const std::size_t row, const std::size_t column, const float epsilon);

	//
	// Constructor with number of row and column and initial value of by one dim.
	//
	Matrix(const std::vector<float> & vec);

	//
	// Constructor with number of row and column and initial value of by two dim.
	//
	Matrix(const std::vector<std::vector<float>> & mat);

	//
	// Getter and Setter, in this code, you can access member value without
	// getter and setter because all menmbers are public
	//
	float getData(std::size_t i, std::size_t j) const { return _data[i][j]; };
	void setData(std::size_t i, std::size_t j, float value) { _data[i][j]=value; };
	std::size_t getRow()const { return _row; };
	std::size_t getColumn()const { return _column; };

	//
	// Set up matrix with number of row and column and not initialized
	//
	void setup(const size_t row, const size_t column);

	//
	// Set up matrix with number of row and column and initial value
	//
	void setup(const size_t row, const size_t column, const float epsilon);

	//
	// Return row and column as tuple
	//
	std::tuple<size_t, size_t> shape() const;

	//
	// Calculate summation of all value in data
	//
	float sum() const;

	//
	// Calculate summation of all value in data
	//
	float max() const;

	//
	// Calculate sigmoid
	//
	void sigmoid();

	//
	// Calculate relu
	//
	void relu();

	//
	// Slice rows of Matrix
	//
	Matrix slice(size_t startOfRow, size_t numOfrows) const;

	//
	// Return the index of max value for each row as vector
	//
	std::vector<uint16_t> maxIndex() const;

	// *****************************
	// static functions
	// *****************************

	//
	// Return Transverse matrix 
	//
	static Matrix trans(const Matrix & matrix);

	//
	// Return inner product of two matrixes
	//
	static Matrix dot(const Matrix & lhs, const Matrix & rhs);

	//
	// Calculate sigmoid
	//
	static Matrix sigmoid(const Matrix & matrix);

	//
	// Calculate relu
	//
	static Matrix relu(const Matrix & matrix)
	{
		Matrix result(matrix._row, matrix._column);
		for (size_t i = 0; i < matrix._row; ++i)
		{
			for (size_t j = 0; j < matrix._column; ++j)
			{
				result._data[i][j] = std::max(0.0f, matrix._data[i][j]);
			}
		}
		return result;
	}

	//
	// Calculate softmax of all value in data
	//
	static Matrix softmax(const Matrix & matrix);
};


//
// Output contents of tuple of shape (row and column)
//
std::ostream& operator<<(std::ostream& os, const std::tuple<size_t, size_t> & shape);

//
// Output contents of matrix
/*
	np.array ([
	  [0.779659, -0.755285, 0.225202, 0.326331],
	  [0.129933, -0.653668, -0.540027, 0.449548],
	])
*/
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
