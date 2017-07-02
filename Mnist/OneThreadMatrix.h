#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <string>

#include "Matrix.h"

struct OneThreadMatrix: public Matrix
{

public:
	///// Must be private for object oriented approach:
	std::vector<vector<double>>	_data;
	int				_row;
	int				_column;

public:

	//
	// Default Constructor
	//
	OneThreadMatrix() {};

	//
	// Copy constructor
	//
	OneThreadMatrix(const OneThreadMatrix &rhs);

	//
	// Constructor with number of row and column and not initialized
	//
	OneThreadMatrix(const int row, const int column);

	//
	// Constructor with number of row and column and initial value
	//
	OneThreadMatrix(const int row, const int column, const double epsilon);

	//
	// Constructor with number of row and column and initial value of by one dim.
	//
	OneThreadMatrix(const std::vector<double> & vec);

	//
	// Constructor with number of row and column and initial value of by two dim.
	//
	OneThreadMatrix(const std::vector<std::vector<double>> & mat);

	//
	// Getter and Setter, in this code, you can access member value without
	// getter and setter because all menmbers are public
	//
	double getData(int i, int j) const { return _data[i][j]; };
	void setData(int i, int j, double value) { _data[i][j]=value; };
	int getRow()const { return _row; };
	int getColumn()const { return _column; };

	//
	// Set up matrix with number of row and column and not initialized
	//
	void setup(const int row, const int column);

	//
	// Set up matrix with number of row and column and initial value
	//
	void setup(const int row, const int column, const double epsilon);

	//
	// Return row and column as tuple
	//
	std::tuple<int, int> shape() const;

	//
	// Calculate summation of all value in data
	//
	double sum() const;

	//
	// Calculate summation of all value in data
	//
	double max() const;

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
	OneThreadMatrix slice(int startOfRow, int numOfrows) const;

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
	static OneThreadMatrix trans(const OneThreadMatrix & matrix);

	//
	// Return inner product of two matrixes
	//
	static OneThreadMatrix dot(const OneThreadMatrix & lhs, const OneThreadMatrix & rhs);

	//
	// Calculate sigmoid
	//
	static OneThreadMatrix sigmoid(const OneThreadMatrix & matrix);

	//
	// Calculate relu
	//
	static OneThreadMatrix relu(const OneThreadMatrix & matrix);

	//
	// Calculate softmax of all value in data
	//
	static OneThreadMatrix softmax(const OneThreadMatrix & matrix);
};


//
// Output contents of tuple of shape (row and column)
//
std::ostream& operator<<(std::ostream& os, const std::tuple<int, int> & shape);

//
// Output contents of matrix
/*
	np.array ([
	  [0.779659, -0.755285, 0.225202, 0.326331],
	  [0.129933, -0.653668, -0.540027, 0.449548],
	])
*/
std::ostream& operator<<(std::ostream& os, const OneThreadMatrix& matrix);
