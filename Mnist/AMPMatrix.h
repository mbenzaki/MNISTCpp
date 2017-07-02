#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <string>

//
// The following is Windows only
// https://msdn.microsoft.com/ja-jp/library/hh265136.aspx
// and don;t use Japanse version. It is very difficult to undersntad because machine translation
// Go to English and mouse hover, you can see Japanese version to check
// 
//
// And don't refer to http://www.aerospace.sd.tmu.ac.jp/hydrodynamics/main/colums/CPPEsIndex.html
// It is old infomation
//
#include <amp.h>

#include "Matrix.h"

class AMPMatrix 
{

public:
	///// Must be private for object oriented approach:
	int					_row;
	int					_column;
	concurrency::array<double,2>	_array;

	//
	// Default Constructor
	//
	AMPMatrix();


	//
	// Default Constructor
	//
	virtual ~AMPMatrix() {};

	//
	// Copy constructor
	//
	AMPMatrix(const AMPMatrix &rhs);

	//
	// Constructor with number of row and column and not initialized
	//
	AMPMatrix(const int row, const int column);

	//
	// Constructor with number of row and column and initial value
	//
	AMPMatrix(const int row, const int column, const double epsilon);

	//
	// Constructor with number of row and column and initial value of by two dim.
	//
	AMPMatrix(const int row, const int column, const std::vector<double> & mat);

	//
	// Getter and Setter, in this code, you can access member value without
	// getter and setter because all menmbers are public
	//
	double getData(int i, int j) const { return _array[i][j]; };
	void setData(int i, int j, double value) { _array[i][j] = value; };
	int getRow()const { return _row; };
	int getColumn()const { return _column; };

	//
	// Return row and column as tuple
	//
	std::tuple<int, int> shape() const;

	//
	// Calculate summation of all value in data
	//
	double sum() const;

	//
	// Calculate maximum value in data
	//
	double maximum() const;

	//
	// Slice rows of Matrix
	//
	AMPMatrix slice(int startOfRow, int numOfrows) const;

	//
	// Return the index of max value for each row as vector
	//
	concurrency::array<uint16_t, 1> AMPMatrix::maxIndex() const;

	// *****************************
	// static functions
	// *****************************

	//
	// Return Transverse matrix 
	//
	static AMPMatrix trans(const AMPMatrix & matrix);

	//
	// Return inner product of two matrixes
	//
	static AMPMatrix dot(const AMPMatrix & lhs, const AMPMatrix & rhs);

	//
	// Calculate sigmoid
	//
	static AMPMatrix sigmoid(const AMPMatrix & matrix);

	//
	// Calculate relu
	//
	static AMPMatrix relu(const AMPMatrix & matrix);

	//
	// Calculate softmax of all value in data
	//
	static AMPMatrix softmax(const AMPMatrix & matrix);

};

