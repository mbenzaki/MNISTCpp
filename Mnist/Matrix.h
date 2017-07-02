#pragma once

#include <iostream>
#include <tuple>
#include <string>

#include "Matrix.h"
using namespace std;

class Matrix
{


public:

	//
	// Default Constructor
	//
	Matrix() {};

	//
	// Default Constructor
	//
//	virtual ~Matrix() {};

	//
	// Copy constructor
	//
	Matrix(const Matrix &rhs)
	{
		throw std::exception("Not implemented");
	};

	//
	// Constructor with number of row and column and not initialized
	//
	Matrix(const int row, const int column)
	{
		throw std::exception("Not implemented");
	};

	//
	// Constructor with number of row and column and initial value
	//
	Matrix(const int row, const int column, const double epsilon)
	{
		throw std::exception("Not implemented");
	};

	//
	// Constructor with number of row and column and initial value of by one dim.
	//
	Matrix(const std::vector<double> & vec)
	{
		throw std::exception("Not implemented");
	};

	//
	// Constructor with number of row and column and initial value of by two dim.
	//
	Matrix(const std::vector<std::vector<double>> & mat)
	{
		throw std::exception("Not implemented");
	};


	//
	// Set up matrix with number of row and column and not initialized
	//
	virtual void setup(const int row, const int column) = 0;

	//
	// Set up matrix with number of row and column and initial value
	//
	virtual void setup(const int row, const int column, const double epsilon) = 0;

	//
	// Return row and column as tuple
	//
	virtual std::tuple<int, int> shape() const = 0;

	//
	// Calculate summation of all value in data
	//
	virtual double sum() const = 0;

	//
	// Calculate sigmoid
	//
	virtual void sigmoid() = 0;

	//
	// Calculate relu
	//
	virtual void relu() = 0;

	//
	// Slice rows of Matrix
	//

	//
	// Return the index of max value for each row as vector
	//
	virtual std::vector<uint16_t> maxIndex() const = 0;

};

//
// Output contents of tuple of shape (row and column)
//
//  Caution: this function is defiend in OneThreadMatrix.cpp
//
std::ostream& operator<<(std::ostream& os, const tuple<int, int> & shape);