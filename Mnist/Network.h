#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "Matrix.h"

class Network
{

protected:
	Matrix	_x;		// Input Layer Matrix (_numItems x (MnistImageSize))
	Matrix	_y;		// Estimated Output Layer 

public:

	Network()
		: _setup(false)
	{
	}

	virtual ~Network()
	{
	}

	//
	// Setup Network
	//
	virtual void setup() = 0;

	//
	// Forward Propagation
	//
	virtual void forward(size_t batchSize = 1)
	{
		throw exception("You should implemt in your sub class");
	};

	//
	// Backward Propagation
	//
	virtual void backword()
	{
		throw exception("You should implemt in your sub class");
	};

protected:
	bool	_setup;
};


//
// MINST implementation layers and two hidden layers and output layer
//
class ThreeLayerNeuralNetwork : public Network
{

protected:
	static const size_t	MninstImageSize = (28 * 28);

	size_t _numOfItems;	// Total number of Supervised data
	Matrix _t;	// Supervised data from MNIST

	Matrix _w1;	// Weight from input layer to hidden layer(a1)		shape(28x28, 50)
	Matrix _w2;	// Weight from hidden layer(a1) to hidden layer(a2) shape(50,100)
	Matrix _w3;	// Weight from hidden layer(a2) output layer		shape(100,10)

	size_t	_numOfTotalCalculated;
	size_t	_numOfaccuracy;


	static int32_t readInt32(istream & is);
	void loadMNISTData();
	virtual Matrix forwardMiniBatch(const Matrix & subX, bool bSoftmax = false);


public :
	ThreeLayerNeuralNetwork() {};
	virtual ~ThreeLayerNeuralNetwork() {};
	virtual void setup();
	virtual void forward(size_t batchSize = 1);

};
