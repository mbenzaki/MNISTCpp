#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "OneThreadMatrix.h"
#include "Network.h"


int32_t ThreeLayerNeuralNetwork::readInt32(istream & is)
{

	// MNIST is big/high endian
	int32_t number;
	unsigned char data[4];

	number = 0;
	is.read(reinterpret_cast<char*>(data), 4);
	number += data[0] << 24;
	number += data[1] << 16;
	number += data[2] << 8;
	number += data[3];

	return number;
}


void ThreeLayerNeuralNetwork::loadMNISTData()
{
	// read images of traing set
	const streamsize bufferSize = 0x10000;
	const string imageFileName = "..\\Data\\train-images-idx3-ubyte";
	const string labelFileName = "..\\Data\\train-labels-idx1-ubyte";
	stringstream ss;

	int32_t magic;

	// Read MNIST Training set Image
	{
		const char * fileName = imageFileName.c_str();
		cout << "Start load image file" << fileName << endl;

		// Read MNIST Training Set Image
		ifstream ifs(imageFileName, fstream::in | fstream::binary);
		//	ifs.rdbuf()->pubsetbuf(0, bufferSize);
		if (!ifs)
		{
			ss << "File(" << fileName << ") can not be opened";
			throw exception(ss.str().c_str());
		}

		magic = readInt32(ifs);
		if (magic != 0x0803)
		{
			ss << "File(" << fileName << ") magic number is wrong, it might not MNIST image file";
			throw exception(ss.str().c_str());
		}
		_numOfItems = readInt32(ifs);
		size_t rows = readInt32(ifs);
		size_t columns = readInt32(ifs);
		if ((rows != 28) || (columns != 28))
		{
			ss << "File(" << fileName << ") rows or/and columns is not 28";
			throw exception(ss.str().c_str());
		}

		vector<uint8_t> buffer(MninstImageSize * _numOfItems);
		ifs.read(reinterpret_cast<char*>(&buffer[0]), MninstImageSize*_numOfItems);
		if (!ifs)
		{
			ss << "error: only " << ifs.gcount() << " could be read";
			throw exception(ss.str().c_str());
		}
		ifs.close();

		// resize the matrix of input layer x
		_x.setup(_numOfItems, MninstImageSize);
		for (size_t i = 0; i < _numOfItems; ++i)
		{
			for (size_t j = 0; j < MninstImageSize; ++j)
			{
				// Normarize the input value
				_x._data[i][j] = static_cast<float>(buffer[i*MninstImageSize + j] / 255.0f);
			}
		}
		// End of set up MNIST image file
	}
	// End of Read MNIST Training set Image

	// Read MNIST Training set label
	{
		// Start of set up MINIS label file
		const char * fileName = labelFileName.c_str();
		cout << "Start load label" << fileName << endl;

		// Read MNIST Training Set Label
		ifstream ifs(labelFileName, fstream::in | fstream::binary);

		if (!ifs)
		{
			ss << "File(" << fileName << ") can not be opened";
			throw exception(ss.str().c_str());
		}

		magic = readInt32(ifs);
		if (magic != 0x0801)
		{
			ss << "File(" << fileName << ") magic number is wrong, it might not MNIST label file";
			throw exception(ss.str().c_str());
		}
		size_t numOfItems = readInt32(ifs);
		if (numOfItems != _numOfItems)
		{
			ss << "A number of items is mismatched Image File(" << imageFileName.c_str() << ") Label File("
				<< fileName << ")";
			throw exception(ss.str().c_str());
		}

		vector<uint8_t> buffer(_numOfItems);
		ifs.read(reinterpret_cast<char*>(&buffer[0]), _numOfItems);
		if (!ifs)
		{
			ss << "error: only " << ifs.gcount() << " could be read";
			throw exception(ss.str().c_str());
		}
		ifs.close();

		// resize the matrix of input layer x
		_t.setup(_numOfItems, 10);
		for (size_t i = 0; i < _numOfItems; ++i)
		{

			_t._data[i][buffer[i]] = static_cast<float>(1.0f);
		}
		ifs.close();
	}
	// End of Read MNIST Training set label

	cout << "Start load MNIST" << endl;
}

//
// Setup newtwork
//
void ThreeLayerNeuralNetwork::setup()
{

	//
	// Load MNIST data set(image and label)
	//
	loadMNISTData();

	//
	// Set up weights of input, a1, a2, a3 and output
	//

	// Weight from input layer to hidden layer(a1)		shape(28x28, 50)
	_w1.setup(MninstImageSize, 50, 1.0);

	// Weight from hidden layer(a1) to hidden layer(a2) shape(50,100)
	_w2.setup(50, 100, 1.0);

	// Weight from hidden layer(a2) output layer		shape(100,10)
	_w3.setup(100, 10, 1.0);

	_setup = true;
}

void ThreeLayerNeuralNetwork::forward(size_t batchSize)
{

//
// Initialize  values of accuracy calculation 
//
	_numOfTotalCalculated = 0;
	_numOfaccuracy = 0;

	auto totalRows = _t.getRow();
	int64_t miniBatchSize;

	for (size_t i = 0; ; i += batchSize)
	{
		if (i + batchSize > totalRows)
		{
			miniBatchSize = totalRows - i;
			if (miniBatchSize < 1)	break;
		}
		else
		{
			miniBatchSize = static_cast<size_t>(batchSize);
		}

		auto subX = _x.slice(i, miniBatchSize);
		cout << "i=" << i << ", miniBatchSize=" << miniBatchSize <<
			" shape=" << subX.shape() << endl;
		auto result = forwardMiniBatch(subX, false);

		//
		// Calculate accuracy
		//

		// Slice of teacher data
		auto subT = _t.slice(i, miniBatchSize);
		if (subX.getRow() != subT.getRow())
		{
			throw exception("something wrong 1. in forward");
		}

		auto resultMaxIndex = result.maxIndex();
		auto tMaxIndex = subT.maxIndex();
		if (resultMaxIndex.size() != tMaxIndex.size())
		{
			throw exception("something wrong 2. in forward");
		}

		for (auto i = 0; i < resultMaxIndex.size(); ++i)
		{
			if (resultMaxIndex[i] == tMaxIndex[i])	_numOfaccuracy++;
		}

		_numOfTotalCalculated += resultMaxIndex.size();
		cout << "*** Current Accuracy is " << fixed
			<< _numOfaccuracy << " of " << _numOfTotalCalculated << " "
			<< static_cast<double>(_numOfaccuracy) / _numOfTotalCalculated * 100.0
			<< "%" << endl << endl;

	} // end of 	for (size_t i = 0; ; i += batchSize)
}

//
// Forward Propagation
//  return value is y which is matrix calculated by this neural network
OneThreadMatrix ThreeLayerNeuralNetwork::forwardMiniBatch(const OneThreadMatrix & subX, bool bSoftmax)
{
	OneThreadMatrix y;

	{
		// Ignore batch size, try toi all data
		//   MninstImageSize is 28 x 28 = 784

		// a1's shape is (miniBatchSize, 50)
		OneThreadMatrix a1 = OneThreadMatrix::dot(subX, _w1);
		OneThreadMatrix::sigmoid(a1);

		// a2's shape is (miniBatchSize, 100)
		OneThreadMatrix a2 = OneThreadMatrix::dot(a1, _w2);
		OneThreadMatrix::sigmoid(a2);

		// a3 and y's shape is (miniBatchSize, 100)
		OneThreadMatrix a3 = OneThreadMatrix::dot(a2, _w3);

		if (bSoftmax)
		{
			y = OneThreadMatrix::softmax(a3);
		}
		else
		{
			y = a3;
		}
	}

		return y;
	};
