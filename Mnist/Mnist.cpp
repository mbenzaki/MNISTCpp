#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <algorithm>
#include <memory>

#include "OneThreadMatrix.h"
#include "Network.h"

using namespace std;

void testConstructor()
{
	cout << "****************Test " << __func__ << "****************" << endl;

#ifdef _ONETHREAD

	OneThreadMatrix dc1(5, 10);
	cout << "w = " << dc1 << endl;

	OneThreadMatrix dc2(10, 10);
	cout << "w = " << dc2 << endl;

	OneThreadMatrix  w(3, 4, 1.0);
	cout << "w = " << w;
	cout << "np.max(w) is " << w.max() << endl;
	cout << "np.sum(w) is " << w.sum() << endl;
	cout << "np.shape(w) is " << w.shape() << endl;
	cout << endl;

	OneThreadMatrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});
	cout << "a = " << a;
	cout << "np.max(a) is " << a.max() << endl;
	cout << "np.sum(a) is " << a.sum() << endl;
	cout << endl;

	OneThreadMatrix b({
		{ -1, -2, -3 },
		{ -4, -5, -6 }
	});
	cout << "b = " << b;
	cout << "np.max(b) is " << b.max() << endl;
	cout << "np.sum(b) is " << b.sum() << endl;
	cout << endl;

	vector<vector<double>> cdata = {
		{ 1.0 },
		{ 2.0 },
		{ 3.0 },
	};
	OneThreadMatrix c(cdata);
	cout << "c = " << c;
	cout << "np.max(c) is " << c.max() << endl;
	cout << "np.sum(c) is " << c.sum() << endl;
	cout << endl;
#else
#endif
}

void testDot()
{
	cout << "****************Test " << __func__ << "****************" << endl;;

#ifdef _ONETHREAD

	OneThreadMatrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});
	cout << "a = " << a << endl;

	OneThreadMatrix t = OneThreadMatrix::trans(a);
	cout << "a.T is " << t << endl;
		
	cout << "np.dot(a,a.T) is " << OneThreadMatrix::dot(a, OneThreadMatrix::trans(a));
	cout << endl;
#else
#endif
}

void testSigmoid()
{
	cout << "****************Test " << __func__ << "****************" << endl;;

#ifdef _ONETHREAD
	vector<double> data = { -1,0, 1.0, 2.0 };
	OneThreadMatrix a(data);
	cout << "a = " << a << endl;

	OneThreadMatrix sig = OneThreadMatrix::sigmoid(a);
	cout << "Class Method = " << sig << endl;

	a.sigmoid();
	cout << "Member Method is " << a << endl;
	cout << endl;
#else
#endif
}

void testMaxIndex()
{
	cout << "****************Test " << __func__ << "****************" << endl;;

#ifdef _ONETHREAD
	OneThreadMatrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});

	auto vec = a.maxIndex();
	cout << "a = " << a ;

	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

	OneThreadMatrix b(3, 20, 10.0);
	vec = b.maxIndex();
	cout << "b = " << b;
	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

	OneThreadMatrix c(10, 5, 30.0);
	vec = c.maxIndex();
	cout << "c = " << c;
	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

#else
#endif
}

void testSoftmax()
{
	cout << "****************Test " << __func__ << "****************" << endl;;

#ifdef _ONETHREAD

	vector<double> data = { 0.3, 2.9, 4.0 };
	OneThreadMatrix a(data);
	cout << "a = " << a << endl;

	OneThreadMatrix softmax = OneThreadMatrix::softmax(a);
	cout << "softmax of a = " << softmax << endl;

	cout << endl;
#else
#endif
}

int main()
{

	try
	{

#if 0
		auto_ptr<Network>	network1(new ThreeLayerNeuralNetwork());
		network1->setup();
		network1->forward(100); // mini batch size is 100
#else
		// Test code
		testConstructor();
		testDot();
		testSigmoid();
		testMaxIndex();
		testSoftmax();
#endif
	}
	catch (exception & e)
	{
		cerr << "[ERR] Exception occured " << e.what() << endl;
	}

	cout << "Wait to terminate: Push enter key>";
	getchar();
	return 0;
}

