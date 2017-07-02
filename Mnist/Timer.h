#pragma once

#include <iostream>
#include <chrono>

class Timer
{

private:
	std::chrono::system_clock::time_point  _totalStart;	 // Type can be used in auto
	std::chrono::system_clock::time_point  _start;		 // Type can be used in auto
	bool								   _bPrintEnd;	 // The flag wheather cout in destctor

public:

	Timer(bool printEnd = true)
		: _bPrintEnd(printEnd)
	{
		_totalStart = _start = std::chrono::system_clock::now();
	};

	void start() 
	{
		_start = std::chrono::system_clock::now();
	};

	void stop()
	{
		auto end = std::chrono::system_clock::now();
		double elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count());
		std::cout << "The time between start and stop is " << elapsed << " msec.";
		_start = std::chrono::system_clock::now();
	};

	~Timer()
	{
		if (_bPrintEnd)
		{
			auto end = std::chrono::system_clock::now();
			double elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - _totalStart).count());
			std::cout << "Elastic time from constructer to destruter is " << elapsed << " msec.";
		}
	};
};

