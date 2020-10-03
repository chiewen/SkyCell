#include "Timer.h"

std::chrono::time_point<std::chrono::system_clock> Timer::t1;
std::chrono::time_point<std::chrono::system_clock> Timer::t2;
double Timer::d1;
double Timer::d2;
int Timer::N1;

void Timer::start1()
{
	d1 = 0;
	t1 = std::chrono::system_clock::now();
}

void Timer::stop1()
{
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - t1);
	d1 += double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
}

void Timer::resume1()
{
	t1 = std::chrono::system_clock::now();
}

double Timer::time1()
{
	return d1;
}

void Timer::start2()
{
	d2 = 0;
	t2 = std::chrono::system_clock::now();
}

void Timer::stop2()
{
	auto end = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - t2);
	d2 += double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
}

void Timer::resume2()
{
	t2 = std::chrono::system_clock::now();
}

double Timer::time2()
{
	return d2;
}
