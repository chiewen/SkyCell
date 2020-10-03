#pragma once
#include <chrono>

struct Timer
{
	static void start1();
	static void stop1();
	static void resume1();
	static double time1();

	static void start2();
	static void stop2();
	static void resume2();
	static double time2();

	static int N1;
private:
	static std::chrono::time_point<std::chrono::system_clock> t1;
	static std::chrono::time_point<std::chrono::system_clock> t2;
	static double d1;
	static double d2;
};

