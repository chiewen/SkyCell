#include <iostream>
#include <iterator>
#include <algorithm>

#include "DataSet.h"
#include "DataSet3.h"
#include "shrink_parallel.cuh"
#include <chrono>  

#include <thrust/version.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

using namespace std::chrono;
using namespace std;

void skyline_2D(int i)
{
	DataSet ds(i);
	DataSet ds1 (ds);

	std::vector<DataPoint> skyline;
	ds1.skyline_points(ds1.data_points, skyline);

	std::cout << skyline.size() << ":";
	copy(skyline.begin(), skyline.end(), std::ostream_iterator<DataPoint>(std::cout, " "));

	cout << endl << "==========================" << endl;

	skyline = ds.skyline_serial();

	sort(skyline.begin(), skyline.end(), [](const DataPoint& p1, const DataPoint& p2)-> bool
	{
		return p1.x + p1.y < p2.x + p2.y;
	});
	cout << skyline.size() << ":";
	copy(skyline.begin(), skyline.end(), ostream_iterator<DataPoint>(cout, " "));

	cout << endl << "==========================";
}

void print_skyline(vector<DataPointD<3>>& skyline)
{
	sort(skyline.begin(), skyline.end(), [](const DataPoint3& p1, const DataPoint3& p2)-> bool
	{
		return p1[0] < p2[0] || (p1[0] == p2[0] && p1[1] < p2[1]) ||
			(p1[0] == p2[0] && p1[1] == p2[1] && p1[2] < p2[2]);
	});

	std::cout << "found:" << skyline.size() << "    \t";
	// copy(skyline.begin(), skyline.end(), std::ostream_iterator<DataPoint3>(std::cout, " "));
	
}

void exp_ds3(int point_num, int layer)
{
		srand(3);
		DataSet3 ds3(point_num, layer);
		vector<DataPoint3> skyline;
		auto start = system_clock::now();
		ds3.skyline_points(*ds3.data_points, skyline);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		print_skyline(skyline);
		cout << "real >> points: " << point_num << "    \tlayer: " << layer << "    \ttime: "
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< endl;
}
void exp_ds3_serial(int point_num, int layer)
{
		srand(3);
		DataSet3 ds3(point_num, layer);
		auto start = system_clock::now();
		auto skyline = ds3.skyline_serial();
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		print_skyline(skyline);
		cout << "serial >> points: " << point_num << "    \tlayer: " << layer << "    \ttime: "
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< endl;
}
void exp_ds3_parallel(int point_num, int layer)
{
		srand(3);
		DataSet3 ds3(point_num, layer);
		// auto start = system_clock::now();
		auto skyline = ds3.skyline_parallel();
		// auto end = system_clock::now();
		// auto duration = duration_cast<microseconds>(end - start);
		print_skyline(skyline);
		cout << "parallel >> points: " << point_num << "    \tlayer: " << layer << "    \ttime: "
			<< Timer::time1() + Timer::time2() << "\t" << Timer::time1() << "\t" << Timer::time2()
			<< "\t" << Timer::N1 << endl;
}
int main()
{
	// vector<int> times{5, 50, 500, 5000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000, 4000000};
	vector<int> times{5000, 10000, 50000};

	// for (auto t : times) {
	// 	auto start = system_clock::now();
	// 	 skyline_2D(t);
	// 	auto end = system_clock::now();
	// 	auto duration = duration_cast<microseconds>(end - start);
	// 	cout << "处理" << t << "个2D点花费了"
	// 		<< double(duration.count()) * microseconds::period::num / microseconds::period::den
	// 		<< "秒" << endl;
	// }	
	//
	// for (auto t : times) {
	// 	auto start = system_clock::now();
	// 	DataSet3 ds3(t);
	// 	DataSet3 ds3_1 = ds3;
	// 	vector<DataPoint3> skyline;
	// 	ds3.skyline_points(*ds3.data_points, skyline);
	// 	auto skyline_1 = ds3_1.skyline_serial();
	// 	print_skyline(skyline);
	// 	print_skyline(skyline_1);
	// 	auto end = system_clock::now();
	// 	auto duration = duration_cast<microseconds>(end - start);
	// 	cout << "处理" << t << "个3D点花费了"
	// 		<< double(duration.count()) * microseconds::period::num / microseconds::period::den
	// 		<< "秒" << endl;
	// }	
	//
	exp_ds3_parallel(1000, 2);
	cout << "*********" << endl;
	for (auto t : times) {
		cout << "num: " << t << endl;
		exp_ds3(t, 7);
		exp_ds3_serial(t, 2);
		exp_ds3_serial(t, 3);
		exp_ds3_serial(t, 4);
		exp_ds3_serial(t, 5);
		exp_ds3_serial(t, 6);
		exp_ds3_serial(t, 7);
		exp_ds3_parallel(t, 2);
		exp_ds3_parallel(t, 3);
		exp_ds3_parallel(t, 4);
		exp_ds3_parallel(t, 5);
		exp_ds3_parallel(t, 6);
		exp_ds3_parallel(t, 7);
		cout << endl << "==========================" << endl << flush;
	}	

	// testParallel2();
	return 0;
}

