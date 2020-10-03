#pragma once
#include "DataPoint.h"
#include <vector>

#include "KeyCell.h"


class DataSet
{
public:
	int kDataPointNum;
	std::vector<DataPoint> data_points;

	const static int kWidth = 32768; // 2^15
	const static int kMaxLayer = 6;

	DataSet(int num = 50);
	void skyline_points(std::vector<DataPoint>& points, std::vector<DataPoint>& result) const;
	std::vector<DataPoint> skyline_serial();
	std::vector<DataPoint> skyline_parallel();

	template <class T>
	friend int shrink_parallel2(const std::vector<KeyCell<2>>& kc_a, std::vector<KeyCell<2>>& kc_b, int ce_max, T& t);

private:
	bool t1[2][2]{}, t2[4][4]{}, t3[8][8]{}, t4[16][16]{}, t5[32][32]{}, t6[64][64]{};
	// , t7[128][128], t8[256][256], t9[512][512], t10[1024][1024];
	int p6[64][64][2]{};

	void init_data_points();
	void prepare_cells();
	void sort_data_points(int layer);

	template <class T1, class T2>
	static void fill_empty_cells(T1& t1, T2& t2, int max);

	template <class T>
	static void shrink_candidates_serial(const std::vector<KeyCell<2>>& kc_a, std::vector<KeyCell<2>>& kc_b, int ce_max,
	                                     T& t);

	template <class T>
	void refine(std::vector<KeyCell<2>>& kc, std::vector<DataPoint>& skyline, int ce_max, T& t);
};
