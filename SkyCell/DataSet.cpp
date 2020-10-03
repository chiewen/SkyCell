#include "DataSet.h"
#include <algorithm>
#include <iostream>


#include "KeyCell.h"

using namespace std;

const int DataSet::kWidth;
const int DataSet::kMaxLayer;

DataSet::DataSet(int num): kDataPointNum(num)
{
	init_data_points();
}

void DataSet::init_data_points()
{
	std::generate_n(back_inserter(data_points), kDataPointNum, []()-> DataPoint
	{
		return {rand() % kWidth, rand() % kWidth};
	});
	data_points.emplace_back(0, 32767);
	data_points.emplace_back(32767, 0);

	fill_n(t1[0], 1 << 2, 0);
	fill_n(t2[0], 1 << 4, 0);
	fill_n(t3[0], 1 << 6, 0);
	fill_n(t4[0], 1 << 8, 0);
	fill_n(t5[0], 1 << 10, 0);
	fill_n(t6[0], 1 << 12, 0);

	fill_n(p6[0][0], 1 << 13, 0);
}

void DataSet::sort_data_points(const int layer)
{
	const auto width = kWidth >> layer;

	sort(data_points.begin(), data_points.end(), [=](const DataPoint& p1, const DataPoint& p2)-> bool
	{
		const auto p1_cx = p1.x / width;
		const auto p1_cy = p1.y / width;

		const auto p2_cx = p2.x / width;
		const auto p2_cy = p2.y / width;

		return (p1_cy < p2_cy || (p1_cy == p2_cy && p1_cx < p2_cx))
			|| (p1_cx == p2_cx && p1_cy == p2_cy && (p1.y < p2.y
				|| (p1.y == p2.y && p1.x < p2.x)));
	});
}

template <class T1, class T2>
void DataSet::fill_empty_cells(T1& t1, T2& t2, int max)
{
	for (int i = 0; i < max; ++i)
	{
		for (int j = 0; j < max; ++j)
		{
			if (t1[i][j]) t2[i / 2][j / 2] = true;
		}
	}
}

void DataSet::prepare_cells()
{
	const auto width = kWidth >> kMaxLayer;

	// layer 6 and points 6
	int i = 0;
	for (auto& p : data_points)
	{
		t6[p.y / width][p.x / width] = true;
		if (p6[p.y / width][p.x / width][0] == 0)
		{
			p6[p.y / width][p.x / width][0] = i;
		}
		p6[p.y / width][p.x / width][1] = i + 1;

		i++;
	}

	fill_empty_cells(t6, t5, 64);
	fill_empty_cells(t5, t4, 32);
	fill_empty_cells(t4, t3, 16);
	fill_empty_cells(t3, t2, 8);
	fill_empty_cells(t2, t1, 4);
}

template <class T>
void DataSet::shrink_candidates_serial(const vector<KeyCell<2>>& kc_a, vector<KeyCell<2>>& kc_b, int ce_max, T& t)
{
	Iterator<1> iter{0};
	int cs = kc_a[0].get_last();
	int ce = ce_max;

	for (unsigned short k = 1; k < kc_a.size(); k++)
	{
		auto& key_cell = kc_a[k];
		Iterator<1> iter_next = key_cell.get_I();
		for (unsigned short i = iter[0] * 2; i < iter_next[0] * 2; ++i)
		{
			for (unsigned short j = cs; j < ce; ++j)
			{
				if (t[i][j])
				{
					kc_b.push_back(KeyCell<2>{i, j});
					ce = j;
					break;
				}
			}
		}
		iter = iter_next;
		cs = key_cell.get_last() * 2;
	}
	for (unsigned short i = iter[0] * 2; i < ce_max; ++i)
	{
		for (unsigned short j = cs; j < ce; ++j)
		{
			if (t[i][j])
			{
				kc_b.push_back(KeyCell<2>{i, j});
				ce = j;
				break;
			}
		}
	}
}

template <class T>
void DataSet::refine(vector<KeyCell<2>>& kc, vector<DataPoint>& skyline, int ce_max, T& t)
{
	Iterator<1> iter{0};
	int cs = kc[0].get_last();
	int ce = ce_max;

	for (unsigned short k = 1; k < kc.size(); k++)
	{
		vector<DataPoint> points;
		auto& key_cell = kc[k];
		Iterator<1> iter_next = key_cell.get_I();

		for (unsigned short i = iter[0]; i < iter_next[0]; ++i)
		{
			if (i > iter[0] + 1)
				ce = cs + 2;

			for (unsigned short j = cs; j < ce; ++j)
			{
				auto p = t[i][j];
				for (int l = p[0]; l < p[1]; ++l)
				{
					points.push_back(data_points[l]);
				}
			}
		}
		iter = iter_next;
		ce = cs;
		cs = key_cell.get_last();

		skyline_points(points, skyline);
	}
	vector<DataPoint> points;
	for (int i = kc[kc.size() - 1].get_I()[0]; i < ce_max; ++i)
	{
		auto p = t[i][0];
		for (int j = p[0]; j < p[1]; ++j)
		{
			points.push_back(data_points[j]);
		}
	}
	skyline_points(points, skyline);
}

vector<DataPoint> DataSet::skyline_serial()
{
	vector<DataPoint> skyline;

	sort_data_points(kMaxLayer);

	prepare_cells();

	vector<KeyCell<2>> kc_0{{0, 0}}, kc_1, kc_2, kc_3, kc_4, kc_5, kc_6;

	shrink_candidates_serial(kc_0, kc_1, 2, t1);
	shrink_candidates_serial(kc_1, kc_2, 4, t2);
	shrink_candidates_serial(kc_2, kc_3, 8, t3);
	shrink_candidates_serial(kc_3, kc_4, 16, t4);
	shrink_candidates_serial(kc_4, kc_5, 32, t5);
	shrink_candidates_serial(kc_5, kc_6, 64, t6);

	refine(kc_6, skyline, 64, p6);

	return skyline;
}

std::vector<DataPoint> DataSet::skyline_parallel()
{
	vector<DataPoint> skyline;

	sort_data_points(kMaxLayer);

	prepare_cells();

	vector<KeyCell<2>> kc_0{{0, 0}}, kc_1, kc_2, kc_3, kc_4, kc_5, kc_6;

	shrink_candidates_serial(kc_0, kc_1, 2, t1);
	shrink_candidates_serial(kc_1, kc_2, 4, t2);
	shrink_candidates_serial(kc_2, kc_3, 8, t3);
	shrink_candidates_serial(kc_3, kc_4, 16, t4);
	shrink_candidates_serial(kc_4, kc_5, 32, t5);
	shrink_candidates_serial(kc_5, kc_6, 64, t6);

	refine(kc_6, skyline, 64, p6);

	return skyline;
}

void DataSet::skyline_points(vector<DataPoint>& points, vector<DataPoint>& result) const
{
	sort(points.begin(), points.end(), [](const DataPoint& p1, const DataPoint& p2)-> bool
	{
		return p1.x + p1.y < p2.x + p2.y;
	});

	vector<DataPoint> skyline;
	for_each(points.begin(), points.end(), [&skyline](DataPoint& point)-> void
	{
		if (!count_if(skyline.begin(), skyline.end(), [&point](DataPoint& p_s) -> bool
		{
			return p_s.x <= point.x && p_s.y <= point.y;
		}))
		{
			skyline.push_back(point);
		}
	});
	result.insert(result.end(), skyline.begin(), skyline.end());
}
