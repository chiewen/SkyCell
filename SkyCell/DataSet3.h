#pragma once
#include <map>
#include <vector>


#include "Cell.cuh"
#include "DataPoint.h"
#include "KeyCell.h"
#include "Timer.h"

class DataSet3
{
public:
	int kDataPointNum;
	std::shared_ptr<std::vector<DataPoint3>> data_points;

	const static int kWidth = 32768; // 2^15

	const static int kMaxLayer = 7;
	int layer;

	DataSet3(int num = 5, int layer = 7);
	void skyline_points(std::vector<DataPoint3>& points, std::vector<DataPoint3>& result) const;
	std::vector<DataPoint3> skyline_serial();
	std::vector<DataPoint3> skyline_parallel();

	std::shared_ptr<bool[2][2][2]> pt1;
	std::shared_ptr<bool[4][4][4]> pt2;
	std::shared_ptr<bool[8][8][8]> pt3;
	std::shared_ptr<bool[16][16][16]> pt4;
	std::shared_ptr<bool[32][32][32]> pt5;
	std::shared_ptr<bool[64][64][64]> pt6;
	std::shared_ptr<bool[128][128][128]> pt7;

	std::shared_ptr<int[128][128][128][2]> pp;

private:
	void init_data_points();
	void prepare_cells();
	void sort_data_points(int layer);

	template <class T1, class T2>
	static void fill_empty_cells(T1& t1, T2& t2, int max);

	template <class T>
	static void shrink_candidates_serial(const std::vector<KeyCell<3>>& kc_a, std::vector<KeyCell<3>>& kc_b, int ce_max,
	                                     T& t);

	template <class T>
	void refine_cell(std::vector<Cell<3>>& kc, std::vector<DataPoint3>& skyline, T& t);
	template <class T>
	void refine(std::vector<KeyCell<3>>& kc, std::vector<DataPoint3>& skyline, T& t);
};

template <class T1, class T2>
void DataSet3::fill_empty_cells(T1& t1, T2& t2, int max)
{
	for (int i = 0; i < max; ++i)
	{
		for (int j = 0; j < max; ++j)
		{
			for (int k = 0; k < max; ++k)
			{
				if (t1[i][j][k]) t2[i / 2][j / 2][k / 2] = true;
			}
		}
	}
}

template <class T>
void DataSet3::shrink_candidates_serial(const std::vector<KeyCell<3>>& kc_a, std::vector<KeyCell<3>>& kc_b, int ce_max,
	T& t)
{
	const int Dimension = 3;
	Iterator<2> iter{ 0, 0 };
	int cs = kc_a[0].get_last();
	int ce = ce_max;

	std::map<Iterator<2>, int> m_cs;
	std::map<Iterator<2>, int> m_ce;
	m_cs.insert(std::make_pair(iter, cs));
	m_ce.insert(std::make_pair(iter, ce));

	for (int k = 1; k < kc_a.size(); k++)
	{
		auto& key_cell = kc_a[k];
		auto iter_next = key_cell.get_I().next_layer();

		while (iter != iter_next)
		{
			auto fs = m_cs.find(iter);
			if (fs == m_cs.end())
			{
				for (int i = 0; i < Dimension - 1; ++i)
				{
					auto iter2 = iter;
					iter2[i]--;
					auto f = m_cs.find(iter2);
					if (f != m_cs.end() && f->second > cs)
					{
						cs = f->second;
					}
				}
				m_cs.insert(std::make_pair(iter, cs));
			}
			else
			{
				cs = fs->second;
			}

			ce = ce_max;
			for (int i = 0; i < Dimension - 1; ++i)
			{
				auto iter2 = iter;
				iter2[i]--;
				auto f = m_ce.find(iter2);
				if (f != m_ce.end() && f->second < ce)
				{
					ce = f->second;
				}
			}
			for (unsigned short j = cs; j < ce; ++j)
			{
				if (t[iter[0]][iter[1]][j] != 0)
				{
					kc_b.push_back(KeyCell<3>{iter[0], iter[1], j});
					ce = j;
					m_ce.insert(std::make_pair(iter, ce));
					break;
				}
			}
			iter.advance(ce_max);
		}
		cs = key_cell.get_last() * 2;
		m_cs.insert(std::make_pair(iter, cs));
	}
	for (unsigned short i = iter[0]; i < ce_max; ++i)
	{
		for (unsigned short j = iter[1]; j < ce_max; ++j)
		{
			for (unsigned short k = cs; k < ce; ++k)
			{
				if (t[i][j][k] != 0)
				{
					kc_b.push_back(KeyCell<3>{i, j, k});
					ce = k;
					break;
				}
			}
		}
	}
}

template <class T>
void DataSet3::refine_cell(std::vector<Cell<3>>& kc, std::vector<DataPointD<3>>& skyline, T& t)
{
	Iterator<2> iter{0, 0};
	std::vector<DataPoint3> points;
	for (auto& cell : kc)
	{
		int m = pow(2, kMaxLayer - layer);
		for (int i1 = cell[0] * m; i1 < cell[0] * m + m; ++i1)
			for (int i2 = cell[1] * m; i2 < cell[1] * m + m; ++i2)
				for (int i3 = cell[2] * m; i3 < cell[2] * m + m; ++i3) {
					auto p0 = t[i1][i2][i3][0];
					auto p1 = t[i1][i2][i3][1];
					for (int i = p0; i < p1; ++i)
					{
						points.push_back((*data_points)[i]);
					}
				}
	}

	Timer::N1 = points.size();
	Timer::start2();
	skyline_points(points, skyline);
	Timer::stop2();
}


template <class T>
void DataSet3::refine(std::vector<KeyCell<3>>& kc, std::vector<DataPointD<3>>& skyline, T& t)
{
	Iterator<2> iter{0, 0};
	std::vector<DataPoint3> points;
	for (auto& cell : kc)
	{
		// auto p = t[cell[0]][cell[1]][cell[2]];
		// for (int i = p[0]; i < p[1]; ++i)
		// {
		// 	points.push_back((*data_points)[i]);
		// }
		int m = pow(2, kMaxLayer - layer);
		for (int i1 = cell[0] * m; i1 < cell[0] * m + m; ++i1)
			for (int i2 = cell[1] * m; i2 < cell[1] * m + m; ++i2)
				for (int i3 = cell[2] * m; i3 < cell[2] * m + m; ++i3) {
					auto p0 = t[i1][i2][i3][0];
					auto p1 = t[i1][i2][i3][1];
					for (int i = p0; i < p1; ++i)
					{
						points.push_back((*data_points)[i]);
					}
				}
	}
	skyline_points(points, skyline);
}
