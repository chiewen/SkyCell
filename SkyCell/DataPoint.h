#pragma once
#include <ostream>

class DataPoint
{
public:
	int x;
	int y;

	DataPoint(int x, int y);

	friend std::ostream& operator<<(std::ostream& os, const DataPoint& p)
	{
		return os << "<" << p.x << "," << p.y << ">";
	}
};

template <int D>
class DataPointD
{
public:
	short indices[D];

	short& operator[](int i) { return indices[i]; }
	short operator[](int i) const { return indices[i]; }

	DataPointD(std::initializer_list<short> il);

	friend std::ostream& operator<<(std::ostream& os, const DataPointD& p)
	{
		os << "<";
		for (auto& index : p.indices)
		{
			os << index << ",";
		}
		os << ">";
		return os;
	}
};

template <int D>
DataPointD<D>::DataPointD(std::initializer_list<short> il)
{
	std::copy(il.begin(), il.end(), std::begin(indices));
}

using DataPoint3 = DataPointD<3>;

using DataPoint2 = DataPointD<2>;
