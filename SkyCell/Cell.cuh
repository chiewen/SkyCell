#ifndef CELL_H
#define CELL_H

#include <cuda_runtime.h>
#include <vector>
#include "KeyCell.h"
#include "DataSet3.h"
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/device_vector.h>

template <int D>
class Cell
{
public:
	int indices[D];

	__host__ __device__
	Cell();
	Cell(std::initializer_list<int> il);
	Cell( std::initializer_list<int> il, bool isf);
	bool isFilled = false;

	__host__ __device__
	int& operator[](int i) { return indices[i]; }

	friend std::ostream& operator<<(std::ostream& os, const Cell& p)
	{
		os << "[" << p.isFilled << ": ";
		std::copy(std::begin(p.indices), std::end(p.indices), std::ostream_iterator<int>(os, " "));
		os << "]";
		return os;
	}

	__host__ __device__

	friend bool operator==(const Cell& kc1, const Cell& kc2)
	{
		return thrust::equal(std::begin(kc1.indices), std::end(kc1.indices), std::begin(kc2.indices));
	}

	__host__ __device__

	friend bool operator!=(const Cell& kc1, const Cell& kc2)
	{
		return !(kc1 == kc2);
	}
};

template <int D>
Cell<D>::Cell()
{
	for (int i = 0; i < D; ++i)
	{
		indices[i] = -1;
	}
}

template <int D>
Cell<D>::Cell(std::initializer_list<int> il)
{
	int i = 0;
	for (auto elem : il)
	{
		indices[i++] = elem;
	}
} 

template <int D>
Cell<D>::Cell( std::initializer_list<int> il, bool isf) : Cell(il)
{
	this->isFilled = isf;
}

#endif
