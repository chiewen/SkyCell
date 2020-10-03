#pragma once
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

#include "Cell.cuh"

template <int D>
struct CellNode
{
	Cell<D> original;
	Cell<D> previous_dominater;
	Cell<D> dominater;

	__host__ __device__

	CellNode(Cell<D> o): original(o), previous_dominater(), dominater(o)
	{
	}

	__host__ __device__

	CellNode()
	{
	}
};

// make the dominater of key cells be theirselves
template <int D>
struct CellComparer
{
	int _p;

	CellComparer<D>(int p) : _p(p)
	{
	}

	__host__ __device__

	CellNode<D> operator()(CellNode<D>& ca, CellNode<D>& cb)
	{
		bool is_dom = true;
		for (int i = 0; i < D; ++i)
		{
			if (ca.dominater[i] >= cb.original[i])
			{
				is_dom = false;
				break;
			}
		}
		if (is_dom)
		{
			cb.dominater = ca.dominater;
			return cb;
		}
		
		for (int i = _p + 2; i < D; ++i)
		{
			if (ca.dominater[i] != cb.dominater[i]) return cb;
		}
		if (ca.dominater[_p] <= cb.dominater[_p] && ca.dominater[(_p + 1)%D] <= cb.dominater[(_p + 1)%D] && ca.dominater.isFilled)
		{
			cb.dominater = ca.dominater;
		}
		else if (ca.dominater.isFilled && !cb.dominater.isFilled)
		{
			cb.dominater = ca.dominater;
		}
		return cb;
	}
};

//make the previous_dominater be correct
template <int D>
struct CellComparer2
{
	__host__ __device__

	CellNode<D> operator()(CellNode<D>& ca, CellNode<D>& cb)
	{
		for (int i = 0; i < D; ++i)
		{
			if (cb.original[i] != cb.dominater[i])
				return cb;
		}
		if (ca.dominater.isFilled)
			cb.previous_dominater = ca.dominater;

		return cb;
	}
};

//make dominater be correct (some be changed to their previous dominator)
template <int D>
struct CellComparer3
{
	__host__ __device__

	CellNode<D> operator()(CellNode<D>& ca, CellNode<D>& cb)
	{
		for (int i = 0; i < D; ++i)
		{
			if (cb.dominater[i] != ca.dominater[i])
				return cb;
		}
		cb.previous_dominater = ca.previous_dominater;
		return cb;
	}
};

template <int D>
struct CellPermutation
{
	int _p;

	CellPermutation<D>(int p) : _p(p)
	{
	}

	__host__ __device__

	bool operator()(CellNode<D>& l, CellNode<D>& r)
	{
		for (int i = _p; i < D; ++i)
		{
			if (l.original[i] < r.original[i]) return true;
			if (l.original[i] > r.original[i]) return false;
		}
		for (int i = 0; i < _p; ++i)
		{
			if (l.original[i] < r.original[i]) return true;
			if (l.original[i] > r.original[i]) return false;
		}
		return false;
	}
};

template <int D>
struct Dominater
{
	__host__ __device__

	bool operator()(const CellNode<D>& p)
	{
		if (!p.dominater.isFilled)
			return false;

		for (int i = 0; i < D; ++i)
		{
			if (p.original.indices[i] == p.dominater.indices[i])
			{
				return true;
			}
		}
		return false;
	}
};

template <int D>
struct Cleaner
{
	int _p;
	__host__ __device__
		Cleaner<D>(int p) :_p(p) {}

	__host__ __device__

	CellNode<D> operator()(CellNode<D>& p)
	{
		if (p.previous_dominater.isFilled == false)
			return p;

		for (int i = 0; i < D; ++i)
		{
			if (p.previous_dominater[i] >= p.original[i])
				return p;
		}
		p.dominater = p.previous_dominater;
		return p;
	}
};
