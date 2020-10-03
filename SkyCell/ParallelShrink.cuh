#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "KeyCell.h"
#include "DataSet3.h"
// #include "DataSet2.h"
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>

#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "Cell.cuh"
#include "Utils.cuh"

void testParallel2();

struct ParallelShrinker
{
	// std::vector<Cell<2>> shrink_parallel2(DataSet2& ds_set2);
	std::vector<Cell<3>> shrink_parallel3(DataSet3& ds_set3);

	template <class T, int D>
	std::vector<Cell<D>> expand_cells2(std::vector<Cell<D>> cells, T& t);

	template <class T, int D>
	std::vector<Cell<D>> expand_cells3(std::vector<Cell<D>> cells, T& t);

	template <int D>
	std::vector<Cell<D>> process(std::vector<Cell<D>> cells) const;
};

template <class T, int D>
std::vector<Cell<D>> ParallelShrinker::expand_cells2(std::vector<Cell<D>> cells, T& t)
{
	std::vector<Cell<D>> result;
	for (auto & cell : cells)
	{
		result.emplace_back(Cell<2>{ 2 * cell[0], 2 * cell[1]    , t[2 * cell[0]][2 * cell[1]]    });
		result.emplace_back(Cell<2>{ 2 * cell[0], 2 * cell[1]+1  , t[2 * cell[0]][2 * cell[1]+1]   });
		result.emplace_back(Cell<2>{ 2 * cell[0]+1, 2 * cell[1]  , t[2 * cell[0]+1][2 * cell[1]]  });
		result.emplace_back(Cell<2>{ 2 * cell[0]+1, 2 * cell[1]+1, t[2 * cell[0]+1][2 * cell[1]+1]});
	}
	return result;
}


template <class T, int D>
std::vector<Cell<D>> ParallelShrinker::expand_cells3(std::vector<Cell<D>> cells, T& t)
{
	std::vector<Cell<D>> result;
	for (auto & cell : cells)
	{
		result.emplace_back(Cell<3>{2 * cell[0], 2 * cell[1], 2 * cell[2], t[2 * cell[0]][2 * cell[1]][2 * cell[2]]});
		result.emplace_back(Cell<3>{2 * cell[0], 2 * cell[1], 2 * cell[2]+1    , t[1 * cell[0]][2 * cell[1]][2 * cell[2]+1]     });
		result.emplace_back(Cell<3>{2 * cell[0], 2 * cell[1]+1, 2 * cell[2]    , t[1 * cell[0]][2 * cell[1]+1][2 * cell[2]]     });
		result.emplace_back(Cell<3>{2 * cell[0], 2 * cell[1]+1, 2 * cell[2]+1  , t[1 * cell[0]][2 * cell[1]+1][2 * cell[2]+1]   });
		result.emplace_back(Cell<3>{2 * cell[0]+1, 2 * cell[1], 2 * cell[2]    , t[1 * cell[0]+1][2 * cell[1]][2 * cell[2]]     });
		result.emplace_back(Cell<3>{2 * cell[0]+1, 2 * cell[1], 2 * cell[2]+1  , t[1 * cell[0]+1][2 * cell[1]][2 * cell[2]+1]   });
		result.emplace_back(Cell<3>{2 * cell[0]+1, 2 * cell[1]+1, 2 * cell[2]  , t[1 * cell[0]+1][2 * cell[1]+1][2 * cell[2]]   });
		result.emplace_back(Cell<3>{2 * cell[0]+1, 2 * cell[1]+1, 2 * cell[2]+1, t[1 * cell[0]+1][2 * cell[1]+1][2 * cell[2]+1] });
	}

	return result;
}

template <int D>
std::vector<Cell<D>> ParallelShrinker::process(std::vector<Cell<D>> cells) const
{
	thrust::device_vector<Cell<D>> d_cells(cells);
	thrust::device_vector<CellNode<D>> d_nodes(d_cells.size());
	thrust::device_vector<CellNode<D>> d_result_nodes(d_cells.size());
	thrust::transform(d_cells.begin(), d_cells.end(), d_nodes.begin(), [] __host__ __device__ (Cell<D>& cell)
	{
		return CellNode<D>(cell);
	});

	// std::vector<CellNode<D>> h_cells(cells.size());
	for (int i = 0; i < D - 1; ++i)
	{
		thrust::sort(d_nodes.begin(), d_nodes.end(), CellPermutation<D>((i + 2) % D));
		// thrust::copy(d_nodes.begin(), d_nodes.end(), h_cells.begin());
		thrust::inclusive_scan(d_nodes.begin(), d_nodes.end(), d_nodes.begin(), CellComparer<D>(i));
		// thrust::copy(d_nodes.begin(), d_nodes.end(), h_cells.begin());
		thrust::inclusive_scan(d_nodes.begin(), d_nodes.end(), d_nodes.begin(), CellComparer2<D>());
		// thrust::copy(d_nodes.begin(), d_nodes.end(), h_cells.begin());
		thrust::inclusive_scan(d_nodes.begin(), d_nodes.end(), d_nodes.begin(), CellComparer3<D>());
		// thrust::copy(d_nodes.begin(), d_nodes.end(), h_cells.begin());
		thrust::transform(d_nodes.begin(), d_nodes.end(), d_nodes.begin(), Cleaner<D>(i));
		// thrust::copy(d_nodes.begin(), d_nodes.end(), h_cells.begin());
	}
	// thrust::sort(d_nodes.begin(), d_nodes.end(), CellPermutation<D>(0));
	auto e = thrust::copy_if(d_nodes.begin(), d_nodes.end(), d_result_nodes.begin(), Dominater<D>());

	int i = e - d_result_nodes.begin();
	thrust::device_vector<Cell<D>> d_result(e - d_result_nodes.begin());
	
	std::vector<Cell<D>> results(d_result.size());

	thrust::transform(d_result_nodes.begin(), e, d_result.begin(),
	                  [] __host__ __device__ (const CellNode<D>& p)
	                  {
		                  return p.original;
	                  });
	if (i > 0)
		thrust::copy(d_result.begin(), d_result.end(), results.begin());

	return results;
}
