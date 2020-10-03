//Deprecated

//all functions are moved to parallalShrinker
#include "shrink_parallel.cuh"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

void checkResult(float* hostRef, float* gpuRef, const int N)
{
	double epsilon = 1.0E-8;
	bool match = true;
	for (int i = 0; i < N; i++)
	{
		if (abs(hostRef[i] - gpuRef[i]) > epsilon)
		{
			match = false;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

template <int D>
__device__ Cell<D> Comp(Cell<D>& ca, Cell<D>& cb)
{
	for (int i = 2; i < D; ++i)
	{
		if (ca[i] != cb[i]) return cb;
	}
	if (ca[0] <= cb[0] && ca[1] <= cb[1])
		return ca;
	return cb;
}

template <int D>
__global__ void ProcessBo(int j_max, Cell<D>* bo_low, Cell<D>* bo_high)
{
	const int j = threadIdx.x;
	if (j <= j_max)
		bo_high[j] = Comp(bo_low[2 * j], bo_low[2 * j + 1]);
}

template <int D>
__global__ void ProcessBl(int j_max, Cell<D>* bl_low, Cell<D>* bl_high, Cell<D>* bo)
{
	const int j = threadIdx.x;
	if (j <= j_max)
	{
		if (j == 0)
		{
			bl_low[j] = bo[j];
			return;
		}
		if (j % 2 == 1)
		{
			bl_low[j] = bl_high[(j - 1) / 2];
			return;
		}
		bl_low[j] = Comp(bl_high[j / 2 - 1], bo[j]);
	}
}


int process3(std::vector<Cell<3>>& cells, std::vector<KeyCell<3>>& key_cells)
{
	const int l = log2(cells.size());

	const auto dev = 0;
	cudaSetDevice(dev);

	const int cell_num = cells.size();
	const auto n_bytes = l * cell_num * sizeof(Cell<3>);
	Cell<3>* h_bo = static_cast<Cell<3>*>(malloc(n_bytes));
	Cell<3>* h_bl = static_cast<Cell<3>*>(malloc(n_bytes));

	for (int i = 0; i < cells.size(); ++i)
	{
		h_bo[i] = cells[i];
	}

	Cell<3> *d_bo, *d_bl;
	cudaMalloc(static_cast<Cell<3>**>(&d_bo), n_bytes);
	cudaMalloc(static_cast<Cell<3>**>(&d_bl), n_bytes);
	cudaMemcpy(d_bo, h_bo, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bl, h_bl, n_bytes, cudaMemcpyHostToDevice);

	dim3 block(32 > cell_num ? cell_num : 32);
	dim3 grid((cell_num + block.x - 1) / block.x);

	for (int i = 1; i <= l; ++i)
	{
		std::cout << "pow result:" << pow(2, l - i) << std::endl;

		ProcessBo <<<grid, block>>>(int(pow(2, l - i)), d_bo + (i - 1) * cell_num, d_bo + i * cell_num);
		cudaDeviceSynchronize();
	}
	for (int i = l; i >= 0; i--)
	{
		std::cout << "pow result bl:" << pow(2, l - i) << std::endl;
		ProcessBl<<<grid, block>>>(pow(2, l - i), d_bl + i * cell_num, d_bl + (i + 1) * cell_num, d_bo + i * cell_num);
		cudaDeviceSynchronize();
	}
	// sumArraysOnGPU <<< grid, block >>>(d_Bo, d_Bl);
	printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
	// cudaMemcpy(gpu_ref, d_c, n_bytes, cudaMemcpyDeviceToHost);
	cudaFree(d_bo);
	cudaFree(d_bl);
	free(h_bo);
	free(h_bl);
	std::cout << "this is parallel" << std::endl;

	cudaDeviceReset();

	thrust::host_vector<int> H(4);

	// initialize individual elements
	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;

	thrust::device_vector<int> D = H;

	int sum = reduce(D.begin(), D.end(), static_cast<int>(0), thrust::plus<int>());

	std::cout << "sum:" << sum << std::endl;

	std::cout << D.size() << ", " << D[0] << std::endl;
	thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, "\n"));

	return 0;
}
