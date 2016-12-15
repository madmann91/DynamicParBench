#include <vector>
#include <numeric>
#include <cuda_runtime_api.h>
#include "common.cuh"

namespace cuda {
    static int* nitems_per_cell;
    static int* cell_data;
    static int* start_index;
    static int* output;
    static int num_cells;
    static int num_items;
}

void process_naive  (const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items);
void process_blocked(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items);
void process_bsearch(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items);
void process_dynamic(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items);

void init(const int* nitems_per_cell, const int* cell_data, int n) {
    cuda::num_cells = n;
    std::vector<int> start_index(n + 1);
    start_index[0] = 0;
    std::partial_sum(nitems_per_cell, nitems_per_cell + n, start_index.data() + 1);
    cuda::num_items = start_index.back();

    CHECK_CUDA_CALL(cudaMalloc(&cuda::nitems_per_cell, sizeof(int) * n));
    CHECK_CUDA_CALL(cudaMalloc(&cuda::cell_data,       sizeof(int) * n));
    CHECK_CUDA_CALL(cudaMalloc(&cuda::start_index,     sizeof(int) * n));
    CHECK_CUDA_CALL(cudaMalloc(&cuda::output,          sizeof(int) * cuda::num_items));

    CHECK_CUDA_CALL(cudaMemcpy(cuda::nitems_per_cell, nitems_per_cell,    sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(cuda::cell_data,       cell_data,          sizeof(int) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(cuda::start_index,     start_index.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
}

float bench(int num_iters) {
    cudaEvent_t start, end;
    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&end));

    float total = 0;
    for (int i = 0; i < num_iters; i++) {
        CHECK_CUDA_CALL(cudaEventRecord(start));

        BENCH_FN(cuda::nitems_per_cell, cuda::cell_data, cuda::start_index, cuda::output, cuda::num_cells, cuda::num_items);

        CHECK_CUDA_CALL(cudaEventRecord(end));
        CHECK_CUDA_CALL(cudaEventSynchronize(end));
        float ms;
        CHECK_CUDA_CALL(cudaEventElapsedTime(&ms, start, end));
        total += ms;
    }

    CHECK_CUDA_CALL(cudaEventDestroy(start));
    CHECK_CUDA_CALL(cudaEventDestroy(end));

    CHECK_CUDA_CALL(cudaFree(cuda::nitems_per_cell));
    CHECK_CUDA_CALL(cudaFree(cuda::cell_data));
    CHECK_CUDA_CALL(cudaFree(cuda::start_index));
    return total;
}

void copy(std::vector<int>& output) {
    output.resize(cuda::num_items);
    CHECK_CUDA_CALL(cudaMemcpy(output.data(), cuda::output, sizeof(int) * cuda::num_items, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL(cudaFree(cuda::output));
}
