#include "common.cuh"

__device__ __forceinline__ int upper_bound(const int* __restrict__ start_index, int num_cells, int id) {
    int a = 0, b = num_cells;
    while (a < b) {
        int m = (a + b) / 2;
        int k = start_index[m];
        if (id >= k) a = m + 1;
        else         b = m;
    }
    return b - 1;
}

__global__ void process(const int* __restrict__ cell_data,
                        const int* __restrict__ start_index,
                        int* __restrict__ output,
                        int num_cells,
                        int num_items) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_items) return;

    int top_id = upper_bound(start_index, num_cells, id);
    int data = cell_data[top_id];
    int i = id - start_index[top_id];
    output[id] = data + i;
}

void process_bsearch(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items) {
    process<<<round_to(num_items, block_size), block_size>>>(cell_data, start_index, output, num_cells, num_items);
}
