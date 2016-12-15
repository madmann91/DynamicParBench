#include <cstdint>
#include "common.cuh"

__global__ void inner_loop(int* __restrict__ output, int start, int n, int data) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= n) return;

    output[start + id] = data + id;
}

__global__ void process(const int* __restrict__ nitems_per_cell,
                        const int* __restrict__ cell_data,
                        const int* __restrict__ start_index,
                        int* __restrict__ output,
                        int num_cells) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_cells) return;

    int n = nitems_per_cell[id];
    int start = start_index[id];
    int data  = cell_data[id];
    if (n > 4096)
        inner_loop<<<round_to(n, blockDim.x), blockDim.x>>>(output, start, n, data);
    else for (int i = 0; i < n; i++) output[start + i] = data + i;
}

void process_dynamic(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items) {
    process<<<round_to(num_cells, block_size), block_size>>>(nitems_per_cell, cell_data, start_index, output, num_cells);
}
