#include "common.cuh"

__global__ void process(const int* __restrict__ nitems_per_cell,
                        const int* __restrict__ cell_data,
                        const int* __restrict__ start_index,
                        int* __restrict__ output,
                        int num_cells) {
    int id = threadIdx.x + 32 * blockIdx.x;
    int n = id >= num_cells ? 0 : nitems_per_cell[id];
    bool block = n >= 16;
    int mask = __ballot(block);

    if (mask != 0) {
        do {
            int bit = __ffs(mask) - 1;
            mask &= ~(1 << bit);

            int block_id = bit + 32 * blockIdx.x;
            int block_n = nitems_per_cell[block_id];
            int block_start = start_index[block_id];
            int block_data  = cell_data[block_id];
            for (int i = threadIdx.x; i < block_n; i += 32) output[block_start + i] = block_data + i;
        } while (mask != 0);
    }

    if (id < num_cells && !block) {
        int start = start_index[id];
        int data  = cell_data[id];
        for (int i = 0; i < n; i++) output[start + i] = data + i;
    }
}

void process_blocked(const int* nitems_per_cell, const int* cell_data, const int* start_index, int* output, int num_cells, int num_items) {
    process<<<round_to(num_cells, 32), 32>>>(nitems_per_cell, cell_data, start_index, output, num_cells);
}
