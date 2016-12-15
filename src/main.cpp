#include <iostream>
#include <random>

void init(const int* nitems_per_cell, const int* cell_data, int n);
float bench(int num_iters);
void copy(std::vector<int>& output);

int generate(int* nitems_per_cell, int* cell_data, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_data(0, 1 << 30);
    std::uniform_int_distribution<> dis_nums(0, 10);

    for (int i = 0; i < n; i++) cell_data[i] = dis_data(gen);

    int total = 0;
    for (int i = 0; i < n; i++) {
        int nitems = 1 << dis_nums(gen);//1 << (8 * std::max(0, dis_nums(gen) - 8));
        nitems_per_cell[i] = nitems;
        total += nitems;
    }
    return total;
}

bool check(const int* nitems_per_cell, const int* cell_data, const int* output, int num_cells) {
    for (int i = 0, j = 0; i < num_cells; i++) {
        int data = cell_data[i];
        for (int k = 0, n = nitems_per_cell[i]; k < n; k++, j++) {
            if (output[j] != data + k) return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const int num_iters = 1000;

    if (argc != 2) {
        std::cerr << "incorrect number of arguments" << std::endl;
        return 1;
    }

    int n = std::strtol(argv[1], nullptr, 10);
    if (n <= 0) {
        std::cerr << "incorrect number of cells" << std::endl;
        return 1;
    }

    std::vector<int> cell_data(n);
    std::vector<int> nitems_per_cell(n);
    std::vector<int> output;

    int total = generate(nitems_per_cell.data(), cell_data.data(), n);
    std::cout << n << " cells, " << total << " items" << std::endl;
    init(nitems_per_cell.data(), cell_data.data(), n);
    float ms = bench(num_iters);
    copy(output);
    if (!check(nitems_per_cell.data(), cell_data.data(), output.data(), n)) {
        std::cerr << "ERROR" << std::endl;
        return 1;
    }
    std::cout << ms / num_iters << " ms/iter." << std::endl;

    return 0;
}
