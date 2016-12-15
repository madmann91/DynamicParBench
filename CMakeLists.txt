cmake_minimum_required(VERSION 3.1)
project(DynamicParBench)

find_package(CUDA)
set(CMAKE_CXX_STANDARD 11)
add_subdirectory(src)
