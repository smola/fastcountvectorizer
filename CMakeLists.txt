# FindPython3 - Numpy requires CMake 3.16
# Otherwise, 3.12 would be fine.
cmake_minimum_required(VERSION 3.16)

project(fastcountvectorizer)

set(CMAKE_CXX_STANDARD 14)

enable_testing()
add_subdirectory(fastcountvectorizer/tests)

