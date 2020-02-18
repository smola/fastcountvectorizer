# FindPython3 - Numpy requires CMake 3.16
# Otherwise, 3.12 would be fine.
cmake_minimum_required(VERSION 3.16)

project(fastcountvectorizer)

set(CMAKE_CXX_STANDARD 11)

set(Python3_FIND_STRATEGY LOCATION)
unset(Python3_USE_STATIC_LIBS)
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_REGISTRY LAST)
set(Python3_FIND_FRAMEWORK LAST)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

add_subdirectory(fastcountvectorizer)
add_subdirectory(fastcountvectorizer/tests)

