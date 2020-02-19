# FindPython3 - Numpy requires CMake 3.16
# Otherwise, 3.12 would be fine.
cmake_minimum_required(VERSION 3.16)

project(fastcountvectorizer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python3_FIND_STRATEGY LOCATION)
unset(Python3_USE_STATIC_LIBS)
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_REGISTRY LAST)

# [macOS] On GitHub Actions there are two conflicting Python paths:
#   - /usr/local/Frameworks/Python.framework/Versions/3.7
#   - /Users/runner/hostedtoolcache/Python/3.7.6/x64
#
# Setting Python3_FIND_FRAMEWORK will make it compile but numpy import_array
# will still fail on runtime, because interpreter is detected on hostedtoolcache
# but dynamic library is detected from Frameworks.
#
# See https://github.com/actions/virtual-environments/issues/125
set(Python3_FIND_FRAMEWORK LAST)

find_package(Python3 COMPONENTS Interpreter Development NumPy)

add_subdirectory(fastcountvectorizer)
add_subdirectory(fastcountvectorizer/tests)

