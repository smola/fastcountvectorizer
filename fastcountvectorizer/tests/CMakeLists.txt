add_executable(_ext_test
        test__ext.cpp
        test__ext_impl.cpp
        test__strings_impl.cpp
        test__sputils_impl.cpp)

add_custom_target(run-tests COMMAND _ext_test)

target_link_libraries(_ext_test PRIVATE Python3::Python)
target_link_libraries(_ext_test PRIVATE Python3::NumPy)

# Using NumPy C API requires loading the NumPy module; when Python is linked statically, this requires exporting symbols
# right from the test executable. Some environments when we link statically may be:
#   - pyenv-managed installs, when no shared libraries where installed explicitly:
#     https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared
#   - some continuous integration systems such as macOS on GitHub Actions:
#     https://github.com/actions/virtual-environments/issues/125
set_target_properties(_ext_test PROPERTIES ENABLE_EXPORTS 1)

target_link_libraries(_ext_test PRIVATE _ext)
target_include_directories(_ext_test PRIVATE ${pybind11_INCLUDE_DIRS})
target_include_directories(_ext_test SYSTEM
        PRIVATE ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/thirdparty
        PRIVATE ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/tests/thirdparty)

# set warning levels
# see https://foonathan.net/2018/10/cmake-warnings/
target_compile_options(_ext_test PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Werror -Wextra -Wconversion -Wsign-conversion -pedantic-errors>
        $<$<CXX_COMPILER_ID:MSVC>:
        /W4>)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(_ext_test PRIVATE --coverage)
    target_link_libraries(_ext_test PRIVATE gcov)
endif()

