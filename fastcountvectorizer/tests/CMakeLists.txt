
set(Python3_FIND_VIRTUALENV FIRST)

# pyenv does not install shared libraries by default:
# https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared
# so enabling static libraries may be convenient.
#
# however, executing numpy's import_array from tests requires Python shared library,
# everything is much simpler if we just require the shared library for test execution
#
# note that enabling back static libraries may require something like this:
#  link_libraries(-lcrypt -lpthread -ldl  -lutil -lm)
# (from python3-config --libs)
set(Python3_USE_STATIC_LIBS FALSE)

find_package (Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
link_directories(AFTER ${Python3_LIBRARY_DIRS})
link_libraries(${Python3_LIBRARIES})


include_directories(${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer)
include_directories(SYSTEM ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/thirdparty)
include_directories(SYSTEM ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/tests/thirdparty)

add_executable(_ext_test
        ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/_ext.cpp
        pyutils.cpp
        test__ext.cpp
        test__ext_impl.cpp)

# set warning levels
# see https://foonathan.net/2018/10/cmake-warnings/
target_compile_options(_ext_test PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Werror -Wextra -Wconversion -Wsign-conversion -pedantic-errors>
        $<$<CXX_COMPILER_ID:MSVC>:
        /W4>)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(_ext_test PRIVATE --coverage)
    target_link_libraries(_ext_test gcov)
endif()

add_test(NAME _ext COMMAND _ext_test)
add_custom_target(run-tests COMMAND _ext_test)