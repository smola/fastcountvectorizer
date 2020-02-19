add_library(_ext STATIC
        _strings.cpp
        _sputils.cpp
        _ext.cpp)

target_link_libraries(_ext PRIVATE Python3::Module)
target_link_libraries(_ext PUBLIC Python3::NumPy)

target_include_directories(_ext PUBLIC ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer)
target_include_directories(_ext SYSTEM PRIVATE ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/thirdparty)

# set warning levels
# see https://foonathan.net/2018/10/cmake-warnings/
target_compile_options(_ext PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
        -Wall -Werror -Wextra -Wconversion -Wsign-conversion -pedantic-errors>
        $<$<CXX_COMPILER_ID:MSVC>:
        /W4>)