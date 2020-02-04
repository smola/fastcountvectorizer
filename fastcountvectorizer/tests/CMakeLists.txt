

set(Python3_FIND_VIRTUALENV FIRST)
find_package (Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})
link_directories(AFTER ${Python3_LIBRARY_DIRS})
link_libraries(${Python3_LIBRARIES})
# set required libraries when Python static library is found
link_libraries(-lcrypt -lpthread -ldl  -lutil -lm)
include_directories(${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer)

add_executable(_ext_test ${fastcountvectorizer_SOURCE_DIR}/fastcountvectorizer/_ext.cpp test__ext.cpp test__ext_impl.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(_ext_test PRIVATE --coverage)
    target_link_libraries(_ext_test gcov)
endif()

add_test(NAME _ext COMMAND _ext_test)
add_custom_target(run-tests COMMAND _ext_test)