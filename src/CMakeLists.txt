cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project ("src")

include(CTest)
enable_testing()

add_subdirectory ("ndarray")
add_subdirectory("ndarray_tests")
add_subdirectory("lib_tests_console")