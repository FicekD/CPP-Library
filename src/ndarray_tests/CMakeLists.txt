project(ndarray_tests)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)
FetchContent_MakeAvailable(Catch2)

add_executable(ndarray_tests
    linalg_tests.cpp
    matrix_core_tests.cpp
    ndarray_math_tests.cpp
)

target_link_libraries(ndarray_tests
    PRIVATE ndarray
    PRIVATE Catch2::Catch2WithMain
)

target_include_directories(ndarray_tests
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME AllNdarrayTests COMMAND ndarray_tests)