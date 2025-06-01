add_executable(lib_tests_console main.cpp)

target_link_libraries(lib_tests_console PRIVATE ndarray)

target_include_directories(lib_tests_console PRIVATE
    ${PROJECT_SOURCE_DIR}/ndarray
)
