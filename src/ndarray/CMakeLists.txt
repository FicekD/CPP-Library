cmake_minimum_required (VERSION 3.8)

set(SOURCES
    random.cpp
    range.cpp
)

add_library(ndarray STATIC ${SOURCES})

target_include_directories(ndarray PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
