cmake_minimum_required(VERSION 3.20)
project(Richarallele C)

set(CMAKE_C_STANDARD 99)
find_package(OpenMP REQUIRED)

add_executable(Richarallele Ex3/breaker_for_opti.c)
target_link_libraries(Richarallele PRIVATE OpenMP::OpenMP_C crypt)