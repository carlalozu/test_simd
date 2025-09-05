cmake_minimum_required(VERSION 3.19.0)
project(TestSIMD)
include(FetchContent)

include(EnableKokkos.cmake)


add_executable(main main.cpp)
add_executable(main_ggg main_ggg.cpp)
add_executable(main_ggg_hw main_ggg_hw.cpp)

target_link_libraries(main PRIVATE Kokkos::kokkos hwy::hwy)
target_link_libraries(main_ggg PRIVATE Kokkos::kokkos hwy::hwy)
target_link_libraries(main_ggg_hw PRIVATE Kokkos::kokkos hwy::hwy)

if (KOKKOS_ARCH_AVX512XEON)
  message(STATUS "✅ KOKKOS_ARCH_AVX512XEON enabled")
elseif (KOKKOS_ARCH_AVX2)
  message(STATUS "✅ KOKKOS_ARCH_AVX2 enabled")
elseif (KOKKOS_ARCH_ARM_NEON)
  message(STATUS "✅ KOKKOS_ARCH_ARM_NEON enabled")
endif()
