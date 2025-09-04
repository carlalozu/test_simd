# This snippet is adapted from:
# https://kokkos.org/kokkos-core-wiki/get-started/integrating-kokkos-into-your-cmake-project.html#supporting-both-external-and-embedded-kokkos
list(APPEND CMAKE_MESSAGE_CONTEXT "kokkos")
find_package(Kokkos CONFIG) # Try to find Kokkos externally
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    message(STATUS "Kokkos not found externally. Fetching via FetchContent.")
    if(POLICY CMP0135)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(
        Kokkos
        SOURCE_DIR      "/Users/carla/cernbox/kokkos-4.6.01"
    )

    if(POLICY CMP0135)
      cmake_policy(POP)
    endif()
    FetchContent_MakeAvailable(Kokkos)
    message(STATUS "Kokkos_DIR: ${Kokkos_DIR}") 
endif()
list(POP_BACK CMAKE_MESSAGE_CONTEXT)


find_package(HWY REQUIRED)