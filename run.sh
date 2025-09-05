# # SERIAL
export KOKKOS_OPTIONS="-DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF  -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON"
export CMAKE_CXX_FLAGS="-fno-tree-vectorize -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
export HIGHWAY_OPTIONS="-DHWY_COMPILE_ONLY_SCALAR=1"

# # NEON
# export KOKKOS_OPTIONS="-DKOKKOS_ARCH_ARM_NEON=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON"
# export CMAKE_CXX_FLAGS="-march=native -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"

# # AVX2
# export KOKKOS_OPTIONS="-DKOKKOS_ARCH_AVX2=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON"
# export CMAKE_CXX_FLAGS="-mavx2 -mno-avx512f -mno-avx512cd -mno-avx512er -mno-avx512pf -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer -mfma"

# # AVX512
# export KOKKOS_OPTIONS="-DKOKKOS_ARCH_AVX512XEON=ON -DKokkos_ENABLE_AGGRESSIVE_VECTORIZATION=OFF"
# export CMAKE_CXX_FLAGS="-mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512dq -mavx512bw -mavx512vl -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"


# Compile
echo $KOKKOS_OPTIONS
echo $CMAKE_CXX_FLAGS
echo $HIGHWAY_OPTIONS
cmake -S . -B build -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS $HIGHWAY_OPTIONS" $KOKKOS_OPTIONS --fresh 
cmake --build build -j

# Run
time ./build/main_serial 20
time ./build/main_kokkos 20
time ./build/main_hw 20