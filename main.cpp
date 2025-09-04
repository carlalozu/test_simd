
#include <iostream>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <string>

using simd_double = Kokkos::Experimental::simd<double>;
using tag_type = Kokkos::Experimental::vector_aligned_tag;

int main(int argc, char *argv[])
{
#if defined(KOKKOS_ARCH_AVX512XEON)
    std::cout << "KOKKOS_ARCH_AVX512XEON enabled" << std::endl;
#elif defined(KOKKOS_ARCH_AVX2)
    std::cout << "KOKKOS_ARCH_AVX2 enabled" << std::endl;
#elif defined(KOKKOS_ARCH_ARM_NEON)
    std::cout << "KOKKOS_ARCH_ARM_NEON enabled" << std::endl;
#endif
    Kokkos::initialize(argc, argv);
    {
        // take n from command line or default to 1 million
        int p = 3;
        if (argc > 1)
            p = std::atoi(argv[1]);
        
        long int n = 1 << p;
        std::cout << "Total elements: 2^" << p << ": " << n << std::endl;
        Kokkos::View<double *> a_view("a_view", n);
        Kokkos::View<double *> b_view("b_view", n);
        Kokkos::View<double *> c_view("c_view", n);
        // Initialize arrays
        std::cout << "Initializing arrays..." << std::endl;

        auto start_init = std::chrono::high_resolution_clock::now();
        Kokkos::parallel_for("init_arrays", n, KOKKOS_LAMBDA(int i) {
        a_view(i) = static_cast<double>(i + 1);
        b_view(i) = static_cast<double>(i + 5); });
        auto end_init = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_init = end_init - start_init;
        std::cout << "Time taken for filling arrays: " << duration_init.count() << " seconds" << std::endl;
        Kokkos::fence();

        int LANES = simd_double::size();
        const int num_groups = n / LANES;
        int remaining = n % LANES;
        if (LANES == 1){
            LANES = 0; // no SIMD, all elements to be processed in serial mode
            remaining = n; // all elements to be processed in serial mode
        }

        std::cout << "SIMD groups: " << num_groups << std::endl;
        std::cout << "Remaining elements: " << remaining << std::endl;
        if (LANES > 1 && num_groups > 0)
        {
            tag_type tag{};

            std::cout << "Running SIMD operations with " << LANES << " lanes." << std::endl;
            // SIMD operations on arrays

            // time it
            auto start = std::chrono::high_resolution_clock::now();
            Kokkos::parallel_for("simd_operations", Kokkos::RangePolicy<>(0, num_groups), KOKKOS_LAMBDA(int simd_group) {
            const int i = simd_group * LANES;
            simd_double a_simd(&a_view(i), tag);
            simd_double b_simd(&b_view(i), tag);

            simd_double c_simd = a_simd + b_simd;
            c_simd = c_simd * b_simd;
            c_simd = c_simd / a_simd;
            c_simd = c_simd - b_simd;
            c_simd = c_simd + a_simd * b_simd;
            c_simd = c_simd * b_simd + c_simd / b_simd;
            c_simd.copy_to(&c_view(i), tag); });
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Time taken for SIMD loop: " << duration.count() << " seconds" << std::endl;
            Kokkos::fence();
        }
        if (remaining > 0)
        {
            std::cout << "Processing " << remaining << " elements." << std::endl;
            // Handle remaining elements if any
            auto start = std::chrono::high_resolution_clock::now();
            Kokkos::parallel_for("remaining_elements", remaining, KOKKOS_LAMBDA(int i) {
            const int idx = num_groups * LANES + i;
            c_view(idx) = a_view(idx) + b_view(idx);
            c_view(idx) = c_view(idx) * b_view(idx);
            c_view(idx) = c_view(idx) / a_view(idx);
            c_view(idx) = c_view(idx) - b_view(idx);
            c_view(idx) = c_view(idx) + a_view(idx) * b_view(idx);
            c_view(idx) = c_view(idx) * b_view(idx) + c_view(idx) / b_view(idx);
            });
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Time taken for serial loop: " << duration.count() << " seconds" << std::endl;
            Kokkos::fence();
        }
        // Print some results
        std::cout << "Array results (first 16): ";
        for (int i = 0; i < 16; ++i)
        {
            std::cout << c_view(i) << " ";
        }
        std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}