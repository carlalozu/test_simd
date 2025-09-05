
#include <iostream>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "recursion.h"
#include <array>
#include <cmath>
#include <iomanip>
#include <string>

using simd_double = Kokkos::Experimental::simd<double>;

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

        // Initialize arrays, reshape views to n
        Evt evt(n+8*3);
        
        std::cout << "Initializing arrays..." << std::endl;
        auto start_init = std::chrono::high_resolution_clock::now();
        evt.reset_arrays();
        auto end_init = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_init = end_init - start_init;
        std::cout << "Time taken for filling arrays: " << duration_init.count() << " seconds" << std::endl;
        Kokkos::fence();

        const int reps = 5;

        int LANES = Lanes(d);
        const int num_groups = n / LANES;
        int remaining = n % LANES;

        if (num_groups > 0)
        {
            std::cout << "SIMD groups: " << num_groups << std::endl;
            std::cout << "Remaining elements: " << remaining << std::endl;
            tag_type tag{};

            const hn::ScalableTag<uint64_t> d;
            std::cout << "Running SIMD operations with " << Lanes(d) << " lanes." << std::endl;

            // time it
            std::chrono::duration<double> duration = std::chrono::duration<double>::zero();
            for(int i=0;i<reps;i++){
                evt.reset_arrays();
                auto start = std::chrono::high_resolution_clock::now();
                for (size_t idx = 0; idx < n; idx += Lanes(d)){
                    evaluate_ggg_vertex_kernel_highway(evt, idx);
                };
                Kokkos::fence();
                auto end = std::chrono::high_resolution_clock::now();
                duration = duration + (end - start);
                std::cout << "Completed repetition: " << i << std::endl;
            }
            std::cout << "Time taken for SIMD loop: " << duration.count()/reps << " seconds" << std::endl;
            Kokkos::fence();
        }
        if (remaining > 0) // TODO: check if necessary
        {
            std::cout << "Processing " << remaining << " serial elements not unrolled." << std::endl;
            // Handle remaining elements if any
            std::chrono::duration<double> duration = std::chrono::duration<double>::zero();
            for(int i=0;i<reps;i++){
                evt.reset_arrays();
                auto start = std::chrono::high_resolution_clock::now();
                for (size_t idx = 0; idx < n; idx ++){
                    evaluate_ggg_vertex_kernel_unrolled_single_value(evt, idx);
                };
                Kokkos::fence();
                auto end = std::chrono::high_resolution_clock::now();
                duration = duration + (end - start);
                std::cout << "Completed repetition: " << i << std::endl;
            }   
            std::cout << "Time taken for serial loop: " << duration.count()/reps << " seconds" << std::endl;
            Kokkos::fence();
        }
        // Print some results
        std::cout << "Array results (last 16): ";
        for (int i = 0; i < 16; ++i)
        {
            std::cout << "(" << evt._c0r(n-16-i) << "," << evt._c0i(n-16-i) << "), ";
        }
        std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
