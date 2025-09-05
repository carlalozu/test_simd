#include <iostream>
#include <cstdlib>
#include <Kokkos_Core.hpp>
#include "recursion.h"
#include <array>
#include <cmath>
#include <iomanip>
#include <string>

/**
 * Main program for benchmarking the serial version of the ggg kernel.
 *
 * This version does not use SIMD and processes one element at a time.
 * You can switch the function evaluate_ggg_vertex_kernel_unrolled_single_value
 * with evaluate_ggg_vertex_kernel to compare the unrolled version with the
 * non-unrolled version.
 */
int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // take power of 2 from command line or default to 2^5
        int p = 5;
        if (argc > 1)
        p = std::atoi(argv[1]);
        long int n = 1 << p;
        std::cout << "Total elements: 2^" << p << ": " << n << std::endl;
        
        // take number of repetitions from command line or default to 5
        const int reps = 5;
        if (argc > 2)
            p = std::atoi(argv[2]);
        std::cout << "Repetitions: " << reps << std::endl;

        // add padding to ensure loads in the kernel are coming from different
        // places in the array
        Evt evt(n + 8 * 3);

        std::cout << "Initializing arrays..." << std::endl;
        auto start_init = std::chrono::high_resolution_clock::now();
        evt.reset_arrays();
        auto end_init = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_init = end_init - start_init;
        std::cout << "Time taken for filling arrays: " << duration_init.count() << " seconds" << std::endl;
        Kokkos::fence();

        std::cout << "Processing " << n << " serial elements unrolled kernel." << std::endl;
        std::chrono::duration<double> duration = std::chrono::duration<double>::zero();
        for (int i = 0; i < reps; i++)
        {
            evt.reset_arrays();
            auto start = std::chrono::high_resolution_clock::now();
            for (size_t idx = 0; idx < n; idx++)
            {
                evaluate_ggg_vertex_kernel_unrolled_single_value(evt, idx);
            };
            Kokkos::fence();
            auto end = std::chrono::high_resolution_clock::now();
            duration = duration + (end - start);
            std::cout << "Completed repetition: " << i << std::endl;
        }
        std::cout << "Time taken for serial loop: " << duration.count() / reps << " seconds" << std::endl;
        Kokkos::fence();

        // Print some results
        std::cout << "Array results (first 16): ";
        for (int i = 0; i < 16; ++i)
        {
            std::cout << "(" << evt._c0r(i) << "," << evt._c0i(i) << "), ";
        }
        std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
