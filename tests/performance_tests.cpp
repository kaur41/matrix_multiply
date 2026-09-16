#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include "../seq/seq_task.hpp"
#include "../omp/omp_task.hpp"
#include "../tbb/tbb_task.hpp"

// Обертки конвейера
CRSMatrix run_perf_seq(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskSequential task(A, B);
    task.validation(); task.pre_scheduling(); task.run(); task.post_scheduling();
    return task.get_result();
}

CRSMatrix run_perf_omp(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskOpenMP task(A, B);
    task.validation(); task.pre_scheduling(); task.run(); task.post_scheduling();
    return task.get_result();
}

CRSMatrix run_perf_tbb(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskTBB task(A, B);
    task.validation(); task.pre_scheduling(); task.run(); task.post_scheduling();
    return task.get_result();
}

// Тест производительности в формате GTest
TEST(SparseMatrix_Performance, Multithreading_Speed_Benchmark) {
    int N = 2500; // Увеличили размер до 2500, чтобы потоки гарантированно ускорились!
    double density = 0.02;

    std::cout << "\n[PERF] Generating heavy matrices " << N << "x" << N << "...\n";
    CRSMatrix A = generate_random_sparse_matrix(N, N, density);
    CRSMatrix B = generate_random_sparse_matrix(N, N, density);

    auto start = std::chrono::high_resolution_clock::now();
    CRSMatrix r_seq = run_perf_seq(A, B);
    auto end = std::chrono::high_resolution_clock::now();
    auto t_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ">> SEQ time: " << t_seq << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    CRSMatrix r_omp = run_perf_omp(A, B);
    end = std::chrono::high_resolution_clock::now();
    auto t_omp = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ">> OMP time: " << t_omp << " ms (Boost: x" << (double)t_seq / t_omp << ")\n";

    start = std::chrono::high_resolution_clock::now();
    CRSMatrix r_tbb = run_perf_tbb(A, B);
    end = std::chrono::high_resolution_clock::now();
    auto t_tbb = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ">> TBB time: " << t_tbb << " ms (Boost: x" << (double)t_seq / t_tbb << ")\n";

    ASSERT_TRUE(compare_matrices(r_seq, r_omp));
    ASSERT_TRUE(compare_matrices(r_seq, r_tbb));
}

int main(int argc, char** argv) {
    std::cout << "=== PARALLEL PROGRAMMING COURSE PERFORMANCE SCRIPT ===\n";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
