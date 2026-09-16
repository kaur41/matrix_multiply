#include <gtest/gtest.h>
#include <iostream>
#include <chrono>
#include <vector>
#include "../seq/seq_task.hpp"
#include "../omp/omp_task.hpp"
#include "../tbb/tbb_task.hpp"

// Вспомогательные функции обертки жизненного цикла задач
CRSMatrix run_task_sequential(const CRSMatrix& A_mat, const CRSMatrix& B_mat) {
    SparseMatrixTaskSequential task(A_mat, B_mat);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

CRSMatrix run_task_openmp(const CRSMatrix& A_mat, const CRSMatrix& B_mat) {
    SparseMatrixTaskOpenMP task(A_mat, B_mat);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

// Изменено имя аргументов для избежания конфликтов
CRSMatrix run_task_tbb(const CRSMatrix& A_mat, const CRSMatrix& B_mat) {
    SparseMatrixTaskTBB task(A_mat, B_mat);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

// =========================================================================
// ГРУППА 1: ТЕСТЫ НА КРАЕВЫЕ УСЛОВИЯ И ВАЛИДАЦИЮ
// =========================================================================

TEST(SparseMatrix_Validation, Wrong_Dimensions) {
    CRSMatrix A_test = generate_random_sparse_matrix(5, 10, 0.1);
    CRSMatrix B_test = generate_random_sparse_matrix(8, 5, 0.1);
    SparseMatrixTaskSequential task(A_test, B_test);
    EXPECT_FALSE(task.validation());
}

TEST(SparseMatrix_Validation, Empty_Dimensions_Zero) {
    CRSMatrix A_test = generate_random_sparse_matrix(0, 5, 0.1);
    CRSMatrix B_test = generate_random_sparse_matrix(5, 5, 0.1);
    SparseMatrixTaskSequential task(A_test, B_test);
    EXPECT_FALSE(task.validation());
}

// =========================================================================
// ГРУППА 2: СТРУКТУРНЫЕ ТЕСТЫ (Специфика разреженных форматов)
// =========================================================================

TEST(SparseMatrix_Structure, Single_Element_In_Center) {
    CRSMatrix A_test; A_test.rows = 10; A_test.cols = 10; A_test.row_index.assign(11, 0);
    A_test.values.push_back(5.5); A_test.col_index.push_back(5);
    for (int i = 6; i <= 10; ++i) A_test.row_index[i] = 1;

    CRSMatrix B_test = A_test;

    CRSMatrix res_seq = run_task_sequential(A_test, B_test);
    CRSMatrix res_omp = run_task_openmp(A_test, B_test);
    CRSMatrix res_tbb = run_task_tbb(A_test, B_test);

    EXPECT_TRUE(compare_matrices(res_seq, res_omp));
    EXPECT_TRUE(compare_matrices(res_seq, res_tbb));
}

TEST(SparseMatrix_Structure, Mutual_Annihilation_To_Zero) {
    CRSMatrix A_test; A_test.rows = 2; A_test.cols = 2;
    A_test.values = { 2.0, -2.0 }; A_test.col_index = { 0, 1 }; A_test.row_index = { 0, 2, 2 };

    CRSMatrix B_test; B_test.rows = 2; B_test.cols = 2;
    B_test.values = { 1.0, 1.0 }; B_test.col_index = { 0, 0 }; B_test.row_index = { 0, 1, 2 };

    CRSMatrix res_seq = run_task_sequential(A_test, B_test);
    CRSMatrix res_omp = run_task_openmp(A_test, B_test);
    CRSMatrix res_tbb = run_task_tbb(A_test, B_test);

    EXPECT_EQ(res_seq.values.size(), 0);
    EXPECT_TRUE(compare_matrices(res_seq, res_omp));
    EXPECT_TRUE(compare_matrices(res_seq, res_tbb));
}

// =========================================================================
// ГРУППА 3: МАССИВНЫЕ СТРЕСС-ТЕСТЫ КОМБИНАЦИЙ (Чистые циклы без макросов)
// =========================================================================

TEST(SparseMatrix_Stress, Combinations_Loop) {
    std::vector<int> rows_sizes = { 5, 25, 100 };
    std::vector<int> cols_sizes = { 5, 45, 150 };
    std::vector<double> densities = { 0.0, 0.04, 0.2, 1.0 };

    for (int r : rows_sizes) {
        for (int c : cols_sizes) {
            for (double d : densities) {
                CRSMatrix A_loop = generate_random_sparse_matrix(r, c, d);
                CRSMatrix B_loop = generate_random_sparse_matrix(c, r, d);

                CRSMatrix res_seq = run_task_sequential(A_loop, B_loop);
                CRSMatrix res_omp = run_task_openmp(A_loop, B_loop);
                CRSMatrix res_tbb = run_task_tbb(A_loop, B_loop);

                ASSERT_TRUE(compare_matrices(res_seq, res_omp));
                ASSERT_TRUE(compare_matrices(res_seq, res_tbb));
            }
        }
    }
}

// ==========================================
//                 МЕЙН
// ==========================================
int main(int argc, char** argv) {
    std::cout << "=== RUNNING EXPANDED ACADEMIC UNIT TESTS ===\n";
    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();

    if (test_result != 0) {
        std::cout << "\nTests failed! Speed benchmark aborted.\n";
        return test_result;
    }

    std::cout << "\n=== RUNNING PERFORMANCE BENCHMARK ===\n";
    int benchmark_N = 1000;
    double benchmark_density = 0.03;

    std::cout << "Generating big sparse matrices " << benchmark_N << "x" << benchmark_N << "...\n";
    CRSMatrix A_large = generate_random_sparse_matrix(benchmark_N, benchmark_N, benchmark_density);
    CRSMatrix B_large = generate_random_sparse_matrix(benchmark_N, benchmark_N, benchmark_density);

    auto start_time = std::chrono::high_resolution_clock::now();
    CRSMatrix r_seq = run_task_sequential(A_large, B_large);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << ">> Sequential (SEQ) time: " << time_seq << " ms\n";

    start_time = std::chrono::high_resolution_clock::now();
    CRSMatrix r_omp = run_task_openmp(A_large, B_large);
    end_time = std::chrono::high_resolution_clock::now();
    auto time_omp = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << ">> OpenMP (OMP) time:     " << time_omp << " ms";
    if (time_seq > 0) std::cout << " (Boost: x" << (double)time_seq / time_omp << ")\n";
    else std::cout << "\n";

    start_time = std::chrono::high_resolution_clock::now();
    CRSMatrix r_tbb = run_task_tbb(A_large, B_large);
    end_time = std::chrono::high_resolution_clock::now();
    auto time_tbb = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << ">> Intel TBB (TBB) time:  " << time_tbb << " ms";
    if (time_seq > 0) std::cout << " (Boost: x" << (double)time_seq / time_tbb << ")\n";
    else std::cout << "\n";

    return 0;
}
