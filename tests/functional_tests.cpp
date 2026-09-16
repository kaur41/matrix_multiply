#include <gtest/gtest.h>
#include <vector>
#include "../seq/seq_task.hpp"
#include "../omp/omp_task.hpp"
#include "../tbb/tbb_task.hpp"

// Обертки жизненного цикла для тестов
CRSMatrix run_seq(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskSequential task(A, B);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

CRSMatrix run_omp(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskOpenMP task(A, B);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

CRSMatrix run_tbb(const CRSMatrix& A, const CRSMatrix& B) {
    SparseMatrixTaskTBB task(A, B);
    if (task.validation()) {
        task.pre_scheduling();
        task.run();
        task.post_scheduling();
    }
    return task.get_result();
}

// --- НАБОР ФУНКЦИОНАЛЬНЫХ ТЕСТОВ ---

TEST(SparseMatrix_Functional, Incorrect_Dimensions_Validation) {
    CRSMatrix A = generate_random_sparse_matrix(5, 12, 0.1);
    CRSMatrix B = generate_random_sparse_matrix(8, 5, 0.1);
    SparseMatrixTaskSequential task(A, B);
    EXPECT_FALSE(task.validation());
}

TEST(SparseMatrix_Functional, Empty_Matrix_Validation) {
    CRSMatrix A = generate_random_sparse_matrix(0, 5, 0.1);
    CRSMatrix B = generate_random_sparse_matrix(5, 5, 0.1);
    SparseMatrixTaskSequential task(A, B);
    EXPECT_FALSE(task.validation());
}

TEST(SparseMatrix_Functional, Boundary_Matrix_1x1) {
    CRSMatrix A = generate_random_sparse_matrix(1, 1, 1.0);
    CRSMatrix B = generate_random_sparse_matrix(1, 1, 1.0);

    CRSMatrix res_seq = run_seq(A, B);
    CRSMatrix res_omp = run_omp(A, B);
    CRSMatrix res_tbb = run_tbb(A, B);

    EXPECT_TRUE(compare_matrices(res_seq, res_omp));
    EXPECT_TRUE(compare_matrices(res_seq, res_tbb));
}

TEST(SparseMatrix_Functional, Zero_Density_Matrices) {
    int N = 50;
    CRSMatrix A = generate_random_sparse_matrix(N, N, 0.0);
    CRSMatrix B = generate_random_sparse_matrix(N, N, 0.0);

    CRSMatrix res_seq = run_seq(A, B);
    CRSMatrix res_omp = run_omp(A, B);
    CRSMatrix res_tbb = run_tbb(A, B);

    EXPECT_EQ(res_seq.values.size(), 0);
    EXPECT_TRUE(compare_matrices(res_seq, res_omp));
    EXPECT_TRUE(compare_matrices(res_seq, res_tbb));
}

TEST(SparseMatrix_Functional, Absolute_Annihilation_To_Zero) {
    // Матрицы, дающие чистый ноль при перемножении (тест очистки буферов)
    CRSMatrix A; A.rows = 2; A.cols = 2;
    A.values = { 2.5, -2.5 }; A.col_index = { 0, 1 }; A.row_index = { 0, 2, 2 };
    CRSMatrix B; B.rows = 2; B.cols = 2;
    B.values = { 1.0, 1.0 }; B.col_index = { 0, 0 }; B.row_index = { 0, 1, 2 };

    CRSMatrix res_seq = run_seq(A, B);
    EXPECT_EQ(res_seq.values.size(), 0);
}

TEST(SparseMatrix_Functional, Identity_Diagonal_Execution) {
    int N = 40;
    CRSMatrix A; A.rows = N; A.cols = N; A.row_index.push_back(0);
    for (int i = 0; i < N; ++i) {
        A.values.push_back(1.0); A.col_index.push_back(i);
        A.row_index.push_back(static_cast<int>(A.values.size()));
    }
    CRSMatrix B = generate_random_sparse_matrix(N, N, 0.12);
    EXPECT_TRUE(compare_matrices(run_seq(A, B), run_omp(A, B)));
}

TEST(SparseMatrix_Functional, Massive_Combinations_Loop) {
    std::vector<int> r_sizes = { 5, 20, 80 };
    std::vector<int> c_sizes = { 5, 30, 100 };
    std::vector<double> dens = { 0.0, 0.05, 0.3, 1.0 };

    for (int r : r_sizes) {
        for (int c : c_sizes) {
            for (double d : dens) {
                CRSMatrix A = generate_random_sparse_matrix(r, c, d);
                CRSMatrix B = generate_random_sparse_matrix(c, r, d);
                ASSERT_TRUE(compare_matrices(run_seq(A, B), run_omp(A, B)));
                ASSERT_TRUE(compare_matrices(run_seq(A, B), run_tbb(A, B)));
            }
        }
    }
}
