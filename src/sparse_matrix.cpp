#include "sparse_matrix.hpp"
#include <random>
#include <cmath>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

CRSMatrix generate_random_sparse_matrix(int rows, int cols, double density) {
    CRSMatrix M; M.rows = rows; M.cols = cols; M.row_index.push_back(0);
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_val(1.0, 5.0);
    std::uniform_real_distribution<double> dis_prob(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis_prob(gen) < density) {
                M.values.push_back(dis_val(gen));
                M.col_index.push_back(j);
            }
        }
        M.row_index.push_back(static_cast<int>(M.values.size()));
    }
    return M;
}

bool compare_matrices(const CRSMatrix& A, const CRSMatrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    if (A.row_index != B.row_index || A.col_index != B.col_index) return false;
    for (size_t i = 0; i < A.values.size(); ++i) {
        if (std::abs(A.values[i] - B.values[i]) > 1e-5) return false;
    }
    return true;
}

// 1. онякеднбюрекэмюъ бепяхъ
CRSMatrix multiply_sequential(const CRSMatrix& A, const CRSMatrix& B) {
    CRSMatrix C; C.rows = A.rows; C.cols = B.cols; C.row_index.push_back(0);
    std::vector<double> temp_row(B.cols, 0.0);

    for (int i = 0; i < A.rows; ++i) {
        for (int k = A.row_index[i]; k < A.row_index[i + 1]; ++k) {
            int a_col = A.col_index[k]; double a_val = A.values[k];
            for (int j = B.row_index[a_col]; j < B.row_index[a_col + 1]; ++j) {
                temp_row[B.col_index[j]] += a_val * B.values[j];
            }
        }
        for (int j = 0; j < B.cols; ++j) {
            if (std::abs(temp_row[j]) > 1e-9) {
                C.values.push_back(temp_row[j]); C.col_index.push_back(j);
                temp_row[j] = 0.0;
            }
        }
        C.row_index.push_back(static_cast<int>(C.values.size()));
    }
    return C;
}

// 2. OpenMP бепяхъ
CRSMatrix multiply_openmp(const CRSMatrix& A, const CRSMatrix& B) {
    CRSMatrix C; C.rows = A.rows; C.cols = B.cols;
    std::vector<std::vector<double>> local_vals(A.rows);
    std::vector<std::vector<int>> local_cols(A.rows);
    std::vector<int> local_nnz(A.rows, 0);

#pragma omp parallel
    {
        std::vector<double> temp_row(B.cols, 0.0);
#pragma omp for
        for (int i = 0; i < A.rows; ++i) {
            for (int k = A.row_index[i]; k < A.row_index[i + 1]; ++k) {
                int a_col = A.col_index[k]; double a_val = A.values[k];
                for (int j = B.row_index[a_col]; j < B.row_index[a_col + 1]; ++j) {
                    temp_row[B.col_index[j]] += a_val * B.values[j];
                }
            }
            for (int j = 0; j < B.cols; ++j) {
                if (std::abs(temp_row[j]) > 1e-9) {
                    local_vals[i].push_back(temp_row[j]);
                    local_cols[i].push_back(j);
                    temp_row[j] = 0.0;
                }
            }
            local_nnz[i] = static_cast<int>(local_vals[i].size());
        }
    }

    C.row_index.push_back(0);
    for (int i = 0; i < A.rows; ++i) {
        C.values.insert(C.values.end(), local_vals[i].begin(), local_vals[i].end());
        C.col_index.insert(C.col_index.end(), local_cols[i].begin(), local_cols[i].end());
        C.row_index.push_back(static_cast<int>(C.values.size()));
    }
    return C;
}

// 3. TBB бепяхъ
CRSMatrix multiply_tbb(const CRSMatrix& A, const CRSMatrix& B) {
    CRSMatrix C; C.rows = A.rows; C.cols = B.cols;
    std::vector<std::vector<double>> local_vals(A.rows);
    std::vector<std::vector<int>> local_cols(A.rows);

    tbb::parallel_for(tbb::blocked_range<int>(0, A.rows), [&](const tbb::blocked_range<int>& r) {
        std::vector<double> temp_row(B.cols, 0.0);
        for (int i = r.begin(); i != r.end(); ++i) {
            for (int k = A.row_index[i]; k < A.row_index[i + 1]; ++k) {
                int a_col = A.col_index[k]; double a_val = A.values[k];
                for (int j = B.row_index[a_col]; j < B.row_index[a_col + 1]; ++j) {
                    temp_row[B.col_index[j]] += a_val * B.values[j];
                }
            }
            for (int j = 0; j < B.cols; ++j) {
                if (std::abs(temp_row[j]) > 1e-9) {
                    local_vals[i].push_back(temp_row[j]);
                    local_cols[i].push_back(j);
                    temp_row[j] = 0.0;
                }
            }
        }
        });

    C.row_index.push_back(0);
    for (int i = 0; i < A.rows; ++i) {
        C.values.insert(C.values.end(), local_vals[i].begin(), local_vals[i].end());
        C.col_index.insert(C.col_index.end(), local_cols[i].begin(), local_cols[i].end());
        C.row_index.push_back(static_cast<int>(C.values.size()));
    }
    return C;
}
S