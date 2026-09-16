#include "tbb_task.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

SparseMatrixTaskTBB::SparseMatrixTaskTBB(const CRSMatrix& in_A, const CRSMatrix& in_B)
    : A(in_A), B(in_B) {
}

bool SparseMatrixTaskTBB::validation() {
    return A.cols == B.rows && A.rows > 0 && A.cols > 0 && B.cols > 0;
}

bool SparseMatrixTaskTBB::pre_scheduling() {
    C.rows = A.rows;
    C.cols = B.cols;
    C.values.clear();
    C.col_index.clear();
    C.row_index.clear();

    local_vals.assign(A.rows, std::vector<double>());
    local_cols.assign(A.rows, std::vector<int>());
    return true;
}

bool SparseMatrixTaskTBB::run() {
    // Вызываем параллельный цикл TBB по строкам матрицы А
    tbb::parallel_for(tbb::blocked_range<int>(0, A.rows), [&](const tbb::blocked_range<int>& r) {
        // Локальный аккумулятор текущего потока для выделенного поддиапазона строк
        std::vector<double> temp_row(B.cols, 0.0);

        for (int i = r.begin(); i != r.end(); ++i) {
            for (int k = A.row_index[i]; k < A.row_index[i + 1]; ++k) {
                int a_col = A.col_index[k];
                double a_val = A.values[k];

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
    return true;
}

bool SparseMatrixTaskTBB::post_scheduling() {
    C.row_index.push_back(0);
    // Финальная сборка CRS структуры
    for (int i = 0; i < A.rows; ++i) {
        C.values.insert(C.values.end(), local_vals[i].begin(), local_vals[i].end());
        C.col_index.insert(C.col_index.end(), local_cols[i].begin(), local_cols[i].end());
        C.row_index.push_back(static_cast<int>(C.values.size()));
    }
    return true;
}

CRSMatrix SparseMatrixTaskTBB::get_result() const {
    return C;
}
