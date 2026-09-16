#include "seq_task.hpp"

SparseMatrixTaskSequential::SparseMatrixTaskSequential(const CRSMatrix& in_A, const CRSMatrix& in_B)
    : A(in_A), B(in_B) {
}

bool SparseMatrixTaskSequential::validation() {
    // Число столбцов первой матрицы должно быть равно числу строк второй матрицы
    return A.cols == B.rows && A.rows > 0 && A.cols > 0 && B.cols > 0;
}

bool SparseMatrixTaskSequential::pre_scheduling() {
    C.rows = A.rows;
    C.cols = B.cols;
    C.values.clear();
    C.col_index.clear();
    C.row_index.clear();
    C.row_index.push_back(0);

    // Выделяем память под строку-аккумулятор
    temp_row.assign(B.cols, 0.0);
    return true;
}

bool SparseMatrixTaskSequential::run() {
    for (int i = 0; i < A.rows; ++i) {
        // Шаг 1: Скалярно умножаем элементы строки i матрицы A на соответствующие строки B
        for (int k = A.row_index[i]; k < A.row_index[i + 1]; ++k) {
            int a_col = A.col_index[k];
            double a_val = A.values[k];

            for (int j = B.row_index[a_col]; j < B.row_index[a_col + 1]; ++j) {
                temp_row[B.col_index[j]] += a_val * B.values[j];
            }
        }

        // Шаг 2: Извлекаем ненулевые результаты в результирующую матрицу C
        for (int j = 0; j < B.cols; ++j) {
            if (std::abs(temp_row[j]) > 1e-9) {
                C.values.push_back(temp_row[j]);
                C.col_index.push_back(j);
                temp_row[j] = 0.0; // Сразу зачищаем ячейку для следующих строк
            }
        }
        C.row_index.push_back(static_cast<int>(C.values.size()));
    }
    return true;
}

bool SparseMatrixTaskSequential::post_scheduling() {
    return true;
}

CRSMatrix SparseMatrixTaskSequential::get_result() const {
    return C;
}
