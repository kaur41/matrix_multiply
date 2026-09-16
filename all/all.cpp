#include "../common/include/sparse_matrix.hpp"
#include <random>

CRSMatrix generate_random_sparse_matrix(int rows, int cols, double density) {
    CRSMatrix M;
    M.rows = rows;
    M.cols = cols;
    M.row_index.push_back(0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_val(1.0, 5.0);
    std::uniform_real_distribution<double> dis_prob(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Если случайная величина меньше плотности, элемент становится ненулевым
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

    // Сравниваем double значения с эпсилон-погрешностью
    for (size_t i = 0; i < A.values.size(); ++i) {
        if (std::abs(A.values[i] - B.values[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}
