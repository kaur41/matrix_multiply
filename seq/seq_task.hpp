#pragma once
#include "../common/include/sparse_matrix.hpp"

class SparseMatrixTaskSequential {
private:
    CRSMatrix A;
    CRSMatrix B;
    CRSMatrix C;
    std::vector<double> temp_row; // Временный массив для сборки строки

public:
    // Конструктор принимает входные матрицы
    SparseMatrixTaskSequential(const CRSMatrix& in_A, const CRSMatrix& in_B);

    // Обязательные методы жизненного цикла задачи в курсе
    bool validation();
    bool pre_scheduling();
    bool run();
    bool post_scheduling();

    // Метод для получения результата
    CRSMatrix get_result() const;
};
