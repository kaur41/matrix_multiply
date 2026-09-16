#pragma once
#include <vector>
#include <cmath>

// Структура разреженной матрицы в формате CRS (Compressed Row Storage)
struct CRSMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> values;     // Ненулевые значения
    std::vector<int> col_index;     // Номера столбцов для каждого элемента
    std::vector<int> row_index;     // Индексы начала строк (размер: rows + 1)
};

// Функция для генерации случайной разреженной матрицы (плотность density от 0.0 до 1.0)
CRSMatrix generate_random_sparse_matrix(int rows, int cols, double density);

// Функция сравнения двух матриц с учетом погрешности double (для валидации результатов)
bool compare_matrices(const CRSMatrix& A, const CRSMatrix& B);
