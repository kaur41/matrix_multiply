#pragma once
#include <vector>

struct CRSMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> values;
    std::vector<int> col_index;
    std::vector<int> row_index;
};

CRSMatrix generate_random_sparse_matrix(int rows, int cols, double density);
CRSMatrix multiply_sequential(const CRSMatrix& A, const CRSMatrix& B);
CRSMatrix multiply_openmp(const CRSMatrix& A, const CRSMatrix& B);
CRSMatrix multiply_tbb(const CRSMatrix& A, const CRSMatrix& B);
bool compare_matrices(const CRSMatrix& A, const CRSMatrix& B);
