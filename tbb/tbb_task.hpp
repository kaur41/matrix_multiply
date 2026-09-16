#pragma once
#include "../common/include/sparse_matrix.hpp"

class SparseMatrixTaskTBB {
private:
    CRSMatrix A;
    CRSMatrix B;
    CRSMatrix C;

    // Ћокальные буферы дл€ изол€ции потоков в TBB
    std::vector<std::vector<double>> local_vals;
    std::vector<std::vector<int>> local_cols;

public:
    SparseMatrixTaskTBB(const CRSMatrix& in_A, const CRSMatrix& in_B);

    bool validation();
    bool pre_scheduling();
    bool run();
    bool post_scheduling();

    CRSMatrix get_result() const;
};
