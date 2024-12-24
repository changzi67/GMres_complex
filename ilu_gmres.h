#pragma once
#ifndef ILU_GMRES_H
#define ILU_GMRES_H

#include <mkl.h>
#include <vector>
#include <complex>

using idx_t = int;
using val_t = std::complex<double>;
using VecI = std::vector<idx_t>;
using VecV = std::vector<val_t>;

enum PRECOND_TYPE {PRECOND_NONE=0, PRECOND_ILUTP=1, PRECOND_MN=2};

class IluGmres {
public:
    struct Stats {
        double analyze_time;
        double factorize_time;
        double solve_time;
        idx_t lu_nnz;
        idx_t npivots;
    };

    struct GmresResult {
        bool success;
        double rel_res;
        idx_t its;
        std::vector<double> res_vec;
    };

    IluGmres() = default;

    IluGmres(const IluGmres &) = delete;

    IluGmres &operator=(const IluGmres &) = delete;

    IluGmres(IluGmres &&) = delete;

    IluGmres &operator=(IluGmres &&) = delete;

    ~IluGmres() = default;

    // reorder the matrix with MC64 and amd
    void analyze(idx_t n, const idx_t *ap, const idx_t *ai, const val_t *ax, int optimization);

    // factorize the matrix with ILUTP
    void factorize_ilutp(const val_t *ax, double pivot_tol, double drop_tol, double fill_factor, idx_t extra_fill);

    // calculate MN(n) Preconditioner
    void factorize_mn(const val_t *ax, idx_t num_neigh);

    // solve Ax=b with GMRES
    GmresResult solve(const val_t *b, val_t *x, double tol, idx_t maxit, idx_t restart);

    [[nodiscard]] const Stats &get_stats() const;

private:
    void apply_precondition(const val_t *b, val_t *x, val_t *work);

    PRECOND_TYPE precond_type = PRECOND_NONE; // 0: None. 1: ILU. 2: ?
    idx_t n = 0;
    const idx_t *ap{}, *ai{};      // csr: ap(len n) row index; ai(len nnz) col index
    const val_t *ax{};             // csr: values
    VecI lup, lui;                 // (Now Also used to save P (CSC Format))
    VecV lux;
    sparse_matrix_t A_mkl = nullptr;
    VecI rp;   // row permutation: i-th row in permuted A is the rp[i]-th row in A
    VecI irp;  // inverse row permutation: i-th row in A is the irp[i]-th row in the permuted A
    VecI cp;   // column permutation: i-th column in the permuted A is the p[i]-th column in A
    VecI icp;  // inverse column permutation: i-th column in A is the ip[i]-th column in the permuted A
    std::vector<VecI> lu_indices;  // column indices of a row in L+U-I (order: U, L, diag)
    std::vector<VecV> lu_values;   // corresponding values
    VecI u_nnz;                    // number of non-zeros in U(i,:) (diagonal element excluded)
    VecI stack;                    // stack for dfs
    VecI visited;                  // dfs visited
    VecI iter_pos;                 // dfs iterate position
    VecI l_indices;                // non-zero indices in L(i,:) (diagonal element excluded)
    VecI u_indices;                // non-zero indices in U(i,:) (diagonal element included)
    VecV x;                        // dense vector for numerical update
    VecV gmres_work;               // work space for GMRES
    VecV mn_work;                  // work space for MN Precondition
    Stats stats{};
};

#endif  // ILU_GMRES_H
