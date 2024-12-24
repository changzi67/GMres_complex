#include "ilu_gmres.h"
#include <amd.h>
#include <mc64.h> 
#include <mkl.h>
#include <numeric>
#include "utils.h"
#define DEBUG

typedef MKL_Complex16 mkl_t;
using namespace std;

const matrix_descr A_descr{SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_FULL, SPARSE_DIAG_NON_UNIT};

void IluGmres::analyze(idx_t n_, const idx_t *ap_, const idx_t *ai_, const val_t *ax_, int optimization) {
    tic();
    n = n_;
    ap = ap_;
    ai = ai_;
    ax = ax_;
    rp.resize(n);
    irp.resize(n);
    cp.resize(n);
    icp.resize(n);
    lu_indices.resize(n);
    lu_values.resize(n);
    u_nnz.resize(n);
    stack.resize(n);
    visited.resize(n);
    iter_pos.resize(n);
    l_indices.resize(n);
    u_indices.resize(n);
    x.assign(n, 0.0);
    memset(&stats, 0, sizeof(stats));

    // mkl optimization
    if (A_mkl != nullptr) {
        mkl_sparse_destroy(A_mkl);
    }
    mkl_sparse_z_create_csr(&A_mkl, SPARSE_INDEX_BASE_ZERO, n, n, const_cast<idx_t *>(ap), const_cast<idx_t *>(ap + 1),
                            const_cast<idx_t *>(ai), reinterpret_cast<mkl_t *>(const_cast<val_t *>(ax)));
    if (optimization > 0) {
        mkl_sparse_set_mv_hint(A_mkl, SPARSE_OPERATION_NON_TRANSPOSE, A_descr, optimization);
        mkl_sparse_optimize(A_mkl);
    }

    // mc64
    idx_t job = 5, nnz = ap[n], matched, liw = 5 * n, ldw = 3 * n + nnz;
    VecI iw(liw);
    std::vector<double> dw(ldw);
    double *ax_d = new double[nnz];
    int icntl[10], info[10];
    mc64id(icntl);
    for (idx_t m = 0; m <= n; m++) {
        const_cast<idx_t *>(ap)[m]++;
    }
    for (idx_t m = 0; m < nnz; m++) {
        const_cast<idx_t *>(ai)[m]++;
    }
    for (idx_t m = 0; m < nnz; m++) {
        ax_d[m] = std::abs(ax[m]);
    }
    mc64ad(&job, &n, &nnz, const_cast<idx_t *>(ap), const_cast<idx_t *>(ai), ax_d, &matched,
           icp.data(), &liw, iw.data(), &ldw, dw.data(), icntl, info);
    if (matched != n) {
        throw runtime_error("static pivoting failed");
    }
    delete[] ax_d;
    for (idx_t m = 0; m <= n; m++) {
        const_cast<idx_t *>(ap)[m]--;
    }
    for (idx_t m = 0; m < nnz; m++) {
        const_cast<idx_t *>(ai)[m]--;
    }
    for (idx_t i = 0; i < n; i++) {
        icp[i]--;
    }
    inv_perm(icp.data(), cp.data(), n);

    // amd
    for (idx_t m = 0; m < nnz; m++) {
        const_cast<idx_t *>(ai)[m] = icp[ai[m]];
    }
    amd_order(n, ap, ai, rp.data(), nullptr, nullptr);
    for (idx_t m = 0; m < nnz; m++) {
        const_cast<idx_t *>(ai)[m] = cp[ai[m]];
    }
    inv_perm(rp.data(), irp.data(), n);
    for (idx_t i = 0; i < n; i++) {
        icp[i] = irp[icp[i]];
    }
    inv_perm(icp.data(), cp.data(), n);
    stats.analyze_time = toc();
#ifdef DEBUG
    if(n <= 50) {
        printf("\nRow Permute: ");
        for (auto x:rp) printf("%d ", x);
        printf("\nCol Permute: ");
        for (auto x:cp) printf("%d ", x);
        puts("");
        std::vector<VecV> denseMatrix(n, VecV(n, 0.0));
        for (idx_t row = 0; row < n; ++row) {
            for (idx_t idx = ap[row]; idx < ap[row + 1]; ++idx) {
                idx_t col = ai[idx];
                val_t value = ax[idx];
                denseMatrix[irp[row]][icp[col]] = value;
            }
        }
        puts("Dense Matrix Representation:");
        for (const auto& row : denseMatrix) {
            for (const auto& value : row) {
                printf("%8.6f+%8.6fj ", value.real(), value.imag());
            }
            puts("");
        }
    }
#endif
}

void IluGmres::factorize_ilutp(const val_t *ax_, double pivot_tol, double drop_tol, double fill_factor,
                               idx_t extra_fill) {
    tic();
    precond_type = PRECOND_ILUTP;
    ax = ax_;
    fill_n(u_nnz.begin(), n, 0);
    fill_n(visited.begin(), n, -1);
    stats.npivots = 0;
    for (idx_t i = 0; i < n; i++) { // for every row of A
        if((i+1)*10 / n > i*10 / n) printf("%d0%% ...\n", (i+1)*10/n);
        idx_t org_i = rp[i], top = n, l_head = n, u_head = n; // org_i (original i in A), top (stack top)

        // predict non-zero positions
        for (idx_t m = ap[org_i]; m < ap[org_i + 1]; m++) { // for every non zero in row
            idx_t org_k = ai[m];
            x[org_k] = ax[m];
            if (visited[org_k] == i) {
                continue;
            }
            stack[--top] = org_k;
            while (top != n) { // stack not empty
                org_k = stack[top];
                idx_t k = icp[org_k];
                if (visited[org_k] != i) {  // not visited
                    visited[org_k] = i;
                    iter_pos[org_k] = u_nnz[k];
                }
                idx_t pos = iter_pos[org_k] - 1;
                for (; pos >= 0; pos--) {
                    idx_t org_j = lu_indices[k][pos];
                    if (visited[org_j] != i) {  // save the running state and move to the next child
                        iter_pos[org_k] = pos;
                        stack[--top] = org_j;
                        break;
                    }
                }
                if (pos < 0) {  // all children visited
                    top++;
                    if (i > k) {
                        l_indices[--l_head] = org_k;
                    } else {
                        u_indices[--u_head] = org_k;
                    }
                }
            }
        }

        // numerical update
        for (idx_t m = l_head; m < n; m++) {
            idx_t org_k = l_indices[m], k = icp[org_k], *indices = lu_indices[k].data();
            val_t *values = lu_values[k].data();
            x[org_k] /= lu_values[k].back();
            val_t xk = x[org_k];
            for (idx_t t = 0; t < u_nnz[k]; t++) {
                x[indices[t]] -= values[t] * xk;
            }
        }

        // drop elements
        double tol = 0;
        idx_t l_keep = 0, u_keep = 0;
        for (idx_t m = ap[org_i]; m < ap[org_i + 1]; m++) {
            if (abs(ax[m]) < 1e3) {
                tol += (ax[m] * std::conj(ax[m])).real();
            }
            if (i > icp[ai[m]]) {
                l_keep++;
            } else {
                u_keep++;
            }
        }
        tol = drop_tol * sqrt(tol);
        l_keep = idx_t(fill_factor * l_keep) + extra_fill;
        u_keep = idx_t(fill_factor * u_keep) + extra_fill + 1;
        auto l_tail =
            partition(&l_indices[l_head], &l_indices[n], [&](idx_t k) { return abs(x[k]) >= tol; }) - l_indices.data();
        if (l_tail - l_head > l_keep) {
            nth_element(&l_indices[l_head], &l_indices[l_head + l_keep], &l_indices[l_tail],
                        [&](idx_t k1, idx_t k2) { return abs(x[k1]) > abs(x[k2]); });
            l_tail = l_head + l_keep;
        }
        idx_t diag_index = cp[i];
        auto u_tail =
            partition(&u_indices[u_head], &u_indices[n], [&](idx_t k) { return abs(x[k]) >= tol || k == diag_index; }) -
            u_indices.data();
        if (u_tail - u_head > u_keep) {
            nth_element(&u_indices[u_head], &u_indices[u_head + u_keep], &u_indices[u_tail], [&](idx_t k1, idx_t k2) {
                if (k1 == diag_index) {
                    return true;
                }
                if (k2 == diag_index) {
                    return false;
                }
                return abs(x[k1]) > abs(x[k2]);
            });
            u_tail = u_head + u_keep;
        }

        // partial pivoting
        idx_t max_index = -1;
        double max_val = -1;
        for (idx_t m = u_head; m < u_tail; m++) {
            idx_t org_k = u_indices[m];
            double val = abs(x[org_k]);
            if (val > max_val) {
                max_val = val;
                max_index = org_k;
            }
        }
        idx_t pivot_index = abs(x[diag_index]) >= pivot_tol * max_val ? diag_index : max_index;
        if (x[pivot_index] == 0.0) {
            throw runtime_error("numerical error");
        }
        if (diag_index != pivot_index) {
            swap(icp[pivot_index], icp[diag_index]);
            swap(cp[icp[pivot_index]], cp[icp[diag_index]]);
            stats.npivots++;
        }

        // gather
        val_t pivot_val = x[pivot_index];
        auto nnz = l_tail - l_head + u_tail - u_head;
        lu_indices[i].clear();
        lu_indices[i].reserve(nnz);
        lu_values[i].clear();
        lu_values[i].reserve(nnz);
        for (idx_t m = u_head; m < u_tail; m++) {
            idx_t k = u_indices[m];
            if (k != pivot_index) {
                lu_indices[i].push_back(k);
                lu_values[i].push_back(x[k]);
            }
        }
        u_nnz[i] = (idx_t)lu_indices[i].size();
        for (idx_t m = l_head; m < l_tail; m++) {
            idx_t k = l_indices[m];
            lu_indices[i].push_back(k);
            lu_values[i].push_back(x[k]);
        }
        lu_indices[i].push_back(pivot_index);
        lu_values[i].push_back(pivot_val);

        // reset
        for (idx_t m = l_head; m < n; m++) {
            x[l_indices[m]] = 0;
        }
        for (idx_t m = u_head; m < n; m++) {
            x[u_indices[m]] = 0;
        }
#ifdef DEBUG
        if (n < 200) {
            cout << "i = " << i << ", l_nnz = " << lu_indices[i].size() - u_nnz[i] << ", u_nnz = " << u_nnz[i]
                 << "\nL:";
            for (idx_t m = u_nnz[i]; m < (idx_t)lu_indices[i].size(); m++) {
                cout << " (" << icp[lu_indices[i][m]] << ", " << lu_values[i][m] << ")";
            }
            cout << "\nU:";
            for (idx_t m = 0; m < u_nnz[i]; m++) {
                cout << " (" << icp[lu_indices[i][m]] << ", " << lu_values[i][m] << ")";
            }
            cout << "\n";
            cout << "pivot: " << icp[pivot_index] << " <-> " << icp[diag_index] << "\n\n";
        }
#endif
    }
    stats.lu_nnz = accumulate(lu_indices.begin(), lu_indices.end(), 0,
                              [](idx_t sum, const auto &row) { return sum + row.size(); });
    lup.resize(n + 1);
    lui.resize(stats.lu_nnz);
    lux.resize(stats.lu_nnz);
    lup[0] = 0;
    inclusive_scan(
        lu_indices.begin(), lu_indices.end(), &lup[1], [](idx_t sum, const auto &row) { return sum + row.size(); }, 0);
    for (idx_t i = 0; i < n; i++) {
        idx_t *ptr = &lui[lup[i]];
        for (idx_t m = 0; m < (idx_t)lu_indices[i].size(); m++) {
            ptr[m] = icp[lu_indices[i][m]];
        }
        copy(lu_values[i].begin(), lu_values[i].end(), &lux[lup[i]]);
    }
    stats.factorize_time += toc();
}

void IluGmres::factorize_mn(const val_t *ax_, idx_t num_neigh) {
    tic();
    precond_type = PRECOND_MN;
    ax = ax_;

    // aL: Submatrix of A, L indiced.
    // bL: rhs of Equation. L indiced.
    // Lx, Li, Ln: the List of Neighbours. Lx for value, Li for indice. Ln for num.
    mn_work.resize((num_neigh + 1) * (num_neigh + 2));
    val_t *aL=mn_work.data(), *bL = aL + (num_neigh + 1) * (num_neigh + 1);
    idx_t Ln=1;
    VecI &Li = stack, &ipiv = visited; // Temporary Use of ununsed vectors

    // Clear Cache
    lup.clear(); 
    lui.clear();
    lux.clear();

    for (idx_t i = 0; i < n; i++) { // for every row in A
        idx_t org_i = rp[i]; 
        val_t diag_value(0.0);
        
        // Find Values of all ith row
        Ln = 1;
        for (idx_t m = ap[org_i]; m < ap[org_i + 1]; m++) {
            idx_t org_k = ai[m], k = icp[org_k];
            if (k == i) {
                diag_value = ax[m];
                Li[0] = m;// Push Diagonal Element to stack.
            } else {
                Li[Ln++] = m;
            }
        }

        // Select Neighbours (allow to have n neighbours + 1 diagonal element)
        if (Ln > num_neigh + 1) {
            nth_element(Li.begin() + 1, Li.begin() + 1 + num_neigh, Li.begin() + Ln, 
                        [&](const idx_t &a, const idx_t &b) { return abs(ax[a]) < abs(ax[b]); });
            Ln = num_neigh + 1;
        }
        // for (idx_t li = 0; li < Ln; li++) cout << "(" << Li[li] << ", " << icp[ai[Li[li]]] << ")"; 
        // cout << "\n";

        // Gather Elements in A
        fill_n(aL, Ln * Ln, val_t(0.0)); // clean aL
        for (idx_t li = 0; li < Ln; li++) { // li: l indiced i
            idx_t i = icp[ai[Li[li]]], org_i = rp[i];
            bL[li] = (li == 0 ? val_t(1.0) : val_t(0.0));
            for (idx_t m = ap[org_i]; m < ap[org_i + 1]; m++) {
                idx_t org_k = ai[m];
                // cout << "V: " << irp[org_i] << ' ' << icp[org_k] <<' ' << ax[m] << endl;
                idx_t lk = find_if(Li.begin(), Li.begin() + Ln, [&](idx_t a){ return ai[a] == org_k; }) // lk: l indiced k
                           - Li.begin(); // Find Column k in L list 
                if (lk != Ln) {
                    // cout << "A: " << li << ' ' << lk <<' ' << ax[m] << endl;
                    aL[li * Ln + lk] = ax[m];
                }
            }
        }

        // Solve Linear System (Fast Solve for Ln <= 2)
        if (Ln == 1) {
            bL[0] = val_t(1.0) / aL[0];
        } else if (Ln == 2) {
            val_t det = aL[0] * aL[3] - aL[1] * aL[2];
            bL[0] = aL[3] / det;
            bL[1] = -aL[2] / det;
        } else {
            LAPACKE_zgesv(LAPACK_ROW_MAJOR, Ln, 1, reinterpret_cast<mkl_t *>(aL), Ln,
                          ipiv.data(), reinterpret_cast<mkl_t *>(bL), 1);
        }

        // Pack Results to P
        lup.push_back((idx_t) lui.size());
        for (idx_t li = 0; li < Ln; li++) if(abs(bL[li]) > 1e-10) {
            lui.push_back(icp[ai[Li[li]]]);
            lux.push_back(bL[li]);
        }
        
#ifdef DEBUG
        if(n < 200) {
            cout << "i = " << i << "\nP:";
            for (idx_t li = lup[i]; li < (idx_t)lui.size(); li++) {
                cout << " (" << lui[li] << ", " << lux[li] << ")";
            }
            cout << "\n\n";
        }
#endif
    }
    lup.push_back((idx_t) lui.size());

    stats.factorize_time += toc();
}

IluGmres::GmresResult IluGmres::solve(const val_t *b, val_t *sol, double tol, idx_t maxit, idx_t restart) {
    // initialize
    tic();
    idx_t h_dim = restart + 1;
    gmres_work.resize((n + h_dim) * (restart + 3));
    val_t *V_data = gmres_work.data(), *H_data = V_data + n * (restart + 3);
#define V(i, j) V_data[(i) + (j) * n]
#define H(i, j) H_data[(i) + (j) * h_dim]
    val_t *work = &V(0, h_dim), *tmp = work + n;
    val_t *c = &H(0, restart), *s = c + h_dim, *d = s + h_dim;
    val_t alpha, beta;
    fill_n(sol, n, 0.0);
    const double b_norm = cblas_dznrm2(n, b, 1);
    tol *= b_norm;
    double res;
    vector<double> res_vec;
    res_vec.reserve(maxit * restart);

    // gmres
    idx_t its = 0;
    for (idx_t outer = 0; outer < maxit; outer++) {
        std::chrono::steady_clock::time_point Arnoldi_tic;
        tic(Arnoldi_tic);
        // initial residual
        val_t *v = &V(0, 0), *h = &H(0, 0);
        copy_n(b, n, v);
        mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, mkl_t{-1.0, 0.0}, A_mkl, A_descr, 
                        reinterpret_cast<mkl_t *>(sol), mkl_t{1.0, 0.0}, reinterpret_cast<mkl_t *>(v));  // r = b - A * x
        res = cblas_dznrm2(n, v, 1);
        d[0] = val_t(res, 0);
        res_vec.push_back(res);
        printf("Iter [%d/%d] Res %.6le ...\n", outer, maxit, res);
        if (res <= tol) {
            break;
        }
        cblas_zdscal(n, 1 / res, v, 1);

        // Krylov loop
        idx_t j = 0;
        for (; j < restart && res > tol; j++, h += h_dim) {
            // apply preconditioner
            its++;
            apply_precondition(v, tmp, work); // v = M ^ -1 * v
            v += n;

            // Arnoldi process
            mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE, mkl_t{1.0, 0.0}, A_mkl, A_descr,
                           reinterpret_cast<mkl_t *>(tmp), mkl_t{0.0, 0.0}, reinterpret_cast<mkl_t *>(v));  // r = A * r;
            val_t *vi = &V(0, 0);
            for (idx_t i = 0; i <= j; i++, vi += n) {
                cblas_zdotc_sub(n, vi, 1, v, 1, &h[i]);   // H(i, j) = V(:,i) * r
                alpha = -h[i];
                cblas_zaxpy(n, &alpha, vi, 1, v, 1);  // r -= H(i, j) * V(:,i)
            }
            h[j + 1] = val_t(cblas_dznrm2(n, v, 1), 0);     // H(j + 1, j) = ||r||//
            cblas_zdscal(n, 1 / h[j + 1].real(), v, 1);  // V(:,j + 1) = r / ||r||

            // transform H into triangular form with Givens rotation
            for (int i = 0; i < j; i++) {
                val_t temp = h[i];
                h[i] = c[i] * temp - conj(s[i]) * h[i + 1];
                h[i + 1] = s[i] * temp + conj(c[i]) * h[i + 1];
            }
            double len = max(sqrt(conj(h[j]) * h[j] + conj(h[j + 1]) * h[j + 1]).real(), numeric_limits<double>::epsilon());
            c[j] = conj(h[j]) / len;
            s[j] = -h[j + 1] / len;
            h[j] = c[j] * h[j] - conj(s[j]) * h[j + 1];
            d[j + 1] = s[j] * d[j];
            d[j] *= c[j];
            res = abs(d[j + 1]);
            // cout<< "Outer " << outer << " Inner " << j << " Res " << res <<endl;
            res_vec.push_back(res);
            if (isnan(res) || isinf(res)) {
                return {false, res, its, res_vec};
            }
        }
        stats.Arnoldi_time += toc(Arnoldi_tic);

        std::chrono::steady_clock::time_point backward_tic;
        tic(backward_tic);

        // solve H * y = d for the coefficient of the linear combination
        cblas_ztrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, j, H_data, h_dim, d, 1); // Solve H * y = d
        alpha = val_t(1); beta = val_t(0);
        cblas_zgemv(CblasColMajor, CblasNoTrans, n, j, &alpha, V_data, n, d, 1, &beta, tmp, 1); // Calc update = V * y
        stats.backward_time += toc(backward_tic);

        std::chrono::steady_clock::time_point apply_precond_tic;
        tic(apply_precond_tic);
        apply_precondition(tmp, tmp, work);
        stats.apply_precond_time += toc(apply_precond_tic);
        vzAdd(n, reinterpret_cast<mkl_t*>(sol), reinterpret_cast<mkl_t*>(tmp), reinterpret_cast<mkl_t*>(sol));
    }
#undef V
#undef H
    stats.solve_time += toc();
    return {res <= tol, res / b_norm, its, res_vec};
}

const IluGmres::Stats &IluGmres::get_stats() const { return stats; }

void IluGmres::apply_precondition(const val_t *b, val_t *x, val_t *work) {
    if (precond_type == PRECOND_NONE) { // copy b to x
        for (idx_t i = 0; i < n; i++) x[i] = b[i];
    }
    else if (precond_type == PRECOND_ILUTP) { // calculate x = U^-1 L^-1 b
        for (idx_t i = 0; i < n; i++) {
            work[i] = b[rp[i]];
        }
        for (idx_t i = 0; i < n; i++) {
            val_t sum = 0;
            idx_t *indices = &lui[lup[i]];
            val_t *values = &lux[lup[i]];
            for (idx_t m = u_nnz[i]; m < lup[i + 1] - lup[i] - 1; m++) {
                sum += work[indices[m]] * values[m];
            }
            work[i] -= sum;
        }
        for (idx_t i = n - 1; i >= 0; i--) {
            val_t sum = 0;
            idx_t *indices = &lui[lup[i]];
            val_t *values = &lux[lup[i]];
            for (idx_t m = 0; m < u_nnz[i]; m++) {
                sum += work[indices[m]] * values[m];
            }
            work[i] = (work[i] - sum) / lux[lup[i + 1] - 1];
        }
        for (idx_t i = 0; i < n; i++) {
            x[i] = work[icp[i]];
        }
    } else if (precond_type == PRECOND_MN) { // calculate x = Pb
        for (idx_t i = 0; i < n; i++) work[i] = 0;
        for (idx_t i = 0; i < n; i++) {
            idx_t *indices = &lui[lup[i]];
            val_t *values = &lux[lup[i]];
            val_t mul = b[rp[i]];
            for (idx_t m = 0; m < lup[i + 1] - lup[i]; m++) {
                work[indices[m]] += values[m] * mul;
            }
        }
        for (idx_t i = 0; i < n; i++) {
            x[i] = work[icp[i]];
        }
    }
}
