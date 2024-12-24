#include "ilu_gmres.h"
#include <cxxopts.h>
#include <fast_matrix_market/app/Eigen.hpp>
#include <fstream>

using namespace std;
using namespace Eigen;
using SpMat = SparseMatrix<val_t, RowMajor, idx_t>;

int main(int argc, char **argv) {
    cxxopts::Options options(argv[0], "ILUTP+GMRES solver");
    // clang-format off
    options.add_options()
        ("m,mat", "Coefficient matrix file (Matrix Market)", cxxopts::value<string>())
        ("b,rhs", "Right hand side vector (Matrix Market)", cxxopts::value<string>()->default_value("none"))
        ("o,optimization", "MKL optimization level", cxxopts::value<int>()->default_value("10"))
        ("p,pivot-tol", "ILUTP pivot tolerance", cxxopts::value<double>()->default_value("0"))
        ("f,fill-factor", "ILUTP fill factor", cxxopts::value<double>()->default_value("1"))
        ("e,extra-fill", "ILUTP extra fills", cxxopts::value<idx_t>()->default_value("0"))
        ("d,drop-tol", "ILUTP drop tolerance", cxxopts::value<double>()->default_value("1e-3"))
        ("n,num-neighbours", "MN maximum neighbours", cxxopts::value<idx_t>()->default_value("1"))
        ("t,tol", "GMRES tolerance", cxxopts::value<double>()->default_value("1e-6"))
        ("i,maxit", "GMRES max iterations", cxxopts::value<idx_t>()->default_value("10"))
        ("r,restart", "GMRES restart iterations", cxxopts::value<idx_t>()->default_value("100"))
        ("c,precond-type", "Precondition Method 0:None 1:ilutp 2:??", cxxopts::value<int>()->default_value("1"))
        ("h,help", "Print usage");
    // clang-format on
    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        printf("%s\n", options.help().c_str());
        return 0;
    }
    auto mat_path = args["mat"].as<string>();
    auto b_path = args["rhs"].as<string>();
    auto optimization = args["optimization"].as<int>();
    auto pivot_tol = args["pivot-tol"].as<double>();
    auto fill_factor = args["fill-factor"].as<double>();
    auto extra_fill = args["extra-fill"].as<idx_t>();
    auto drop_tol = args["drop-tol"].as<double>();
    auto num_neigh = args["num-neighbours"].as<idx_t>();
    auto tol = args["tol"].as<double>();
    auto maxit = args["maxit"].as<idx_t>();
    auto restart = args["restart"].as<idx_t>();
    auto precond_type = args["precond-type"].as<int>();

    // read matrix and generate rhs
    SpMat A;
    ifstream fin(mat_path);
    fast_matrix_market::read_matrix_market_eigen(fin, A, {.num_threads = 8});
    fin.close();
    auto n = (idx_t)A.rows();
    VectorXcd x = VectorXcd ::Ones(n), b = A * x, sol(n);
    if (b_path != "none") {
        ifstream fin(b_path);
        VectorXd _b;
        fast_matrix_market::read_matrix_market_eigen_dense(fin, _b, {.num_threads = 8});
        fin.close();
        b = _b;
    }
    cout << "matrix loaded" << endl;

    // solve with ILUTP+GMRES solver
    IluGmres solver;
    cout << "analyzing..." << endl;
    solver.analyze(n, A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), optimization);
    cout << "factorizing..." << endl;
    if (precond_type == PRECOND_ILUTP) {
        solver.factorize_ilutp(A.valuePtr(), pivot_tol, drop_tol, fill_factor, extra_fill);
    } else if (precond_type == PRECOND_MN) {
        solver.factorize_mn(A.valuePtr(), num_neigh);
    }
    cout << "solving..." << endl;
    auto gmres_result = solver.solve(b.data(), sol.data(), tol, maxit, restart);
    ofstream fout("sol.txt");
    fast_matrix_market::matrix_market_header header(sol.size(), 1);
    header.nnz = sol.size();
    write_matrix_market_array(fout, header, sol);

    // print summary
    auto stats = solver.get_stats();
    printf("\nInput summary\n");
    printf("matrix:                 %s\n", mat_path.c_str());
    printf("n:                      %ld\n", (int64_t)n);
    printf("nnz:                    %td\n", A.nonZeros());
    printf("ILUTP\n");
    printf("  fill factor:          %g\n", fill_factor);
    printf("  extra fill:           %ld\n", (int64_t)extra_fill);
    printf("  drop tol:             %g\n", drop_tol);
    printf("GMRES\n");
    printf("  tol:                  %g\n", tol);
    printf("  maxit:                %d\n", maxit);
    printf("  restart:              %ld\n", (int64_t)restart);
    printf("\nOutput summary\n");
    printf("ILUTP\n");
    printf("  number of pivots:     %ld\n", (int64_t)stats.npivots);
    printf("  nnz (LU):             %ld\n", (int64_t)stats.lu_nnz);
    printf("GMRES\n");
    printf("  success:              %d\n", gmres_result.success);
    printf("  relative residual:    %g\n", gmres_result.rel_res);
    printf("  iterations:           %ld\n", (int64_t)gmres_result.its);
    printf("  relative error:       %g\n", (sol - x).norm() / x.norm());
    printf("time\n");
    printf("  analyze:              %f\n", stats.analyze_time);
    printf("  factorize:            %f\n", stats.factorize_time);
    printf("  solve:                %f\n", stats.solve_time);
    return 0;
}