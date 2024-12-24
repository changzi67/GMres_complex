#pragma once
#ifndef LU_UTILS_H
#define LU_UTILS_H

#include <chrono>

inline thread_local std::chrono::steady_clock::time_point TIC;

inline std::chrono::steady_clock::time_point tic() {
    TIC = std::chrono::steady_clock::now();
    return TIC;
}

inline double toc(std::chrono::steady_clock::time_point tic_ = TIC) {
    auto toc = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic_).count();
}

template <typename idx_t>
inline void inv_perm(const idx_t *p, idx_t *ip, idx_t n) {
    for (idx_t i = 0; i < n; i++) {
        ip[p[i]] = i;
    }
}

#endif  // LU_UTILS_H
