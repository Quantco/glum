#include <stdbool.h>
#include "immintrin.h"

inline
double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

void dense_base(double* restrict X, double* restrict d, double* restrict out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) {
    for (int i = imin; i < imax; i++) {
        int jmaxinner = jmax;
        if (jmaxinner > i + 1) {
            jmaxinner = i + 1;
        }
        int kmaxavx = kmin + ((kmax - kmin) / 4) * 4;
        for (int j = jmin; j < jmaxinner; j++) {
            __m256d accumavx = _mm256_set1_pd(0.0);
            for(int k = kmin; k < kmaxavx; k += 4) {
                __m256d XTavx = _mm256_loadu_pd(&X[i * n + k]);
                __m256d Xavx = _mm256_loadu_pd(&X[j * n + k]);
                __m256d davx = _mm256_loadu_pd(&d[k]);
                accumavx = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(XTavx, davx), Xavx), accumavx);
            }
            double accum = hsum_double_avx(accumavx);
            for (int k = kmaxavx; k < kmax; k++) {
                accum += X[i * n + k] * d[k] * X[j * n + k];
            }
            out[i * m + j] += accum;
        }
    }
}

void recurse_ij(double* restrict X, double* restrict d, double* restrict out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) {
    int size = (imax - imin) * (jmax - jmin);
    bool parallel = size >= 256;
    if (size < 16) {
        int kstep = 200;
        for (int kstart = kmin; kstart < kmax; kstart += kstep) {
            int kend = kstart + kstep;
            if (kend > kmax) {
                kend = kmax;
            }
            dense_base(X, d, out, m, n, imin, imax, jmin, jmax, kstart, kend);
        }
        return;
    }
    int isplit = (imax + imin) / 2;
    int jsplit = (jmax + jmin) / 2;

    {
        // TODO: don't recurse if we're outside the range of the lower triangle?
#pragma omp task if(parallel)
        {
            dense_recurse_ij(X, d, out, m, n, imin, isplit, jmin, jsplit, kmin, kmax);
        }

#pragma omp task if(parallel)
        {
            dense_recurse_ij(X, d, out, m, n, imin, isplit, jsplit, jmax, kmin, kmax);
        }

#pragma omp task if(parallel)
        {
            dense_recurse_ij(X, d, out, m, n, isplit, imax, jmin, jsplit, kmin, kmax);
        }

#pragma omp task if(parallel)
        {
            dense_recurse_ij(X, d, out, m, n, isplit, imax, jsplit, jmax, kmin, kmax);
        }
    }
    /* if (isize >= jsize && isize >= ksize) { */
    /*     int isplit = (imax + imin) / 2; */
    /*     dense_recurse(X, d, out, m, n, imin, isplit, jmin, jmax, kmin, kmax); */
    /*     dense_recurse(X, d, out, m, n, isplit, imax, jmin, jmax, kmin, kmax); */
    /* } else if (jsize >= isize && jsize >= ksize) { */
    /*     int jsplit = (jmax + jmin) / 2; */
    /*     dense_recurse(X, d, out, m, n, imin, imax, jmin, jsplit, kmin, kmax); */
    /*     dense_recurse(X, d, out, m, n, imin, imax, jsplit, jmax, kmin, kmax); */
    /* } else { */
    /*     int ksplit = (kmax + kmin) / 2; */
    /*     dense_recurse(X, d, out, m, n, imin, imax, jmin, jmax, kmin, ksplit); */
    /*     dense_recurse(X, d, out, m, n, imin, imax, jmin, jmax, ksplit, kmax); */
    /* } */
}

void dense_C2(double* restrict X, double* restrict d, double* restrict out, int m, int n) {
    #pragma omp parallel
    #pragma omp single nowait
    dense_recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n);
    // important for X to be in fortran ordering so the innermost loop has good spatial locality
    // int kblocksize = 200;
    // int nkblocks = n / kblocksize;
    // 
    // int jblocksize = 16;
    // int njblocks = m / jblocksize;
    // 
    // int iblocksize = 16;
    // int niblocks = m / iblocksize;

    // for (int idx = 0; idx < niblocks * njblocks; idx++) {
    //     int ib = idx / njblocks;
    //     int jb = idx % njblocks;

    //     int min_i = ib * iblocksize;
    //     int max_i = (ib + 1) * iblocksize;
    //     if (max_i > m) {
    //         max_i = m;
    //     }
    //     int min_j = jb * jblocksize;
    //     int max_j = (jb + 1) * jblocksize;
    //     for (int kb = 0; kb < nkblocks; kb++) {
    //         int min_k = kb * kblocksize;
    //         int max_k = (kb + 1) * kblocksize;
    //         if (max_k > n) {
    //             max_k = n;
    //         }
    //         dense_base(X, d, out, m, n, min_i, max_i, min_j, max_j, min_k, max_k);
    //     }
    // }
}
