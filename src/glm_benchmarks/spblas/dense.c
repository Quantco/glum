#include "immintrin.h"

void dense_C(double* X, double* d, double* out, int m, int n) {
    int kblocksize = 200;
    int nkblocks = n / kblocksize;
    
    int jblocksize = 16;
    int njblocks = m / jblocksize;
    
    int iblocksize = 16;
    int niblocks = m / iblocksize;

    #pragma omp parallel for
    for (int idx = 0; idx < niblocks * njblocks; idx++) {
        int ib = idx / njblocks;
        int jb = idx % njblocks;

        int min_i = ib * iblocksize;
        int max_i = (ib + 1) * iblocksize;
        if (max_i > m) {
            max_i = m;
        }
        int min_j = jb * jblocksize;
        for (int kb = 0; kb < nkblocks; kb++) {
            int min_k = kb * kblocksize;
            int max_k = (kb + 1) * kblocksize;
            if (max_k > n) {
                max_k = n;
            }
            for (int i = min_i; i < max_i; i++) {
                int max_j = (jb + 1) * jblocksize;
                if (max_j > i + 1) {
                    max_j = i + 1;
                }
                
                if (max_k == (kb + 1) * kblocksize) {
                    for (int j = min_j; j < max_j; j++) {
                        __m256d accumavx = _mm256_set1_pd(0.0);
                        for(int k = min_k; k < max_k; k += 4) {
                            __m256d XTavx = _mm256_loadu_pd(&X[i * n + k]);
                            __m256d Xavx = _mm256_loadu_pd(&X[j * n + k]);
                            __m256d davx = _mm256_loadu_pd(&d[k]);
                            accumavx = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(XTavx, davx), Xavx), accumavx);
                        }
                        double output[4];
                        _mm256_store_pd(output, accumavx);
                        out[i * m + j] += output[0] + output[1] + output[2] + output[3];
                    }
                } else {
                    for (int j = min_j; j < max_j; j++) {
                        double accum = 0;
                        for (int k = min_k; k < max_k; k++) {
                            accum += X[i * n + k] * d[k] * X[j * n + k];
                        }
                        out[i * m + j] += accum;
                    }
                }
            }
        }
    }
}
