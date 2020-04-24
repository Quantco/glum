#include <stdbool.h>
#include "immintrin.h"

//TODO: copied pasted from dense
inline
double hsum_double_avx(__m256d v) 
{
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

void _sparse_sandwich(
        double* restrict Adata, int* restrict Aindices, int* restrict Aindptr,
        double* restrict ATdata, int* restrict ATindices, int* restrict ATindptr,
        double* d,
        double* out, 
        int m, int n, int nnz)
{
#pragma omp parallel for
    for (int j = 0; j < m; j++) {
        for (int A_idx = Aindptr[j]; A_idx < Aindptr[j+1]; A_idx++) {
            int k = Aindices[A_idx];
            double Aval = Adata[A_idx] * d[k];
            __m256d Avalavx = _mm256_set1_pd(Aval);

            int loop_end = ATindptr[k] + ((ATindptr[k+1] - ATindptr[k]) / 4) * 4;

            bool done = false;
            int AT_idx = ATindptr[k];
            for (; AT_idx < loop_end; AT_idx += 4) {
                int i = ATindices[AT_idx];
                if (i > j) {
                    done = true;
                    break;
                }

                __m256d ATvalavx = _mm256_loadu_pd(&ATdata[AT_idx]);
                __m256d mulavx = _mm256_mul_pd(ATvalavx, Avalavx);
                double mul[4];
                _mm256_store_pd(mul, mulavx);
                out[i * m + j] += mul[0];

                % for r in range(1, 4):
                {
                    int i = ATindices[AT_idx + ${r}];
                    if (i > j) {
                        done = true;
                        break;
                    }
                    out[i * m + j] += mul[${r}];
                }
                % endfor
            }
            if (!done) {
                for (; AT_idx < ATindptr[k+1]; AT_idx++) {
                    int i = ATindices[AT_idx];
                    if (i > j) {
                        break;
                    }
                    out[i * m + j] += Aval * ATdata[AT_idx];
                }
            }
        }
    }
}


