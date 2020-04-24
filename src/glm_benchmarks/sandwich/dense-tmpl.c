#include <stdbool.h>
#include "immintrin.h"

<%def name="inner_kj_avx(JBLOCK)">
    % for r in range(JBLOCK):
        __m256d accumavx${r};
        accumavx${r} = _mm256_set1_pd(0.0);
    % endfor
    for(int k = kmin; k < kmaxavx; k += KVECTOR) {
        __m256d XTavx = _mm256_loadu_pd(&X[i * n + k]);
        __m256d davx = _mm256_loadu_pd(&d[k]);
        __m256d Xtd = _mm256_mul_pd(XTavx, davx);
        % for r in range(JBLOCK):
            __m256d Xavx${r} = _mm256_loadu_pd(&X[(j + ${r}) * n + k]);
            accumavx${r} = _mm256_add_pd(_mm256_mul_pd(Xtd, Xavx${r}), accumavx${r});
        % endfor
    }
    % for r in range(JBLOCK):
        double accum${r} = hsum_double_avx(accumavx${r});
    % endfor
    for (int k = kmaxavx; k < kmax; k++) {
        double Q = X[i * n + k] * d[k];
        % for r in range(JBLOCK):
            accum${r} += Q * X[(j + ${r}) * n + k];
        % endfor
    }
    % for r in range(JBLOCK):
        out[i * m + (j + ${r})] += accum${r};
    % endfor
</%def>

#define KVECTOR 4
void dense_base(double* restrict X, double* restrict d, double* restrict out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) 
{
    for (int i = imin; i < imax; i++) {
        int jmaxinner = jmax;
        if (jmaxinner > i + 1) {
            jmaxinner = i + 1;
        }
        int kmaxavx = kmin + ((kmax - kmin) / KVECTOR) * KVECTOR;
        int j = jmin;
        % for JBLOCK in [8, 4, 2, 1]:
            for (; j < jmin + ((jmaxinner - jmin) / ${JBLOCK}) * ${JBLOCK}; j += ${JBLOCK}) {
                ${inner_kj_avx(JBLOCK)}
            }
        % endfor
    }
}

void recurse_ij(double* restrict X, double* restrict d, double* restrict out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) 
{
    int size = (imax - imin) * (jmax - jmin);
    bool parallel = size >= 256;
    if (!parallel) {
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
    int ksplit = (kmax + kmin) / 2;
    {
        // guaranteed to be partially in lower triangle
        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, isplit, jmin, jsplit, kmin, kmax);

        // guaranteed to be partially in lower triangle
        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, isplit, jsplit, jmax, kmin, kmax);

        // guaranteed to be partially in lower triangle
        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, isplit, imax, jmin, jsplit, kmin, kmax);

        // check if any of the entries are in the lower triangle
        if (jsplit <= imax) {
            #pragma omp task if(parallel)
            recurse_ij(X, d, out, m, n, isplit, imax, jsplit, jmax, kmin, kmax);
        }
    }
}

void _dense_sandwich(double* restrict X, double* restrict d, double* restrict out,
        int m, int n) 
{
    #pragma omp parallel
    #pragma omp single nowait
    recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n);
}
