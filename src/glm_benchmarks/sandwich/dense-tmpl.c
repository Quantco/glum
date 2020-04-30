// AVX code for dense sandwich products. Uses Mako templating for inner loop unrolling.
// Mako compiles this to C.
// Algorithm pseudocode:
// recurse_ij(X, d, out, i_min, i_max, j_min, j_max):
//     base case, if (i_max - i_min) * (j_max - j_min) small:
//         dense_base(...)
//     else:
//        split X into four blocks by changing i_min and i_max
//        call recurse_ij on each block
#include <stdbool.h>
#include "immintrin.h"

// copied from https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx/49943540#49943540
// a +  b + c + d = (a + b) + (c + d) = ((a) + (b)) + ((c) + (d))
inline
double hsum_double_avx(__m256d v) 
{
    __m128d vlow  = _mm256_castpd256_pd128(v); // get (a,b)
    __m128d vhigh = _mm256_extractf128_pd(v, 1); //get (c,d) // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // compute (a+c,b+d)// reduce down to 128
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

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
    //out[i,j] = sum_k X[k,i] * d[k] * X[k,j]
    recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n);
}


void _sparse_dense_sandwich(
    double* restrict Adata, int* restrict Aindices, int* restrict Aindptr,
    double* restrict B,
    double* restrict d,
    double* restrict out,
    int m, int n, int r) 
{
    <% 
        JBLOCK = 4
    %>
    int ravxmax = (r / ${JBLOCK}) * ${JBLOCK};
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int A_idx = Aindptr[i];
        int A_idx_max = Aindptr[i+1];
        for (; A_idx < A_idx_max; A_idx++) {
            int k = Aindices[A_idx];
            double Q = Adata[A_idx] * d[k];
            __m256d Qavx = _mm256_set1_pd(Q);
            int j = 0;
            for (; j < ravxmax; j+=4) {
                __m256d Bavx = _mm256_loadu_pd(&B[k*r+j]);
                __m256d outavx = _mm256_loadu_pd(&out[i*r+j]);
                outavx = _mm256_add_pd(outavx, (_mm256_mul_pd(Qavx, Bavx)));
                _mm256_storeu_pd(&out[i*r+j], outavx);
            }
            for (; j < r; j++) {
                out[i * r + j] += Q * B[k*r+j];
            }
        }
    }
}
