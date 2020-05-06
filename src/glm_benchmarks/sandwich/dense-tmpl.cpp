// AVX code for dense sandwich products. Uses Mako templating for inner loop unrolling.
// Mako compiles this to C.
// Algorithm pseudocode:
// recurse_ij(X, d, out, i_min, i_max, j_min, j_max):
//     base case, if (i_max - i_min) * (j_max - j_min) small:
//         dense_base(...)
//     else:
//        split X into four blocks by changing i_min and i_max
//        call recurse_ij on each block
#include "xsimd/xsimd.hpp"
#include <iostream>
namespace xs = xsimd;

<%def name="inner_kj_avx(JBLOCK)">
    % for r in range(JBLOCK):
        auto accumsimd${r} = xs::set_simd(((F)0.0));
    % endfor
    for(int k = kmin; k < kmaxavx; k += simd_size) {
        auto XTsimd = xs::load_unaligned(&X[i * n + k]);
        auto dsimd = xs::load_unaligned(&d[k]);
        auto Xtd = XTsimd * dsimd;
        % for r in range(JBLOCK):
            auto Xsimd${r} = xs::load_unaligned(&X[(j + ${r}) * n + k]);
            accumsimd${r} = xs::fma(Xtd, Xsimd${r}, accumsimd${r});
        % endfor
    }
    % for r in range(JBLOCK):
        F accum${r} = xs::hadd(accumsimd${r});
    % endfor
    for (int k = kmaxavx; k < kmax; k++) {
        F Q = X[i * n + k] * d[k];
        % for r in range(JBLOCK):
            accum${r} += Q * X[(j + ${r}) * n + k];
        % endfor
    }
    % for r in range(JBLOCK):
        #pragma omp atomic
        out[i * m + (j + ${r})] += accum${r};
    % endfor
</%def>

template <typename F>
void dense_base(F* X, F* d, F* out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    for (int i = imin; i < imax; i++) {
        int jmaxinner = jmax;
        if (jmaxinner > i + 1) {
            jmaxinner = i + 1;
        }
        int kmaxavx = kmin + ((kmax - kmin) / simd_size) * simd_size;
        int j = jmin;
        % for JBLOCK in [8, 4, 2, 1]:
            for (; j < jmin + ((jmaxinner - jmin) / ${JBLOCK}) * ${JBLOCK}; j += ${JBLOCK}) {
                ${inner_kj_avx(JBLOCK)}
            }
        % endfor
    }
}

template <typename F>
void recurse_ij(F* X, F* d, F* out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax,
                int level) 
{
    size_t isize = (imax - imin);
    size_t jsize = (jmax - jmin);
    size_t ksize = (kmax - kmin);
    size_t size = isize * jsize * ksize;
    if (size < 16 * 4096) {
        // std::stringstream stream; // #include <sstream> for this
        // stream << isize << " " << jsize << " " << ksize << " " << level << std::endl;
        // std::cout << stream.str();
        dense_base(X, d, out, m, n, imin, imax, jmin, jmax, kmin, kmax);
        return;
    }

    bool parallel = level < 7;
    constexpr int kratio = 20;
    if (kratio * isize >= ksize && isize >= jsize) {
        int isplit = (imax + imin) / 2;

        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, isplit, jmin, jmax, kmin, kmax, level + 1);

        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, isplit, imax, jmin, jmax, kmin, kmax, level + 1);

    } else if (jsize >= isize && kratio * jsize >= ksize) {
        int jsplit = (jmax + jmin) / 2;

        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, imax, jmin, jsplit, kmin, kmax, level + 1);

        // check if any of the entries are in the lower triangle
        if (jsplit <= imax) {
            #pragma omp task if(parallel)
            recurse_ij(X, d, out, m, n, imin, imax, jsplit, jmax, kmin, kmax, level + 1);
        }

    } else {
        int ksplit = (kmax + kmin) / 2;

        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, imax, jmin, jmax, kmin, ksplit, level + 1);

        #pragma omp task if(parallel)
        recurse_ij(X, d, out, m, n, imin, imax, jmin, jmax, ksplit, kmax, level + 1);
    }
}

template <typename F>
void _dense_sandwich(F* X, F* d, F* out,
        int m, int n) 
{
    #pragma omp parallel
    #pragma omp single nowait
    //out[i,j] = sum_k X[k,i] * d[k] * X[k,j]
    recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n, 0);
}


template <typename F>
void _csr_dense_sandwich(
    F* Adata, int* Aindices, int* Aindptr,
    F* B, F* d, F* out,
    int m, int n, int r) 
{
    constexpr int simd_size = xsimd::simd_type<F>::size;

    #pragma omp parallel
    {
        F* outtemp = new F[m*r];
        for (int s = 0; s < m * r; s++) {
            outtemp[s] = 0.0;
        }

        #pragma omp for
        for (int k = 0; k < n; k++) {
            int A_idx = Aindptr[k];
            int A_idx_max = Aindptr[k+1];
            for (; A_idx < A_idx_max; A_idx++) {
                int i = Aindices[A_idx];
                F Q = Adata[A_idx] * d[k];
                auto Qsimd = xs::set_simd(Q);
                int j = 0;
                for (; j < (r / simd_size) * simd_size; j+=simd_size) {
                    //TODO: look into memory alignment for numpy arrays
                    auto Bsimd = xs::load_unaligned(&B[k*r+j]);
                    auto outsimd = xs::load_unaligned(&outtemp[i*r+j]);
                    outsimd = xs::fma(Qsimd, Bsimd, outsimd);
                    outsimd.store_unaligned(&outtemp[i*r+j]);
                }

                // TODO: use a smaller simd type for the remainder? e.g. 8,4,2,1
                // given that we often a fairly small number of dense columns,
                // this could have a substantial effect. 4,2,1 is going to be
                // faster than 4,1,1,1

                for (; j < r; j++) {
                    outtemp[i * r + j] += Q * B[k*r+j];
                }
            }
        }

        for (int s = 0; s < m * r; s++) {
            #pragma omp atomic
            out[s] += outtemp[s];
        }
        delete outtemp;
    }
}
