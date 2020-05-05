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

template <typename F>
void _dense_sandwich(F* X, F* d, F* out,
        int m, int n) 
{
    #pragma omp parallel
    #pragma omp single nowait
    //out[i,j] = sum_k X[k,i] * d[k] * X[k,j]
    recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n);
    //TODO: try a new version with a block of k as the outermost loop and using
    //the outtemp and pragma omp atomic trick from below.
}

int calc_nblocks(int n, int blocksize) {
    return ceil(((float)n) / ((float)blocksize));
}

template <typename F>
void _dense_sandwich2(F* XC, F* XF, F* d, F* out,
        int m, int n) 
{
    constexpr int simd_size = xsimd::simd_type<F>::size;

    int kblocksize = 16;
    int nkblocks = calc_nblocks(n, kblocksize);

    int iblocksize = 16;
    int niblocks = calc_nblocks(m, iblocksize);

    int jblocksize = 16;
    int njblocks = calc_nblocks(m, jblocksize);

    #pragma omp parallel
    {
        F* outtemp = new F[m * m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                outtemp[i*m+j] = 0.0;
            }
        }

        #pragma omp for
        for (int kb = 0; kb < nkblocks; kb++) {
            int kmin = kb * kblocksize;
            int kmax = kmin + kblocksize;
            if (kmax > n) {
                kmax = n;
            }

            for (int ib = 0; ib < niblocks; ib++) {
                int imin = ib * iblocksize;
                int imax = imin + iblocksize;
                if (imax > m) {
                    imax = m;
                }

                for (int jb = 0; jb < njblocks; jb++) {
                    int jmin = jb * jblocksize;
                    int jblockmax = jmin + jblocksize;

                    for (int i = imin; i < imax; i++) {
                        int jmax = jblockmax;
                        if (jmax > i + 1) {
                            jmax = i + 1;
                        }
                        int jmaxsimd = jmin + ((jmax - jmin) / simd_size) * simd_size;

                        for (int k = kmin; k < kmax; k++) {
                            F XTd = XC[k*m+i] * d[k];
                            auto XTdsimd = xs::set_simd(XTd);
                            int j = jmin;
                            for (; j < jmaxsimd; j+=simd_size) {
                                auto Xsimd = xs::load_unaligned(&XC[k*m+j]);
                                auto outsimd = xs::load_unaligned(&outtemp[i*m+j]);
                                outsimd = xs::fma(XTdsimd, Xsimd, outsimd);
                                outsimd.store_unaligned(&outtemp[i*m+j]);
                            }
                            for (; j < jmax; j++) {
                                outtemp[i*m+j] += XTd * XC[k*m+j];
                            }
                        }
                    }
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                #pragma omp atomic
                out[i*m+j] += outtemp[i*m+j];
            }
        }
        delete outtemp;
    }

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
