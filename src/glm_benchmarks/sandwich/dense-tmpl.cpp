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
        auto XTsimd = xs::load_unaligned(&temp2[k - kmin]);
        auto dsimd = xs::load_unaligned(&d[k]);
        auto Xtd = XTsimd * dsimd;
        % for r in range(JBLOCK):
            auto Xsimd${r} = xs::load_unaligned(&temp[((j - jmin) + ${r}) * ksize + (k - kmin)]);
            accumsimd${r} = xs::fma(Xtd, Xsimd${r}, accumsimd${r});
        % endfor
    }
    % for r in range(JBLOCK):
        F accum${r} = xs::hadd(accumsimd${r});
    % endfor
    for (int k = kmaxavx; k < kmax; k++) {
        F Q = temp2[k - kmin] * d[k];
        % for r in range(JBLOCK):
            accum${r} += Q * temp[((j - jmin) + ${r}) * n + (k - kmin)];
        % endfor
    }
    % for r in range(JBLOCK):
        out[i * m + (j + ${r})] += accum${r};
    % endfor
</%def>

#define KSTEP 200
#define IJTHRESHOLD 16
template <typename F>
void dense_base(F* X, F* d, F* out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    int ksize = kmax - kmin;

    // Flip to be Fortran-ordered
    F temp[KSTEP * IJTHRESHOLD];
    for (int j = jmin; j < jmax; j++) {
        for (int k = kmin; k < kmax; k++) {
            temp[(j - jmin) * ksize + (k - kmin)] = X[k * m + j];
        }
    }

    F temp2[KSTEP];

    for (int i = imin; i < imax; i++) {
        int jmaxinner = jmax;
        if (jmaxinner > i + 1) {
            jmaxinner = i + 1;
        }
        int kmaxavx = kmin + ((kmax - kmin) / simd_size) * simd_size;
        int j = jmin;
        for (int k = kmin; k < kmax; k++) {
            temp2[k - kmin] = X[k * m + i];
        }
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
    bool parallel = (imax - imin) > IJTHRESHOLD || (jmax - jmin) > IJTHRESHOLD;
    if (!parallel) {
        for (int kstart = kmin; kstart < kmax; kstart += KSTEP) {
            int kend = kstart + KSTEP;
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
        recurse_ij(X, d, out,  m, n, imin, isplit, jsplit, jmax, kmin, kmax);

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
    {
        #pragma omp single nowait
        //out[i,j] = sum_k X[k,i] * d[k] * X[k,j]
        recurse_ij(X, d, out, m, n, 0, m, 0, m, 0, n);
    }
    //TODO: try a new version with a block of k as the outermost loop and using
    //the outtemp and pragma omp atomic trick from below.
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
