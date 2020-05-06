// simd code for dense sandwich products. Uses Mako templating for inner loop unrolling.
// Mako compiles this to C which is included in the sandwich.pyx cython file.
// Algorithm pseudocode:
// recurse(X, d, out, i_min, i_max, j_min, j_max, k_min, k_max):
//     base case, if (i_max - i_min) * (j_max - j_min) * (k_max - k_min) small:
//         dense_base(...)
//     else:
//        split X into two block splitting on the largest dimension
//        call recurse on each block
#include "xsimd/xsimd.hpp"
#include <iostream>
namespace xs = xsimd;

<%def name="inner_k(IBLOCK, JBLOCK, KBLOCK)">
int kmaxblock = kmin + ((kmax - kmin) / (simd_size * ${KBLOCK})) * (simd_size * ${KBLOCK});
for(; k < kmaxblock; k += ${KBLOCK} * simd_size) {

    % for kr in range(KBLOCK):
        auto dsimd${kr} = xs::load_unaligned(&d[k + ${kr} * simd_size]);
    % endfor
    % for ir in range(IBLOCK):
        % for kr in range(KBLOCK):
            auto XTsimd${ir}_${kr} = xs::load_unaligned(&X[basei${ir} + k + ${kr} * simd_size]);
            auto Xtd${ir}_${kr} = XTsimd${ir}_${kr} * dsimd${kr};
        % endfor
    % endfor
    % for jr in range(JBLOCK):
        % for kr in range(KBLOCK):
            auto Xsimd${jr}_${kr} = xs::load_unaligned(&X[basej${jr} + k + ${kr} * simd_size]);
        % endfor
    % endfor
    % for ir in range(IBLOCK):
        % for jr in range(JBLOCK):
            % for kr in range(KBLOCK):
                accumsimd${ir}_${jr} = xs::fma(
                    Xtd${ir}_${kr},
                    Xsimd${jr}_${kr},
                    accumsimd${ir}_${jr}
                );
            % endfor
        % endfor
    % endfor
}
</%def>

<%def name="middle_j(IBLOCK, JBLOCK, KBLOCKS)">
    int jmaxblock = jmin + ((jmaxinner - jmin) / ${JBLOCK}) * ${JBLOCK};
    //TODO: prefetching?
    for (; j < jmaxblock; j += ${JBLOCK}) {

        // setup simd accumulators
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                auto accumsimd${ir}_${jr} = xs::set_simd(((F)0.0));
            % endfor
        % endfor

        % for ir in range(IBLOCK):
            int basei${ir} = (i + ${ir}) * n;
        % endfor
        % for jr in range(JBLOCK):
            int basej${jr} = (j + ${jr}) * n;
        % endfor

        // main simd inner loop
        int k = kmin;
        % for KBLOCK in KBLOCKS:
        {
            ${inner_k(IBLOCK, JBLOCK, KBLOCK)}
        }
        % endfor

        // horizontal sum of the simd blocks
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                F accum${ir}_${jr} = xs::hadd(accumsimd${ir}_${jr});
            % endfor
        % endfor

        // remainder loop
        for (; k < kmax; k++) {
            F dv = d[k];
            % for ir in range(IBLOCK):
                F Xtd${ir} = X[basei${ir} + k] * dv;
            % endfor
            % for jr in range(JBLOCK):
                F Xv${jr} = X[basej${jr} + k];
            % endfor
            % for ir in range(IBLOCK):
                % for jr in range(JBLOCK):
                    accum${ir}_${jr} += Xtd${ir} * Xv${jr};
                % endfor
            % endfor
        }

        // add to the output array
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                #pragma omp atomic
                out[(i + ${ir}) * m + (j + ${jr})] += accum${ir}_${jr};
            % endfor
        % endfor
    }
</%def>

<%def name="outer_i(IBLOCK, JBLOCKS, KBLOCKS)">
    int imaxblock = imin + ((imax - imin) / ${IBLOCK}) * ${IBLOCK};
    for (; i < imaxblock; i += ${IBLOCK}) {
        int jmaxinner = jmax;
        if (jmaxinner > i + ${IBLOCK}) {
            jmaxinner = i + ${IBLOCK};
        }
        int j = jmin;
        % for JBLOCK in JBLOCKS:
        {
            ${middle_j(IBLOCK, JBLOCK, KBLOCKS)}
        }
        % endfor
    }
</%def>

template <typename F>
void dense_base(F* X, F* d, F* out,
                int m, int n,
                int imin, int imax,
                int jmin, int jmax, 
                int kmin, int kmax) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    int i = imin;
    % for IBLOCK in [4, 2, 1]:
    {
        ${outer_i(IBLOCK, [2, 1], [1])}
    }
    % endfor
}

template <typename F>
void recurse(F* X, F* d, F* out,
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
    constexpr size_t thresh1d = 64;
    constexpr size_t thresh3d = thresh1d * thresh1d * thresh1d;
    if (size < thresh3d) {
        // std::stringstream stream;
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
        recurse(X, d, out, m, n, imin, isplit, jmin, jmax, kmin, kmax, level + 1);

        #pragma omp task if(parallel)
        recurse(X, d, out, m, n, isplit, imax, jmin, jmax, kmin, kmax, level + 1);

    } else if (jsize >= isize && kratio * jsize >= ksize) {
        int jsplit = (jmax + jmin) / 2;

        #pragma omp task if(parallel)
        recurse(X, d, out, m, n, imin, imax, jmin, jsplit, kmin, kmax, level + 1);

        // check if any of the entries are in the lower triangle
        if (jsplit <= imax) {
            #pragma omp task if(parallel)
            recurse(X, d, out, m, n, imin, imax, jsplit, jmax, kmin, kmax, level + 1);
        }

    } else {
        int ksplit = (kmax + kmin) / 2;

        #pragma omp task if(parallel)
        recurse(X, d, out, m, n, imin, imax, jmin, jmax, kmin, ksplit, level + 1);

        #pragma omp task if(parallel)
        recurse(X, d, out, m, n, imin, imax, jmin, jmax, ksplit, kmax, level + 1);
    }
}

template <typename F>
void _dense_sandwich(F* X, F* d, F* out,
        int m, int n) 
{
    #pragma omp parallel
    #pragma omp single nowait
    //out[i,j] = sum_k X[k,i] * d[k] * X[k,j]
    recurse(X, d, out, m, n, 0, m, 0, m, 0, n, 0);

    #pragma omp parallel if(m > 100)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            out[j * m + i] = out[i * m + j];
        }
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
