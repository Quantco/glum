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
#include <omp.h>
namespace xs = xsimd;


<%def name="simd_microkernel()">
    for (int k = kk; k < kend; k++) {
        F* Lptr = &L[(ii2-ii)*KBLOCK + (k-kk)*IBLOCK2];

        for (int i = ii2; i < iend2; i++, Lptr++) {
            int jend3 = jend2;
            if (jend3 > i + 1) {
                jend3 = i + 1;
            }

            F* outptr = &outtemp[(i-ii2)*JBLOCK2];
            F* Rptr = &R[(jj2-jj)*KBLOCK + (k-kk)*JBLOCK2];

            int Rlen = jend3 - jj2;
            const F* Rsimdend = Rptr + (Rlen / simd_size) * simd_size;
            const F* Rend = Rptr + Rlen;

            auto Lsimd = xs::set_simd(*Lptr);
            for (; 
                Rptr < Rsimdend;
                Rptr+=simd_size, outptr+=simd_size) 
            {
                auto Rsimd = xs::load_aligned(Rptr);
                auto outsimd = xs::load_aligned(outptr);
                outsimd = xs::fma(Lsimd, Rsimd, outsimd);
                outsimd.store_aligned(outptr);
            }
            for (;
                Rptr < Rend;
                Rptr++, outptr++)
            {
                (*outptr) += (*Lptr) * (*Rptr);
            }
        }
    }
</%def>

<%def name="simple_microkernel(IEND, JEND)">
    for (int k = kk; k < kend; k++) {
        F* Lptr = &L[(ii2-ii)*KBLOCK + (k-kk)*IBLOCK2];

        for (int i = ii2; i < ${IEND}; i++, Lptr++) {
            int jend3 = ${JEND};
            F* outptr = &out[i*m+jj2];
            F* Rptr = &R[(jj2-jj)*KBLOCK + (k-kk)*JBLOCK2];
            F* Rptrend = Rptr + jend3 - jj2;

            for (; Rptr < Rptrend; Rptr++,outptr++) {
                (*outptr) += (*Lptr) * (*Rptr);
            }
        }
    }
</%def>

// TODO: choose KBLOCK, JBLOCK depending on m, n
// TODO: aligned allocation
#define KBLOCK 256
#define JBLOCK 512
#define IBLOCK 32
#define JBLOCK2 4
#define IBLOCK2 4 // IBLOCK % IBLOCK2 must be 0

template <typename F>
void _dense_i_outer_block(F* X, F* R, F* out, int m, int n, int jj, int jend, int kk, int kend) {
    constexpr int simd_size = xsimd::simd_type<F>::size;
    for (int ii = 0; ii < m; ii+=IBLOCK) {

        int iend = ii + IBLOCK;
        if (iend > m) {
            iend = m;
        }

        // Pack our left operand into L in the order that we'll access it later.
        // 3d array indexed by (ii2-ii, k-kk, i-ii2)
        F L[KBLOCK*IBLOCK];
        for (int ii2 = ii; ii2 < iend; ii2+=IBLOCK2) {
            int iend2 = ii2 + IBLOCK2;
            if (iend2 > m) {
                iend2 = m;
            }
            for (int k = kk; k < kend; k++) {
                F* Lptr = &L[(ii2-ii)*KBLOCK + (k-kk)*IBLOCK2];
                for (int i = ii2; i < iend2; i++, Lptr++) {
                    (*Lptr) = X[k*m+i];
                }
            }
        }

        int jendblock = jj + ((jend - jj) / JBLOCK2) * JBLOCK2;
        int iendblock = ii + ((iend - ii) * IBLOCK2) * IBLOCK2;
        int jj2 = jj;
        for (; jj2 < jendblock; jj2+=JBLOCK2) {
            int jend2 = jj2 + JBLOCK2;
            int ii2 = ii;
            for (; ii2 < iendblock; ii2+=IBLOCK2) {
                int iend2 = ii2 + IBLOCK2;
                ${simple_microkernel("iend2", "jend2")}
            }
            ${simple_microkernel("iend", "jend2")}
        }
        int ii2 = ii;
        for (; ii2 < iendblock; ii2+=IBLOCK2) {
            int iend2 = ii2 + IBLOCK2;
            ${simple_microkernel("iend2", "jend")}
        }
        ${simple_microkernel("iend", "jend")}
    }
}

template <typename F>
void _dense_j_outer_block(F* X, F* d, F* out, int m, int n, int jj) {

    int jend = jj + JBLOCK;
    if (jend > m) {
        jend = m;
    }
    for (int kk = 0; kk < n; kk+=KBLOCK) {
        int kend = kk + KBLOCK;
        if (kend > n) {
            kend = n;
        }

        // Pack our right operand into R in the order that we'll access it later.
        // 3d array indexed by (jj2-jj, k-kk, j-jj2)
        F R[KBLOCK*JBLOCK];
        for (int jj2 = jj; jj2 < jend; jj2+=JBLOCK2) {
            int jend2 = jj2 + JBLOCK2;
            if (jend2 > jend) {
                jend2 = jend;
            }
            for (int k = kk; k < kend; k++) {
                F dv = d[k];
                F* Rptr = &R[(jj2-jj)*KBLOCK + (k-kk)*JBLOCK2];
                F* Rptrend = Rptr + jend2 - jj2;
                F* Xptr = &X[k*m+jj2];
                for (; Rptr < Rptrend; Rptr++, Xptr++) {
                    (*Rptr) = dv * (*Xptr);
                }
            }
        }

        _dense_i_outer_block(X, R, out, m, n, jj, jend, kk, kend);
    }
}

template <typename F>
void _dense_sandwich(F* X, F* d, F* out, int m, int n) 
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int jj = 0; jj < m; jj+=JBLOCK) {
                _dense_j_outer_block(X, d, out, m, n, jj);
            }
        }
    }


    // make symmetric!
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
