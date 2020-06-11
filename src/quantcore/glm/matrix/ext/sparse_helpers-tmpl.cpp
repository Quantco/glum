#include <iostream>
#include <vector>
#include <omp.h>

#include <xsimd/xsimd.hpp>

#include "alloc.h"

namespace xs = xsimd;

<%def name="csr_dense_sandwich_tmpl(order)">
template <typename F>
void _csr_dense${order}_sandwich(
    F* Adata, int* Aindices, int* Aindptr,
    F* B, F* d, F* out,
    int m, int n, int r,
    int* rows, int* A_cols, int* B_cols,
    int nrows, int nA_cols, int nB_cols
    ) 
{
    constexpr int simd_size = xsimd::simd_type<F>::size;
    constexpr auto alignment = simd_size*sizeof(F);

    int kblock = 128;
    int jblock = 128;
    std::size_t Rglobal_size = round_to_align(omp_get_max_threads() * kblock * jblock * sizeof(F), alignment);
#ifndef _WIN32
    F* Rglobal = static_cast<F*>(je_aligned_alloc(alignment, Rglobal_size));
#else
    F* Rglobal = static_cast<F*>(_aligned_malloc(Rglobal_size, alignment));
#endif

    std::vector<int> Acol_map(m, -1);
    // Don't parallelize because the number of columns is small
    for (int Ci = 0; Ci < nA_cols; Ci++) {
        int i = A_cols[Ci];
        Acol_map[i] = Ci;
    }

    #pragma omp parallel
    {
        int nB_cols_rounded = ceil(((float)nB_cols) / ((float)simd_size)) * simd_size;
        std::size_t outtemp_size = round_to_align(nA_cols * nB_cols_rounded * sizeof(F), alignment);
#ifndef _WIN32
        F* outtemp = static_cast<F*>(je_aligned_alloc(alignment, outtemp_size));
#else
        F* outtemp = static_cast<F*>(_aligned_malloc(outtemp_size, alignment));
#endif
        for (int Ci = 0; Ci < nA_cols; Ci++) {
            for (int Cj = 0; Cj < nB_cols; Cj++) {
                outtemp[Ci*nB_cols_rounded+Cj] = 0.0;
            }
        }


        #pragma omp for
        for (int Ckk = 0; Ckk < nrows; Ckk+=kblock) {
            int Ckmax = Ckk + kblock;
            if (Ckmax > nrows) {
                Ckmax = nrows;
            }
            for (int Cjj = 0; Cjj < nB_cols; Cjj+=jblock) {
                int Cjmax = Cjj + jblock;
                if (Cjmax > nB_cols) {
                    Cjmax = nB_cols;
                }

                F* R = &Rglobal[omp_get_thread_num()*kblock*jblock];
                for (int Ck = Ckk; Ck < Ckmax; Ck++) {
                    int k = rows[Ck];
                    for (int Cj = Cjj; Cj < Cjmax; Cj++) {
                        int j = B_cols[Cj];
                        %if order == 'C':
                            F Bv = B[k * r + j];
                        % else:
                            F Bv = B[j * n + k];
                        % endif
                        R[(Ck-Ckk) * jblock + (Cj-Cjj)] = d[k] * Bv;
                    }
                }

                for (int Ck = Ckk; Ck < Ckmax; Ck++) {
                    int k = rows[Ck];
                    for (int A_idx = Aindptr[k]; A_idx < Aindptr[k+1]; A_idx++) {
                        int i = Aindices[A_idx];
                        int Ci = Acol_map[i];
                        if (Ci == -1) {
                            continue;
                        }

                        F Q = Adata[A_idx];
                        auto Qsimd = xs::set_simd(Q);

                        int Cj = Cjj;
                        int Cjmax2 = Cjj + ((Cjmax - Cjj) / simd_size) * simd_size;
                        for (; Cj < Cjmax2; Cj+=simd_size) {
                            auto Bsimd = xs::load_aligned(&R[(Ck-Ckk)*jblock+(Cj-Cjj)]);
                            auto outsimd = xs::load_aligned(&outtemp[Ci*nB_cols_rounded+Cj]);
                            outsimd = xs::fma(Qsimd, Bsimd, outsimd);
                            outsimd.store_aligned(&outtemp[Ci*nB_cols_rounded+Cj]);
                        }

                        for (; Cj < Cjmax; Cj++) {
                            outtemp[Ci*nB_cols_rounded+Cj] += Q * R[(Ck-Ckk)*jblock+(Cj-Cjj)];
                        }
                    }
                }
            }
        }

        for (int Ci = 0; Ci < nA_cols; Ci++) {
            for (int Cj = 0; Cj < nB_cols; Cj++) {
                #pragma omp atomic
                out[Ci*nB_cols+Cj] += outtemp[Ci*nB_cols_rounded+Cj];
            }
        }
#ifndef _WIN32
        je_sdallocx(outtemp, outtemp_size, 0);
#else
        _aligned_free(outtemp);
#endif

    }

#ifndef _WIN32
    je_sdallocx(Rglobal, Rglobal_size, 0);
#else
    _aligned_free(Rglobal);
#endif
}
</%def>

${csr_dense_sandwich_tmpl('C')}
${csr_dense_sandwich_tmpl('F')}
