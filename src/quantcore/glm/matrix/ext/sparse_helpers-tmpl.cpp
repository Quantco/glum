#include <xsimd/xsimd.hpp>
#include <iostream>
#include <omp.h>

#include "alloc.h"

namespace xs = xsimd;

<%def name="csr_dense_sandwich_tmpl(order)">
template <typename F>
void _csr_dense${order}_sandwich(
    F* Adata, int* Aindices, int* Aindptr,
    F* B, F* d, F* out,
    int m, int n, int r) 
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

    #pragma omp parallel
    {
        int r2 = ceil(((float)r) / ((float)simd_size)) * simd_size;
        std::size_t outtemp_size = round_to_align(m * r2 * sizeof(F), alignment);
#ifndef _WIN32
        F* outtemp = static_cast<F*>(je_aligned_alloc(alignment, outtemp_size));
#else
        F* outtemp = static_cast<F*>(_aligned_malloc(outtemp_size, alignment));
#endif
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                outtemp[i*r2+j] = 0.0;
            }
        }

        #pragma omp for
        for (int kk = 0; kk < n; kk+=kblock) {
            int kmax = kk + kblock;
            if (kmax > n) {
                kmax = n;
            }
            for (int jj = 0; jj < r; jj+=jblock) {
                int jmax = jj + jblock;
                if (jmax > r) {
                    jmax = r;
                }

                F* R = &Rglobal[omp_get_thread_num()*kblock*jblock];
                for (int k = kk; k < kmax; k++) {
                    for (int j = jj; j < jmax; j++) {
                        %if order == 'C':
                            F Bv = B[k * r + j];
                        % else:
                            F Bv = B[j * n + k];
                        % endif
                        R[(k-kk) * jblock + (j-jj)] = d[k] * Bv;
                    }
                }

                for (int k = kk; k < kmax; k++) {
                    for (int A_idx = Aindptr[k]; A_idx < Aindptr[k+1]; A_idx++) {
                        int i = Aindices[A_idx];
                        F Q = Adata[A_idx];
                        auto Qsimd = xs::set_simd(Q);

                        int j = jj;
                        int jmax2 = jj + ((jmax - jj) / simd_size) * simd_size;
                        for (; j < jmax2; j+=simd_size) {
                            auto Bsimd = xs::load_aligned(&R[(k-kk)*jblock+(j-jj)]);
                            auto outsimd = xs::load_aligned(&outtemp[i*r2+j]);
                            outsimd = xs::fma(Qsimd, Bsimd, outsimd);
                            outsimd.store_aligned(&outtemp[i*r2+j]);
                        }

                        for (; j < jmax; j++) {
                            outtemp[i*r2+j] += Q * R[(k-kk)*jblock+(j-jj)];
                        }
                    }
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                #pragma omp atomic
                out[i*r+j] += outtemp[i*r2+j];
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
