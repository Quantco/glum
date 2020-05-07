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

<%def name="middle_j(kparallel, IBLOCK, JBLOCK, KBLOCKS)">
    int jmaxblock = jmin + ((jmaxinner - jmin) / ${JBLOCK}) * ${JBLOCK};
    for (; j < jmaxblock; j += ${JBLOCK}) {

        // setup simd accumulators
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                auto accumsimd${ir}_${jr} = xs::set_simd(((F)0.0));
            % endfor
        % endfor

        % for ir in range(IBLOCK):
            int basei${ir} = (i - imin2 + ${ir}) * kstep;
        % endfor
        % for jr in range(JBLOCK):
            int basej${jr} = (j - jmin2 + ${jr}) * kstep;
        % endfor

        // main simd inner loop
        % for ir in range(IBLOCK):
            F* Lptr${ir} = &L[basei${ir}];
        % endfor
        % for jr in range(JBLOCK):
            F* Rptr${jr} = &R[basej${jr}];
        % endfor
        int kblocksize = ((kmax - kmin) / simd_size) * simd_size;
        F* Rptr0end = Rptr0 + kblocksize;
        for(; Rptr0 < Rptr0end; 
            % for jr in range(JBLOCK):
                Rptr${jr}+=simd_size,
            % endfor
            % for ir in range(IBLOCK):
                % if ir == IBLOCK - 1:
                    Lptr${ir} += simd_size
                % else:
                    Lptr${ir} += simd_size,
                % endif
            % endfor
            ) {
            % for ir in range(IBLOCK):
                auto Xtd${ir} = xs::load_aligned(Lptr${ir});
                % for jr in range(JBLOCK):
                {
                    auto Xsimd = xs::load_aligned(Rptr${jr});
                    accumsimd${ir}_${jr} = xs::fma(Xtd${ir}, Xsimd, accumsimd${ir}_${jr});
                }
                % endfor
            % endfor
        }

        // horizontal sum of the simd blocks
        % for ir in range(IBLOCK):
            % for jr in range(JBLOCK):
                F accum${ir}_${jr} = xs::hadd(accumsimd${ir}_${jr});
            % endfor
        % endfor

        // remainder loop
        for (int k = kblocksize; k < kmax - kmin; k++) {
            % for ir in range(IBLOCK):
                F Xtd${ir} = L[basei${ir} + k];
            % endfor
            % for jr in range(JBLOCK):
                F Xv${jr} = R[basej${jr} + k];
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
                % if kparallel:
                    #pragma omp atomic
                % endif
                out[(i + ${ir}) * m + (j + ${jr})] += accum${ir}_${jr};
            % endfor
        % endfor
    }
</%def>

<%def name="outer_i(kparallel, IBLOCK, JBLOCKS, KBLOCKS)">
    int imaxblock = imin + ((imax - imin) / ${IBLOCK}) * ${IBLOCK};
    for (; i < imaxblock; i += ${IBLOCK}) {
        int jmaxinner = jmax;
        if (jmaxinner > i + ${IBLOCK}) {
            jmaxinner = i + ${IBLOCK};
        }
        int j = jmin;
        % for JBLOCK in JBLOCKS:
        {
            ${middle_j(kparallel, IBLOCK, JBLOCK, KBLOCKS)}
        }
        % endfor
    }
</%def>

<%def name="dense_base_tmpl(kparallel)">
template <typename F>
void dense_base${kparallel}(F* R, F* L, F* d, F* out,
                int m, int n,
                int imin2, int imax2,
                int jmin2, int jmax2, 
                int kmin, int kmax, int innerblock, int kstep) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    for (int imin = imin2; imin < imax2; imin+=innerblock) {
        int imax = imin + innerblock; 
        if (imax > imax2) {
            imax = imax2; 
        }
        for (int jmin = jmin2; jmin < jmax2; jmin+=innerblock) {
            int jmax = jmin + innerblock; 
            if (jmax > jmax2) {
                jmax = jmax2; 
            }
            int i = imin;
            % for IBLOCK in [4, 2, 1]:
            {
                ${outer_i(kparallel, IBLOCK, [4, 2, 1], [1])}
            }
            % endfor
        }
    }
}
</%def>

${dense_base_tmpl(True)}
${dense_base_tmpl(False)}

<%def name="k_loop(kparallel, order)">
% if kparallel:
    #pragma omp parallel for
    for (int k = 0; k < n; k+=kratio*thresh1d) {
% else:
    for (int k = 0; k < n; k+=kratio*thresh1d) {
% endif
    int kmax2 = k + kratio*thresh1d; 
    if (kmax2 > n) {
        kmax2 = n; 
    }

    F* R = Rglobal;
    % if kparallel:
    R += omp_get_thread_num()*thresh1d*thresh1d*kratio*kratio;
    for (int jj = j; jj < jmax2; jj++) {
    % else:
    #pragma omp parallel for
    for (int jj = j; jj < jmax2; jj++) {
    % endif
        {
            %if order == 'F':
                F* Rptr = &R[(jj-j)*kratio*thresh1d];
                F* Rptrend = Rptr + kmax2 - k;
                F* dptr = &d[k];
                F* Xptr = &X[jj*n+k];
                for (; Rptr < Rptrend; Rptr++, dptr++, Xptr++) {
                    (*Rptr) = (*dptr) * (*Xptr);
                }
            % else:
                for (int kk=k; kk<kmax2; kk++) {
                    R[(jj-j)*kratio*thresh1d+(kk-k)] = d[kk] * X[kk*m+jj];
                }
            % endif
        }
    }

    % if kparallel:
        for (int i = j; i < m; i+=thresh1d) {
    % else:
        #pragma omp parallel for
        for (int i = j; i < m; i+=thresh1d) {
    % endif
        int imax2 = i + thresh1d; 
        if (imax2 > m) {
            imax2 = m; 
        }
        F* L = &Lglobal[omp_get_thread_num()*thresh1d*thresh1d*kratio];
        for (int ii = i; ii < imax2; ii++) {
            %if order == 'F':
                F* Lptr = &L[(ii-i)*kratio*thresh1d];
                F* Lptrend = Lptr + kmax2 - k;
                F* Xptr = &X[ii*n+k];
                for (; Lptr < Lptrend; Lptr++, Xptr++) {
                    *Lptr = *Xptr;
                }
            % else:
                for (int kk=k; kk<kmax2; kk++) {
                    L[(ii-i)*kratio*thresh1d+(kk-k)] = X[kk*m+ii];
                }
            % endif
        }
        dense_base${kparallel}(R, L, d, out, m, n, i, imax2, j, jmax2, k, kmax2, innerblock, kratio*thresh1d);
    }
}
</%def>


<%def name="dense_sandwich_tmpl(order)">
template <typename F>
void _dense_sandwich${order}(F* X, F* d, F* out,
        int m, int n, int thresh1d, int parlevel, int kratio, int innerblock) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    constexpr auto alignment = std::align_val_t{simd_size*sizeof(F)};

    bool kparallel = (n / (kratio*thresh1d)) > (m / thresh1d);
    size_t Rsize = thresh1d*thresh1d*kratio*kratio;
    if (kparallel) {
        Rsize *= omp_get_max_threads();
    }
    auto Rglobal = new (alignment) F[Rsize];
    auto Lglobal = new (alignment) F[omp_get_max_threads()*thresh1d*thresh1d*kratio];
    for (int j = 0; j < m; j+=kratio*thresh1d) {
        int jmax2 = j + kratio*thresh1d; 
        if (jmax2 > m) {
            jmax2 = m; 
        }
        if (kparallel) {
            ${k_loop(True, order)}
        } else {
            ${k_loop(False, order)}
        }
    }
    ::operator delete(Lglobal, alignment);
    ::operator delete(Rglobal, alignment);

    #pragma omp parallel if(m > 100)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            out[j * m + i] = out[i * m + j];
        }
    }
}
</%def>

${dense_sandwich_tmpl('C')}
${dense_sandwich_tmpl('F')}


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
