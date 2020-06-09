// The dense_sandwich function below implement a BLIS/GotoBLAS-like sandwich
// product for computing A.T @ diag(d) @ A
// It works for both C-ordered and Fortran-ordered matrices.
// It is parallelized to be fast for both narrow and square matrices
//
// A good intro to thinking about matrix-multiply optimization is here:
// https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/lecture-slides/MIT6_172F18_lec1.pdf
//
// For more reading, it'd be good to dig into the GotoBLAS and BLIS implementation. 
// page 3 here has a good summary of the ordered of blocking/loops:
// http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
//
// The innermost simd loop is parallelized using xsimd and should
// use the largest vector instructions available on any given machine.
//
// There's a bit of added complexity here from the use of Mako templates.
// It looks scary, but it makes the loop unrolling and generalization across
// matrix orderings and parallelization schemes much simpler than it would be
// if implemented directly.
//
//
// Also included is a csr_dense_sandwich function for computing the
// off-diagonal blocks when using a dense-sparse split matrix.

#ifndef _WIN32
  #define JEMALLOC_NO_DEMANGLE
  #if __APPLE__
    #define JEMALLOC_NO_RENAME
  #endif
  #include <jemalloc/jemalloc.h>
#endif
#include <xsimd/xsimd.hpp>
#include <iostream>
#include <omp.h>

namespace xs = xsimd;

std::size_t round_to_align(std::size_t size, std::size_t alignment) {
  std::size_t remainder = size % alignment;

  if (remainder == 0) {
    return size;
  } else {
    return size + alignment - remainder;
  }
}

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

        // remainder loop handling the entries that can't be handled in a
        // simd_size stride
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
                    // we only need to be careful about parallelism when we're
                    // parallelizing the k loop. if we're just parallelizing i
                    // and j, the sum here is safe
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
void _dense${order}_sandwich(F* X, F* d, F* out,
        int m, int n, int thresh1d, int kratio, int innerblock) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    constexpr auto alignment = simd_size * sizeof(F);

    bool kparallel = (n / (kratio*thresh1d)) > (m / thresh1d);
    size_t Rsize = thresh1d*thresh1d*kratio*kratio;
    if (kparallel) {
        Rsize *= omp_get_max_threads();
    }
    std::size_t Rglobal_size = round_to_align(Rsize * sizeof(F), alignment);
    std::size_t Lglobal_size = round_to_align(omp_get_max_threads() * thresh1d * thresh1d * kratio * sizeof(F), alignment);
#ifndef _WIN32
    F* Rglobal = static_cast<F*>(je_aligned_alloc(alignment, Rglobal_size));
    F* Lglobal = static_cast<F*>(je_aligned_alloc(alignment, Lglobal_size));
#else
    F* Rglobal = static_cast<F*>(_aligned_malloc(Rglobal_size, alignment));
    F* Lglobal = static_cast<F*>(_aligned_malloc(Lglobal_size, alignment));
#endif
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
#ifndef _WIN32
    je_sdallocx(Lglobal, Lglobal_size, 0);
    je_sdallocx(Rglobal, Rglobal_size, 0);
#else
    _aligned_free(Lglobal);
    _aligned_free(Rglobal);
#endif

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
