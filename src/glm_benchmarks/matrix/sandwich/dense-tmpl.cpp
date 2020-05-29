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
#include "xsimd/xsimd.hpp"
#include <iostream>
#include <omp.h>
namespace xs = xsimd;

<%def name="middle_j(kparallel, IBLOCK, JBLOCK)">
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
                out[(i + ${ir}) * out_m + (j + ${jr})] += accum${ir}_${jr};
            % endfor
        % endfor
    }
</%def>

<%def name="outer_i(kparallel, IBLOCK, JBLOCKS)">
    int imaxblock = imin + ((imax - imin) / ${IBLOCK}) * ${IBLOCK};
    for (; i < imaxblock; i += ${IBLOCK}) {
        int jmaxinner = jmax;
        if (jmaxinner > i + ${IBLOCK}) {
            jmaxinner = i + ${IBLOCK};
        }
        int j = jmin;
        % for JBLOCK in JBLOCKS:
        {
            ${middle_j(kparallel, IBLOCK, JBLOCK)}
        }
        % endfor
    }
</%def>

<%def name="dense_base_tmpl(kparallel)">
template <typename F>
void dense_base${kparallel}(F* R, F* L, F* d, F* out,
                int out_m,
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
                ${outer_i(kparallel, IBLOCK, [4, 2, 1])}
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
    for (int Rk = 0; Rk < in_n; Rk+=kratio*thresh1d) {
% else:
    for (int Rk = 0; Rk < in_n; Rk+=kratio*thresh1d) {
% endif
    int Rkmax2 = Rk + kratio*thresh1d; 
    if (Rkmax2 > in_n) {
        Rkmax2 = in_n; 
    }

    F* R = Rglobal;
    % if kparallel:
    R += omp_get_thread_num()*thresh1d*thresh1d*kratio*kratio;
    for (int Cjj = Cj; Cjj < Cjmax2; Cjj++) {
    % else:
    #pragma omp parallel for
    for (int Cjj = Cj; Cjj < Cjmax2; Cjj++) {
    % endif
        {
            int jj = cols[Cjj];
            %if order == 'F':
                //TODO: this could use some pointer logic for the R assignment?
                for (int Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    int kk = rows[Rkk];
                    R[(Cjj-Cj)*kratio*thresh1d+(Rkk-Rk)] = d[kk] * X[jj*n+kk];
                }
            % else:
                for (int Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    int kk = rows[Rkk];
                    R[(Cjj-Cj)*kratio*thresh1d+(Rkk-Rk)] = d[kk] * X[kk*m+jj];
                }
            % endif
        }
    }

    % if kparallel:
        for (int Ci = Cj; Ci < out_m; Ci+=thresh1d) {
    % else:
        #pragma omp parallel for
        for (int Ci = Cj; Ci < out_m; Ci+=thresh1d) {
    % endif
        int Cimax2 = Ci + thresh1d; 
        if (Cimax2 > out_m) {
            Cimax2 = out_m; 
        }
        F* L = &Lglobal[omp_get_thread_num()*thresh1d*thresh1d*kratio];
        for (int Cii = Ci; Cii < Cimax2; Cii++) {
            int ii = cols[Cii];
            %if order == 'F':
                for (int Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    int kk = rows[Rkk];
                    L[(Cii-Ci)*kratio*thresh1d+(Rkk-Rk)] = X[ii*n+kk];
                }
            % else:
                for (int Rkk=Rk; Rkk<Rkmax2; Rkk++) {
                    int kk = rows[Rkk];
                    L[(Cii-Ci)*kratio*thresh1d+(Rkk-Rk)] = X[kk*m+ii];
                }
            % endif
        }
        dense_base${kparallel}(R, L, d, out, out_m, Ci, Cimax2, Cj, Cjmax2, Rk, Rkmax2, innerblock, kratio*thresh1d);
    }
}
</%def>


<%def name="dense_sandwich_tmpl(order)">
template <typename F>
void _dense${order}_sandwich(int* rows, int* cols, F* X, F* d, F* out,
        int in_n, int out_m, int m, int n, int thresh1d, int kratio, int innerblock) 
{
    constexpr std::size_t simd_size = xsimd::simd_type<F>::size;
    constexpr auto alignment = std::align_val_t{simd_size*sizeof(F)};

    bool kparallel = (in_n / (kratio*thresh1d)) > (out_m / thresh1d);
    size_t Rsize = thresh1d*thresh1d*kratio*kratio;
    if (kparallel) {
        Rsize *= omp_get_max_threads();
    }
    auto Rglobal = new (alignment) F[Rsize];
    auto Lglobal = new (alignment) F[omp_get_max_threads()*thresh1d*thresh1d*kratio];
    for (int Cj = 0; Cj < out_m; Cj+=kratio*thresh1d) {
        int Cjmax2 = Cj + kratio*thresh1d; 
        if (Cjmax2 > out_m) {
            Cjmax2 = out_m; 
        }
        if (kparallel) {
            ${k_loop(True, order)}
        } else {
            ${k_loop(False, order)}
        }
    }
    ::operator delete(Lglobal, alignment);
    ::operator delete(Rglobal, alignment);

    #pragma omp parallel if(out_m > 100)
    for (int Ci = 0; Ci < out_m; Ci++) {
        for (int Cj = 0; Cj <= Ci; Cj++) {
            out[Cj * out_m + Ci] = out[Ci * out_m + Cj];
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
    constexpr auto alignment = std::align_val_t{simd_size*sizeof(F)};

    int kblock = 128;
    int jblock = 128;
    F* Rglobal = new (alignment) F[omp_get_max_threads()*kblock*jblock];

    #pragma omp parallel
    {
        int r2 = ceil(((float)r) / ((float)simd_size)) * simd_size;
        F* outtemp = new (alignment) F[m*r2];
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
        ::operator delete(outtemp, alignment);
    }
}
</%def>

${csr_dense_sandwich_tmpl('C')}
${csr_dense_sandwich_tmpl('F')}
