# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#
# License: BSD 3 clause
#         Substantial modifications by Ben Thompson <t.ben.thompson@gmail.com>
#
from libc.math cimport fabs
cimport numpy as np
import numpy as np
from numpy.math cimport INFINITY

from cython cimport floating
from cython.parallel import prange
import warnings
from sklearn.exceptions import ConvergenceWarning

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

# The following two functions are shamelessly copied from the tree code. (_random.pxd)
# Authors: Arnaud Joly
#
# License: BSD-3-clause
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0):
        seed[0] = 1

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Use the modulo to make sure that we don't return a values greater than the
    # maximum representable value for signed 32bit integers (i.e. 2^31 - 1).
    # Note that the parenthesis are needed to avoid overflow: here
    # RAND_R_MAX is cast to UINT32_t before 1 is added.
    return seed[0] % ((<UINT32_t>RAND_R_MAX) + 1)


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


def identify_active_rows(
    floating[::1] gradient_rows,
    floating[::1] hessian_rows,
    floating[::1] old_hessian_rows,
    floating C
):
    cdef int n = hessian_rows.shape[0]

    hessian_rows_diff_arr = np.empty_like(hessian_rows)
    cdef floating[::1] hessian_rows_diff = hessian_rows_diff_arr

    cdef floating max_diff = 0
    cdef floating abs_val
    cdef int i

    # TODO: This reduction could be parallelized
    for i in range(n):
        hessian_rows_diff[i] = hessian_rows[i] - old_hessian_rows[i]
        abs_val = fabs(hessian_rows_diff[i])
        if abs_val > max_diff:
            max_diff = abs_val

    cdef bint exclude
    for i in prange(n, nogil=True):
        abs_val = fabs(hessian_rows_diff[i])
        exclude = abs_val < C * max_diff
        if exclude:
            hessian_rows_diff[i] = 0.0
            hessian_rows[i] = old_hessian_rows[i]

    active_rows_arr = np.where(hessian_rows_diff_arr != 0)[0].astype(np.int32)

    return hessian_rows_diff_arr, active_rows_arr


def enet_coordinate_descent_gram(int[::1] active_set,
                                 floating[::1] w,
                                 floating[::1] P1,
                                 floating[:,:] Q,
                                 floating[::1] q,
                                 int max_iter, floating tol, object rng,
                                 bint intercept, bint random,
                                 bint has_lower_bounds,
                                 floating[:] lower_bounds,
                                 bint has_upper_bounds,
                                 floating[:] upper_bounds):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression
        We minimize
        (1/2) * w^T Q w - q^T w + P1 norm(w, 1)
        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef unsigned int n_active_features = active_set.shape[0]

    cdef floating w_ii
    cdef floating P1_ii
    cdef floating qii_temp
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating d_w_tol = tol
    cdef floating norm_min_subgrad = 0
    cdef floating max_min_subgrad
    cdef unsigned int active_set_ii, active_set_jj
    cdef unsigned int ii, jj
    cdef int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    with nogil:
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_active_features):  # Loop over coordinates
                if random:
                    active_set_ii = rand_int(n_active_features, rand_r_state)
                else:
                    active_set_ii = f_iter
                ii = active_set[active_set_ii]

                if ii < <unsigned int>intercept:
                    P1_ii = 0.0
                else:
                    P1_ii = P1[ii - intercept]

                if Q[active_set_ii, active_set_ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                qii_temp = q[ii] - w[ii] * Q[active_set_ii, active_set_ii]
                w[ii] = fsign(-qii_temp) * fmax(fabs(qii_temp) - P1_ii, 0) / Q[active_set_ii, active_set_ii]

                if ii >= <unsigned int>intercept:
                    if has_lower_bounds:
                        if w[ii] < lower_bounds[ii - intercept]:
                            w[ii] = lower_bounds[ii - intercept]
                    if has_upper_bounds:
                        if w[ii] > upper_bounds[ii - intercept]:
                            w[ii] = upper_bounds[ii - intercept]

                if w[ii] != 0.0 or w_ii != 0.0:
                    # q +=  (w[ii] - w_ii) * Q[ii] # Update q = X.T (X w - y)
                    for active_set_jj in range(n_active_features):
                        jj = active_set[active_set_jj]
                        q[jj] += (w[ii] - w_ii) * Q[active_set_ii, active_set_jj]

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the minimum norm subgradient as the
                # ultimate stopping criterion
                cython_norm_min_subgrad(
                    active_set,
                    w, q, P1, intercept,
                    has_lower_bounds, lower_bounds, has_upper_bounds, upper_bounds,
                    &norm_min_subgrad, &max_min_subgrad
                )
                if norm_min_subgrad <= tol:
                    break
        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                warnings.warn("Coordinate descent did not converge. You might want to "
                              "increase the number of iterations. Minimum norm "
                              "subgradient: {}, tolerance: {}".format(norm_min_subgrad, tol),
                              ConvergenceWarning)

    return np.asarray(w), norm_min_subgrad, max_min_subgrad, tol, n_iter + 1


cdef void cython_norm_min_subgrad(
    int[::1] active_set,
    floating[::1] coef,
    floating[::1] grad,
    floating[::1] P1,
    bint intercept,
    bint has_lower_bounds,
    floating[:] lower_bounds,
    bint has_upper_bounds,
    floating[:] upper_bounds,
    floating* norm_out,
    floating* max_out
) nogil:
    """Compute the gradient of all subgradients with minimal L2-norm.

    subgrad = grad + P1 * subgrad(|coef|_1)

    g_i = grad_i + (P2*coef)_i

    if coef_i > 0:   g_i + P1_i
    if coef_i < 0:   g_i - P1_i
    if coef_i = 0:   sign(g_i) * max(|g_i|-P1_i, 0)

    Parameters
    ----------
    coef : ndarray
        coef[0] may be intercept.

    grad : ndarray, shape=coef.shape

    P1 : ndarray
        always without intercept

    intercept : bool
        are we including an intercept?

    _lower_bounds : ndarray
        lower bounds. When a coefficient is located at the bound and
        it's gradient suggest that it would be optimal to go beyond
        the bound, we set the gradient of this feature to zero.

    _upper_bounds : ndarray
        see lb.
    """
    cdef floating term, absterm
    cdef int active_set_i
    cdef int i

    norm_out[0] = 0
    max_out[0] = INFINITY
    for active_set_i in range(len(active_set)):
        i = active_set[active_set_i]

        if i < intercept:
            norm_out[0] = fabs(grad[0])
            max_out[0] = norm_out[0]
            continue

        if coef[i] == 0:
            term = fsign(grad[i]) * fmax(fabs(grad[i]) - P1[i - intercept], 0)
        else:
            term = grad[i] + fsign(coef[i]) * P1[i - intercept]
        if has_lower_bounds and coef[i] == lower_bounds[i - intercept] and term > 0:
            term = 0
        if has_upper_bounds and coef[i] == upper_bounds[i - intercept] and term < 0:
            term = 0
        absterm = fabs(term)
        norm_out[0] += absterm
        if absterm > max_out[0]:
            max_out[0] = absterm


def _norm_min_subgrad(
    int[::1] active_set,
    floating[::1] coef,
    floating[::1] grad,
    floating[::1] P1,
    bint intercept,
    bint has_lower_bounds,
    floating[:] lower_bounds,
    bint has_upper_bounds,
    floating[:] upper_bounds
):
    cdef floating norm_out
    cdef floating max_out
    cython_norm_min_subgrad(
        active_set,
        coef,
        grad,
        P1,
        intercept,
        has_lower_bounds,
        lower_bounds,
        has_upper_bounds,
        upper_bounds,
        &norm_out,
        &max_out
    )
    return norm_out, max_out
