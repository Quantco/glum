# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#
# License: BSD 3 clause
#
# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._cython_blas cimport (_axpy, _dot, _asum, _ger, _gemv, _nrm2,
                                   _copy, _scal)
from sklearn.utils._cython_blas cimport RowMajor, ColMajor, Trans, NoTrans


from sklearn.utils._random cimport our_rand_r

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()


# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


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





def enet_coordinate_descent_gram(floating[::1] w,
                                 floating[::1] P1,
                                 np.ndarray[floating, ndim=2, mode='c'] Q,
                                 np.ndarray[floating, ndim=1, mode='c'] q,
                                 int max_iter, floating tol, object rng,
                                 bint intercept, bint random=0):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression
        We minimize
        (1/2) * w^T Q w - q^T w + P1 norm(w, 1)
        which amount to the Elastic-Net problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef unsigned int n_features = Q.shape[0]

    cdef floating w_ii
    cdef floating P1_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating d_w_tol = tol
    cdef unsigned int ii
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    cdef floating* Q_ptr = &Q[0, 0]
    cdef floating* q_ptr = &q[0]

    with nogil:
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):  # Loop over coordinates
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if ii < intercept:
                    P1_ii = 0.0
                else:
                    P1_ii = P1[ii - intercept]

                if Q[ii, ii] == 0.0:
                    continue

                w_ii = w[ii]  # Store previous value

                if w_ii != 0.0:
                    # q -= w_ii * Q[ii]
                    _axpy(n_features, -w_ii, Q_ptr + ii * n_features, 1,
                          q_ptr, 1)

                w[ii] = fsign(-q[ii]) * fmax(fabs(q[ii]) - P1_ii, 0) / Q[ii, ii]

                if w[ii] != 0.0:
                    # q +=  w[ii] * Q[ii] # Update q = X.T (X w - y)
                    _axpy(n_features, w[ii], Q_ptr + ii * n_features, 1,
                          q_ptr, 1)

                # update the maximum absolute coefficient update
                d_w_ii = fabs(w[ii] - w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if fabs(w[ii]) > w_max:
                    w_max = fabs(w[ii])

            #TODO: convergence criteria needs a major overhaul
            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller than
                # the tolerance: check the minimum norm subgradient as the
                # ultimate stopping criterion
                with gil:
                    mn_subgrad = _norm_min_subgrad(w, q, P1, intercept)
                    if mn_subgrad <= tol:
                        break
        else:
            # for/else, runs if for doesn't end with a `break`
            with gil:
                warnings.warn("Coordinate descent did not converge. You might want to "
                              "increase the number of iterations. Minimum norm "
                              "subgradient: {}, tolerance: {}".format(mn_subgrad, tol),
                              ConvergenceWarning)

    return np.asarray(w), mn_subgrad, tol, n_iter + 1


def _norm_min_subgrad(
    floating[::1] coef,
    floating[::1] grad,
    floating[::1] P1,
    bint intercept
) -> floating:
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
    """
    cdef floating out
    if intercept:
        out = fabs(grad[0])
    for i in range(intercept, coef.shape[0]):
        if coef[i] == 0:
            out += fabs(fsign(grad[i]) * fmax(fabs(grad[i]) - P1[i - intercept], 0))
        else:
            out += fabs(grad[i] + fsign(coef[i]) * P1[i - intercept])
    return out
