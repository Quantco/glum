# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#
# License: BSD 3 clause
#         Substantial modifications by Ben Thompson <t.ben.thompson@gmail.com>
#
# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.math cimport fabs
cimport numpy as np  # is it a problem that these two imports have same name???
import numpy as np
import numpy.linalg as linalg
from numpy.math cimport INFINITY

cimport cython
from cpython cimport bool
from cython cimport floating
from cython.parallel import prange
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
        which amounts to the L1 problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """

    # get the data information into easy vars
    cdef unsigned int n_active_features = active_set.shape[0]
    cdef unsigned int n_features = Q.shape[0]  # are you sure this is CORRECT?
    # n_features is exactly now the same as n_active_features; should it be q.shape[0]?
    # however, the variable is never used again, so I guess we are okay

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
                    active_set_ii = f_iter  # _ii is an index
                ii = active_set[active_set_ii]  # odd syntax; gets the active set's row

                if ii < <unsigned int>intercept:  # but `intercept` is binary!?
                    P1_ii = 0.0
                else:
                    P1_ii = P1[ii - intercept]

                if Q[active_set_ii, active_set_ii] == 0.0:  # no need to multiply zeros
                    continue

                w_ii = w[ii]  # Store previous value, the iith element of w

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

def enet_coordinate_descent_gram_diag_fisher(
                                 #floating[:,:] X, 
                                 #floating[:] hessian_rows,
                                 #np.ndarray[np.float64_t, ndim=2] X,
                                 X,
                                 np.ndarray[np.float64_t, ndim=1] hessian_rows,
                                 int[::1] active_set,
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
        which amounts to the L1 problem when:
        Q = X^T X (Gram matrix)
        q = X^T y
    """
    # The CD that never calculates the entire Hessian matrix; only rows, when they are
    # necessary.

    # get the data information into easy vars
    cdef unsigned int n_active_features = active_set.shape[0]
    # cdef unsigned int n_features = q.shape[0]  # check that this is the same as Q.shape[0]

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

    #cdef np.ndarray[np.float64_t, dtype=q.dtype] Qj  # will hold rows of the Q matrix
    # cdef floating[:] Q_active_set_ii = np.empty(n_active_features, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] Q_active_set_ii = np.empty(n_active_features, dtype=np.float64)
    # is empty bad?

    cdef cur_col  # makes sure we can overwrite this and not take up too much memory?

    # used to check correctness of code - compare with Q from non-diag-fisher
    # cdef np.ndarray[np.float64_t, ndim=2] Q_check = np.empty((n_active_features, n_active_features), dtype=np.float64)

    # THE inefficiency here is that we must re-calculate the rows of Q for every iteration.

    # with nogil:
    for n_iter in range(max_iter):
        w_max = 0.0
        d_w_max = 0.0
        for f_iter in range(n_active_features):  # Loop over coordinates
            if random:
                active_set_ii = rand_int(n_active_features, rand_r_state)
            else:
                active_set_ii = f_iter  # _ii is an index
            ii = active_set[active_set_ii]  # odd syntax; gets an active row

            # remember, Q hasn't been created yet.
            # and the only row we need for this iteration is Q[active_set_ii]
            # also remember, sandwich product is between data.X, state.hessian_rows
            # only use first column of X in the X^T part of the sandwich

            # if intercept == 1:  # maybe this should go into the next if statement?
                # Q_active_set_ii[0] = hessian_rows.sum()
                # this is perfectly valid because hessian_rows has length 16087

            if active_set[0] < <unsigned int>intercept:
                if ii == 0:
                    Q_active_set_ii[0] = hessian_rows.sum()
                    Q_active_set_ii[1:] = X.transpose_matvec(hessian_rows)
                    # Q_active_set_ii[1:] = np.matmul(hessian_rows, X)  
                else:
                    cur_col = X[:, active_set_ii - 1]
                    Q_active_set_ii[0] = cur_col.transpose_matvec(hessian_rows)
                    Q_active_set_ii[1:] = X.transpose_matvec(cur_col.multiply(hessian_rows).ravel())

                    # Q_active_set_ii[0] = X[:, active_set_ii - 1].transpose_matvec(hessian_rows)
                    # Q_active_set_ii[1:] = X.transpose_matvec(X[:, active_set_ii - 1].multiply(hessian_rows).ravel())

                    # Q_active_set_ii[1:] = X.transpose_matvec(hessian_rows * X[:, active_set_ii - 1].A.ravel())
                    # Q_active_set_ii[1:] = X.transpose_matvec(hessian_rows * X.A[:, active_set_ii - 1]) 
                    # Q_active_set_ii[0] = np.dot(hessian_rows, X[:, active_set_ii - 1])
                    # Q_active_set_ii[1:] = np.matmul((hessian_rows * X[:, active_set_ii - 1]), X)  # use matvec

            # if ii < <unsigned int>intercept:
            #     # Q_active_set_ii[1:] = np.matmul(hessian_rows, X[:, 1:])
            #     # Q_active_set_ii[0] = hessian_rows.sum()
            #     Q_active_set_ii[1:] = np.matmul(hessian_rows, X)

            # the next condition in the original code checks for sparseness.
            # we won't worry about that here. but we will have to go elem by elem
                # actually no, we don't have to go elem by elem. Cython can handle.
            # do i have to worry about intercept term? or does X already ignore it?

            # what about the active features? do we only need to multiply by those?
            # correct. i believe we only need to multiply by those
            # so X can actually be passed in as X[(active_set_ii) cartesian product (active_set_ii)], right?
            # the only problem is that it would mess up the active_set_ii / active_set_jj logic...

            # NB! We care about the intercept column as well, and must treat differently.
            # does feature_list in sklearnfork include the intercept column?
            else:
                cur_col = X[:, active_set_ii]
                Q_active_set_ii = X.transpose_matvec(cur_col.multiply(hessian_rows).ravel())

                # Q_active_set_ii = X.transpose_matvec(hessian_rows * X[:, active_set_ii].A.ravel())
                # Q_active_set_ii = X.transpose_matvec(hessian_rows * X.A[:, active_set_ii]) 
                # Q_active_set_ii = np.matmul((hessian_rows * X[:, active_set_ii]), X)
                # Q_active_set_ii[intercept:] = np.matmul((hessian_rows * X[:, active_set_ii - intercept]), X)
                # Q_active_set_ii[intercept:] = np.matmul((hessian_rows * X[:, active_set_ii]), X[:, intercept:])
                # CHECK that we've covered all the cases!!!!!


                # Q_active_set_ii[intercept:] = np.matmul((hessian_rows * X[:, intercept + active_set_ii]), X)

            # Q_check[active_set_ii] = Q_active_set_ii  # for checking correctness

            # for col_idx in range(X.shape[1]):
            #     for row_idx in range(X.shape[0]):
            #         Q_active_set_ii[intercept + col_idx] = <array multiplication>

            # the next if statement checks for P2, but we're only using this for L1
            # regularization, so we don't need to worry about it. and that's all!

            # If the above works, can merge the following into the above if-else!!!!
            if ii < <unsigned int>intercept:  # equiv. to (ii=0 and intercept=1)
                P1_ii = 0.0
            else:
                P1_ii = P1[ii - intercept]

            if Q_active_set_ii[active_set_ii] == 0.0:  # no need to multiply zeros
                continue

            w_ii = w[ii]  # Store previous value, the iith element of w

            qii_temp = q[ii] - w[ii] * Q_active_set_ii[active_set_ii]  ##
            w[ii] = fsign(-qii_temp) * fmax(fabs(qii_temp) - P1_ii, 0) / Q_active_set_ii[active_set_ii]

            if ii >= <unsigned int>intercept: # this is big-brain logic
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
                    # this is equivalent to A += Bj * z
                    q[jj] += (w[ii] - w_ii) * Q_active_set_ii[active_set_jj]

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
        # with gil:
        warnings.warn("Coordinate descent did not converge. You might want to "
                            "increase the number of iterations. Minimum norm "
                            "subgradient: {}, tolerance: {}".format(norm_min_subgrad, tol),
                            ConvergenceWarning)

    return np.asarray(w), norm_min_subgrad, max_min_subgrad, tol, n_iter + 1 # , Q_check

def enet_coordinate_descent_gram_diag_only(int[::1] active_set,
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
    # In this version, we don't access the off-diagonal elements. And see what happens.

    # get the data information into easy vars
    cdef unsigned int n_active_features = active_set.shape[0]
    cdef unsigned int n_features = Q.shape[0]

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

                # if w[ii] != 0.0 or w_ii != 0.0:
                #     for active_set_jj in range(n_active_features):
                #         jj = active_set[active_set_jj]
                #         q[jj] += (w[ii] - w_ii) * Q[active_set_ii, active_set_jj]

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
