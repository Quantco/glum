# cython: boundscheck=False, wraparound=False, cdivision=True

from cython cimport floating
from cython.parallel import prange

from libc.math cimport exp, log

def poisson_log_eta_mu_deviance(
    floating factor,
    floating[:] cur_eta,
    floating[:] X_dot_d,
    floating[:] y,
    floating[:] weights,
    floating[:] eta_out,
    floating[:] mu_out
):
    cdef int n = cur_eta.shape[0]
    cdef int i
    cdef floating unit_deviance
    cdef floating deviance = 0.0
    for i in prange(n, nogil=True):
        eta_out[i] = cur_eta[i] + factor * X_dot_d[i]
        mu_out[i] = exp(eta_out[i])
        if y[i] == 0:
            unit_deviance = 2 * (-y[i] + mu_out[i])
        else:
            unit_deviance = 2 * ((y[i] * (log(y[i]) - eta_out[i] - 1)) + mu_out[i])
        deviance += weights[i] * unit_deviance
    return deviance

def poisson_log_rowwise_gradient_hessian(
    floating[:] y,
    floating[:] weights,
    floating[:] eta,
    floating[:] mu,
    floating[:] gradient_rows_out,
    floating[:] fisher_W_out
):
    cdef int n = eta.shape[0]
    cdef int i
    cdef floating unit_variance, sigma_inv, d1, d1_sigma_inv
    for i in prange(n, nogil=True):
        unit_variance = mu[i]
        sigma_inv = weights[i] / unit_variance
        d1 = mu[i]
        d1_sigma_inv = d1 * sigma_inv
        gradient_rows_out[i] = d1_sigma_inv * (y[i] - mu[i])
        fisher_W_out[i] = d1 * d1_sigma_inv

