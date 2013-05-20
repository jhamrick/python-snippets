"""stats.py -- helper functions for working with probabilities/statistics

This software is licensed under the terms of the MIT License as
follows:

Copyright (c) 2013 Jessica B. Hamrick

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import numba

from numpy import log, exp, dot
from numpy.linalg import inv


def normalize(logarr, axis=-1, max_log_value=709.78271289338397):
    """Normalize an array of log-values.

    This function is very useful if you have an array of log
    probabilities that need to be normalized, but some of the
    probabilies might be extremely small (i.e., underflow will occur if
    you try to exponentiate them). This function computes the
    normalization constants in log space, thus avoiding the need to
    exponentiate the values.

    Parameters
    ----------
    logarr: numpy.ndarray
        Array of log values
    axis: integer (default=-1)
        Axis over which to normalize
    max_log_value: float (default=709.78271289338397)
        Largest number that, when exponentiated, will not overflow

    Returns
    -------
    out: (numpy.ndarray, numpy.ndarray)
        2-tuple consisting of the log normalization constants used to
        normalize the array, and the normalized array of log values

    """

    # shape for the normalization constants (that would otherwise be
    # missing axis)
    shape = list(logarr.shape)
    shape[axis] = 1
    # get maximum value of array
    maxlogarr = logarr.max(axis=axis).reshape(shape)
    # calculate how much to shift the array up by
    shift = max_log_value - maxlogarr - 2 - logarr.shape[axis]
    shift[shift < 0] = 0
    # shift the array
    unnormed = logarr + shift
    # convert from logspace
    arr = exp(unnormed)
    # calculate shifted log normalization constants
    _lognormconsts = log(arr.sum(axis=axis)).reshape(shape)
    # calculate normalized array
    lognormarr = unnormed - _lognormconsts
    # unshift normalization constants
    _lognormconsts -= shift
    # get rid of the dimension we normalized over
    lognormconsts = _lognormconsts.sum(axis=axis)

    return lognormconsts, lognormarr


def GP(K, x, y, xo, s=0):
    """Compute the Gaussian Process mean and covariance at points `xo` of the
    posterior distribution over f(xo), given observations `y` at points
    `x`.

    The parameter `s` refers to observation noise, i.e.

    $$K_y = K_f + s^2\delta(x_1-x_2),$$

    where $\delta$ is the Dirac delta function.

    Parameters
    ----------
    K : function
        Kernel function, which takes two vectors as input and returns
        their inner product.
    x : numpy.ndarray
        Vector of input points
    y : numpy.ndarray
        Vector of input observations
    xo : numpy.ndarray
        Vector of inputs for which to estimate output mean/variance
    s : number (default=0)
        Observation noise parameter

    Returns
    -------
    tuple : (yo_mean, yo_cov)
        2-tuple of the mean and covariance for output points yo

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # parameter checking
    if s < 0:
        raise ValueError("invalid value for s: %s" % s)

    # compute the various kernel matrices
    Kxx = K(x, x)
    if s > 0:
        Kxx += np.eye(x.size) * (s ** 2)

    # compute cholesky factorization of Kxx for faster inversion
    try:
        Li = inv(np.linalg.cholesky(Kxx))
    except np.linalg.LinAlgError:
        # matrix is singular, let's try adding some noise and see if
        # we can invert it then
        print "Warning: could not invert kernel matrix, trying with jitter"
        m = np.mean(np.abs(Kxx))
        noise = np.random.normal(0, m * 1e-6, Kxx.shape)
        try:
            Li = inv(np.linalg.cholesky(Kxx + noise))
        except np.linalg.LinAlgError:
            print Kxx
            raise np.linalg.LinAlgError(
                "Could not invert kernel matrix, even with jitter")
    alpha = dot(Li.T, dot(Li, y))

    Kxoxo = K(xo, xo)
    Kxxo = K(x, xo)
    v = dot(Li, Kxxo)

    # estimate the mean and covariance of the function
    mean = dot(Kxxo.T, alpha)
    cov = Kxoxo - dot(v.T, v)

    return mean, cov


def gaussian_kernel(h, w, jit=True):
    """Produces a squared exponential (Gaussian) kernel function of
    the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{(x_1-x_2)^2}{2w^2})$$

    Parameters
    ----------
    h : number
        Output scale kernel parameter
    w : number
        Input scale (Gaussian standard deviation) kernel parameter
    jit : boolean (default=True)
        Whether JIT compile the function with numba

    Returns
    -------
    out : function
        The kernel function takes two 1-d arrays and computes a
        Gaussian kernel covariance matrix. It returns a 2-d array with
        dimensions equal to the size of the input vectors.

        This function will also have attributes corresponding to the
        parameters, i.e. `out.h` and `out.w`.

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # parameter checking
    if h <= 0:
        raise ValueError("invalid value for h: %s" % h)
    if w <= 0:
        raise ValueError("invalid value for w: %s" % w)

    # compute constants
    c = log(h ** 2)

    def kernel(x1, x2):
        out = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                diff = x1[i] - x2[j]
                l = c + (-0.5 * (diff ** 2) / (w ** 2))

                # !!! underflow protection hack, because numba
                # currently can't handle catching/raising exceptions
                if l < -709.78271289338397:
                    out[i, j] = 0
                elif l > 709.78271289338397:
                    out[i, j] = np.inf
                else:
                    out[i, j] = exp(l)

        return out

    # save kernel parameters
    kernel.h = h
    kernel.w = w

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(kernel)
    else:
        K = kernel

    return K


def periodic_kernel(h, w, p, jit=True):
    """Produces a periodic kernel function, of the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{2\sin^2(\frac{x_1-x_2}{2p})}{w^2})$$

    Parameters
    ----------
    h : number
        Output scale kernel parameter
    w : number
        Input scale (Gaussian standard deviation) kernel parameter
    p : number
        Period kernel parameter
    jit : boolean (default=True)
        Whether JIT compile the function with numba

    Returns
    -------
    out : function
        The kernel function takes two 1-d arrays and computes a circular
        Gaussian kernel covariance matrix. It returns a 2-d array with
        dimensions equal to the size of the input vectors.

        This function will also have attributes corresponding to the
        parameters, i.e. `out.h` and `out.w`.

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # parameter checking
    if h <= 0:
        raise ValueError("invalid value for h: %s" % h)
    if w <= 0:
        raise ValueError("invalid value for w: %s" % w)
    if p <= 0:
        raise ValueError("invalid value for p: %s" % p)

    # compute constants
    c1 = log(h ** 2)
    c2 = -2. / (w ** 2)

    def kernel(x1, x2):
        # compute constants to save on computation time
        out = np.empty((x1.size, x2.size))
        for i in xrange(x1.size):
            for j in xrange(x2.size):
                diff = x1[i] - x2[j]
                l = c1 + (c2 * np.sin(diff / (2. * p)) ** 2)

                # !!! underflow protection hack, because numba
                # currently can't handle catching/raising exceptions
                if l < -709.78271289338397:
                    out[i, j] = 0
                elif l > 709.78271289338397:
                    out[i, j] = np.inf
                else:
                    out[i, j] = exp(l)

        return out

    # save kernel parameters
    kernel.h = h
    kernel.w = w
    kernel.p = p

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(kernel)
    else:
        K = kernel

    return K
