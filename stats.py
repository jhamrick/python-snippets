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

from numpy import pi
from numpy import log, exp, sqrt, sign, dot
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
        Variance of noisy observations

    Returns
    -------
    tuple : (yo_mean, yo_cov)
        2-tuple of the mean and covariance for output points yo

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # compute the various kernel matrices
    Kxx = K(x, x) + (np.eye(x.size) * s)
    Kxoxo = K(xo, xo)
    Kxxo = K(x, xo)
    Kxox = K(xo, x)

    # estimate the mean and covariance of the function
    kik = dot(Kxox, inv(Kxx))
    mean = dot(kik, y)
    _cov = Kxoxo - dot(kik, Kxxo)

    # round because we get floating point error around zero and end up
    # with negative variances along the diagonal
    cov = np.round(_cov, decimals=6)

    return mean, cov


def gaussian_kernel(h, w, jit=True):
    """Produces a Gaussian kernel function.

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
        # compute constants to save on computation time
        out = np.empty((x1.size, x2.size))

        for i in xrange(x1.size):
            for j in xrange(x2.size):
                diff = x1[i] - x2[j]
                out[i, j] = exp(c + (-0.5 * (diff ** 2) / (w ** 2)))

        return out

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(kernel)
    else:
        K = kernel

    return K


def circular_gaussian_kernel(h, w, jit=True):
    """Produces a circular Gaussian kernel function.

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
        The kernel function takes two 1-d arrays and computes a circular
        Gaussian kernel covariance matrix. It returns a 2-d array with
        dimensions equal to the size of the input vectors.

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    def kernel(x1, x2):
        # compute constants to save on computation time
        out = np.empty((x1.size, x2.size))
        twopi = 2 * pi
        c = log(h ** 2)

        for i in xrange(x1.size):
            for j in xrange(x2.size):
                # compute circular difference between the points --
                # the idea being, if one is around 2*pi and the other
                # is around 0, they are actually very close
                d = x1[i] - x2[j]
                if abs(d) > pi:
                    diff = d - (sign(d) * twopi)
                else:
                    diff = d

                # log gaussian kernel
                out[i, j] = exp(c + (-0.5 * (diff ** 2) / (w ** 2)))

        return out

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(kernel)
    else:
        K = kernel

    return K
