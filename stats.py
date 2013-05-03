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


def normalize(logarr, axis=-1, max_log_value=709.78271289338397):
    """Normalize an array of log-values.  Returns a tuple of
    (normalization constants, normalized array), where both values are
    again in logspace.

    Parameters

    logarr: array of log values

    axis: axis over which to normalize (default=-1)

    max_log_value: largest number that, when exponentiated, will not
    overflow

    """

    # shape for the normalization constants (that would otherwise be
    # missing axis)
    shape = list(logarr.shape)
    shape[axis] = 1
    # get maximum value of array
    maxlogarr = np.max(logarr, axis=axis).reshape(shape)
    # calculate how much to shift the array up by
    shift = (max_log_value - maxlogarr - 2 - logarr.shape[axis])
    # shift the array
    unnormed = logarr + shift
    # convert from logspace
    arr = np.exp(unnormed)
    # calculate shifted log normalization constants
    _lognormconsts = np.log(np.sum(arr, axis=axis)).reshape(shape)
    # calculate normalized array
    lognormarr = unnormed - _lognormconsts
    # unshift normalization constants
    _lognormconsts -= shift
    # get rid of the dimension we normalized over
    lognormconsts = np.sum(_lognormconsts, axis=axis)

    return lognormconsts, lognormarr


def GP(K, x, y, xo):
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

    Returns
    -------
    tuple : (yo_mean, yo_cov)
        2-tuple of the mean and covariance for output points yo

    """

    # compute the various kernel matrices
    Kxx = K(x, x)
    Kxox = K(xo, x)
    Kxxo = K(x, xo)
    Kxoxo = K(xo, xo)

    dot = np.dot
    inv = np.linalg.inv

    # estimate the mean and covariance of the function
    mean = dot(dot(Kxox, inv(Kxx)), y)
    _cov = Kxoxo - dot(dot(Kxox, inv(Kxx)), Kxxo)

    # round because we get floating point error around zero and end up
    # with negative variances along the diagonal
    cov = np.round(_cov, decimals=6)
    assert (np.diag(cov) >= 0).all()

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

    """

    def K(x1, x2):
        # compute constants to save on computation time
        out = np.empty((x1.size, x2.size))
        twopi = 2 * np.pi
        c = np.log((h ** 2) / np.sqrt(twopi * w))

        for i in xrange(x1.size):
            for j in xrange(x2.size):
                diff = x1[i] - x2[j]
                # log gaussian kernel
                out[i, j] = c + (-0.5 * (diff**2) / w**2)

        # transform the output out of log space
        out[:, :] = np.exp(out)

        return out

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(K)

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

    """

    def K(x1, x2):
        # compute constants to save on computation time
        out = np.empty((x1.size, x2.size))
        twopi = 2 * np.pi
        c = np.log((h ** 2) / np.sqrt(twopi * w))

        for i in xrange(x1.size):
            for j in xrange(x2.size):
                # compute circular difference between the points --
                # the idea being, if one is around 2*pi and the other
                # is around 0, they are actually very close
                d = x1[i] - x2[j]
                if abs(d) > np.pi:
                    diff = d - (np.sign(d) * twopi)
                else:
                    diff = d

                # log gaussian kernel
                out[i, j] = c + (-0.5 * (diff**2) / w**2)

        # transform the output out of log space
        out[:, :] = np.exp(out)

        return out

    # JIT compile with numba
    if jit:
        K = numba.jit('f8[:,:](f8[:],f8[:])')(K)

    return K
