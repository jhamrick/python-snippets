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
try:
    import numba
except ImportError:
    numba = None

from numpy import log, exp, dot
from numpy.linalg import inv

import circstats as circ
from safemath import MIN_LOG, MAX_LOG, normalize


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

    if np.isnan(Kxx).any():
        print K.params, s
        raise ArithmeticError("Kxx contains invalid values")

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
            print K.params, s
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


def _gaussian_kernel(x1, x2, h, w):
    """Produces a squared exponential (Gaussian) kernel function of
    the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{(x_1-x_2)^2}{2w^2})$$

    Parameters
    ----------
    h : number
        Output scale kernel parameter
    w : number
        Input scale (Gaussian standard deviation) kernel parameter

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # compute constants
    c = log(h ** 2)

    out = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            diff = x1[i] - x2[j]
            l = c + (-0.5 * (diff ** 2) / (w ** 2))

            # !!! underflow protection hack, because numba
            # currently can't handle catching/raising exceptions
            if l < MIN_LOG:
                out[i, j] = 0
            elif l > MAX_LOG:
                out[i, j] = np.inf
            else:
                out[i, j] = exp(l)

    return out
if numba:
    _gaussian_kernel = numba.jit(
        'f8[:,:](f8[:],f8[:],f8,f8)')(_gaussian_kernel)


def gaussian_kernel(h, w):
    """Produces a squared exponential (Gaussian) kernel function of
    the form:

    $$k(x_1, x_2) = h^2\exp(-\frac{(x_1-x_2)^2}{2w^2})$$

    Parameters
    ----------
    h : number
        Output scale kernel parameter
    w : number
        Input scale (Gaussian standard deviation) kernel parameter

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

    def kernel(x1, x2):
        return _gaussian_kernel(x1, x2, h, w)

    # save kernel parameters
    kernel.h = h
    kernel.w = w
    kernel.params = (h, w)

    return kernel


def _periodic_kernel(x1, x2, h, w, p):
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

    References
    ----------
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes
        for machine learning. MIT Press.

    """

    # compute constants
    c1 = log(h ** 2)
    c2 = -2. / (w ** 2)

    # compute constants to save on computation time
    out = np.empty((x1.size, x2.size))
    for i in xrange(x1.size):
        for j in xrange(x2.size):
            diff = x1[i] - x2[j]
            l = c1 + (c2 * np.sin(diff / (2. * p)) ** 2)

            # !!! underflow protection hack, because numba
            # currently can't handle catching/raising exceptions
            if l < MIN_LOG:
                out[i, j] = 0
            elif l > MAX_LOG:
                out[i, j] = np.inf
            else:
                out[i, j] = exp(l)

    return out
if numba:
    _periodic_kernel = numba.jit(
        'f8[:,:](f8[:],f8[:],f8,f8,f8)')(_periodic_kernel)


def periodic_kernel(h, w, p):
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

    def kernel(x1, x2):
        return _periodic_kernel(x1, x2, h, w, p)

    # save kernel parameters
    kernel.h = h
    kernel.w = w
    kernel.p = p
    kernel.params = (h, w, p)

    return kernel


def xcorr(x, y, circular=False, deg=False, nanrobust=False):
    """Returns matrix of correlations between x and y

    Parameters
    ----------
    x : np.ndarray
        Columns are different parameters, rows are sets of observations
    y : np.ndarray
        Columns are different parameters, rows are sets of observations
    circular : bool (default=False)
        Whether or not the data is circular
    deg : bool (default=False)
        Whether or not the data is in degrees (if circular)

    Returns
    -------
    out : np.ndarray
        Matrix of correlations between rows of x and y
    """

    # Inputs' shapes
    xshape = x.shape
    yshape = y.shape

    # Store original (output) shapes
    corrshape = xshape[:-1] + yshape[:-1]

    # Prepares inputs' shapes for computations
    if len(x.shape) > 2:
        x = x.reshape((np.prod(xshape[:-1]), xshape[-1]), order='C')
    if len(y.shape) > 2:
        y = y.reshape((np.prod(yshape[:-1]), yshape[-1]), order='C')

    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]

    if circular:
        if deg:
            x = np.radians(x)
            y = np.radians(y)

        # if nanrobust:
        #     corr = circ.nancorrcc(x, y, axis=1)
        # else:
        #     corr = circ.corrcc(x, y, axis=1, nancheck=False)
        if nanrobust:
            corr = circ.nancorrcc(x[:, :, None], y.T[None, :, :], axis=1)
        else:
            corr = circ.corrcc(x[:, :, None], y.T[None, :, :], axis=1)

    else:
        avgfn = np.mean
        stdfn = np.std

        # numerator factors (centered means)
        nx = (x.T - avgfn(x, axis=1)).T
        ny = (y.T - avgfn(y, axis=1)).T

        # denominator factors (std devs)
        sx = stdfn(x, axis=1)
        sy = stdfn(y, axis=1)

        # numerator
        num = np.dot(nx, ny.T) / x.shape[1]

        # correlation
        corr = num / np.outer(sx, sy)

    # reshape to take original
    corr = corr.reshape(corrshape, order='F')

    return corr
