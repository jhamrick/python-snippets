from nose.tools import raises

import numpy as np
import scipy.stats
np.seterr(all='raise')

######################################################################
# normalize
######################################################################

from stats import normalize

N_big = 20
N_small = 5
thresh = 1e-6


def rand_h():
    h = np.random.uniform(0, 2)
    return h


def rand_w():
    w = np.random.uniform(np.pi / 32., np.pi / 2.)
    return w


def rand_p():
    p = np.random.uniform(0.33, 3)
    return p


def check_normalization_constants(arr, axis):
    sum = np.log(np.sum(arr, axis=axis))
    z = normalize(np.log(arr), axis=axis)[0]

    zdiff = np.abs(sum - z)
    if not (zdiff < thresh).all():
        print sum
        print z
        raise AssertionError("wrong normalization constant")


def check_normalization(arr, axis):
    sum = np.sum(arr, axis=axis)
    norm = np.log(arr / np.expand_dims(sum, axis=axis))
    n = normalize(np.log(arr), axis=axis)[1]

    ndiff = np.abs(norm - n)
    if not(ndiff < thresh).all():
        print norm
        print n
        raise AssertionError("wrong normalized values")


def test_normalize_10():
    """Test stats.normalize for a vector"""
    for i in xrange(N_big):
        arr = np.random.gamma(2, scale=2, size=10)
        yield (check_normalization_constants, arr, 0)
        yield (check_normalization, arr, 0)


def test_normalize_5x10x15():
    """Test stats.normalize for a multidimensional array"""
    for i in xrange(N_big):
        arr = np.random.gamma(2, scale=2, size=(5, 15, 20))
        for axis in xrange(3):
            yield (check_normalization_constants, arr, axis)
            yield (check_normalization, arr, axis)


def test_normalize_2x100000():
    """Test stats.normalize for a large array"""
    for i in xrange(N_small):
        arr = np.random.gamma(2, scale=2, size=(2, 100000))
        for axis in xrange(2):
            yield (check_normalization_constants, arr, axis)
            yield (check_normalization, arr, axis)


######################################################################
# gaussian processes
######################################################################

from stats import GP, gaussian_kernel, periodic_kernel


def check_gaussian_kernel(x, dx, h, w):
    kernel = gaussian_kernel(h, w, jit=False)
    K = kernel(x, x)
    print "Kernel parameters:", kernel.params

    pdx = scipy.stats.norm.pdf(dx, loc=0, scale=w)
    pdx *= (h ** 2) * np.sqrt(2 * np.pi) * w

    diff = abs(pdx - K)
    if not (diff < thresh).all():
        print diff
        raise AssertionError("invalid kernel matrix")


def check_periodic_kernel(x, dx, h, w, p):
    kernel = periodic_kernel(h, w, p, jit=False)
    K = kernel(x, x)
    print "Kernel parameters:", kernel.params

    pdx = (h ** 2) * np.exp(-2. * (np.sin(dx / (2. * p)) ** 2) / (w ** 2))

    diff = abs(pdx - K)
    if not (diff < thresh).all():
        print diff
        raise AssertionError("invalid kernel matrix")


@raises(ValueError)
def test_gaussian_kernel_params1():
    """Test invalid h parameter to stats.gaussian_kernel"""
    gaussian_kernel(0, 1, False)


@raises(ValueError)
def test_gaussian_kernel_params2():
    """Test invalid w parameter to stats.gaussian_kernel"""
    gaussian_kernel(1, 0, False)


# def test_gaussian_kernel_jit():
#     gaussian_kernel(1, 1, True)


def test_gaussian_kernel():
    """Test stats.gaussian_kernel output matrix"""
    x = np.linspace(-2, 2, 10)
    dx = x[:, None] - x[None, :]
    for i in xrange(N_big):
        h, w = np.random.gamma(2, scale=2, size=2)
        yield (check_gaussian_kernel, x, dx, h, w)


@raises(ValueError)
def test_periodic_kernel_params1():
    """Test invalid h parameter to stats.periodic_kernel"""
    periodic_kernel(0, 1, 1, False)


@raises(ValueError)
def test_periodic_kernel_params2():
    """Test invalid w parameter to stats.periodic_kernel"""
    periodic_kernel(1, 0, 1, False)


# def test_periodic_kernel_jit():
#     periodic_kernel(1, 1, 1, True)


def test_periodic_kernel():
    """Test stats.periodic_kernel output matrix"""
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    dx = x[:, None] - x[None, :]
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        yield (check_periodic_kernel, x, dx, h, w, p)


def check_GP(kernel, x, y, s):
    print "Kernel parameters:", kernel.params
    mean, cov = GP(kernel, x, y, x, s=s)
    diff = abs(y - mean)
    if not (diff < thresh).all():
        print "Error:"
        print diff
        raise AssertionError("incorrect GP predictions")


def test_gaussian_GP():
    x = np.linspace(-2*np.pi, 2*np.pi, 16)
    y = np.sin(x)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        kernel = gaussian_kernel(h, w, jit=False)
        yield (check_GP, kernel, x, y, 0)


def test_periodic_GP():
    # can't do -2*pi to 2*pi because they are the same -- the
    # resulting covariance matrix will be singular!
    x = np.linspace(-2*np.pi, 2*np.pi-np.radians(1), 16)
    y = np.sin(x)
    for i in xrange(N_big):
        h = rand_h()
        w = rand_w()
        p = rand_p()
        kernel = periodic_kernel(h, w, p, jit=False)
        yield (check_GP, kernel, x, y, 0)
