from nose.tools import raises

import numpy as np
import scipy.stats
np.seterr(all='raise')

######################################################################
# normalize
######################################################################

from stats import normalize


def check_normalization_constants(arr, axis):
    sum = np.log(np.sum(arr, axis=axis))
    z = normalize(np.log(arr), axis=axis)[0]

    zdiff = np.abs(sum - z)
    if not (zdiff < 1e-8).all():
        print sum
        print z
        raise AssertionError("wrong normalization constant")


def check_normalization(arr, axis):
    sum = np.sum(arr, axis=axis)
    norm = np.log(arr / np.expand_dims(sum, axis=axis))
    n = normalize(np.log(arr), axis=axis)[1]

    ndiff = np.abs(norm - n)
    if not(ndiff < 1e-8).all():
        print norm
        print n
        raise AssertionError("wrong normalized values")


def test_normalize_10():
    """Test stats.normalize for a vector"""
    for i in xrange(5):
        arr = np.random.gamma(2, scale=2, size=10)
        yield (check_normalization_constants, arr, 0)
        yield (check_normalization, arr, 0)


def test_normalize_5x10x15():
    """Test stats.normalize for a multidimensional array"""
    for i in xrange(5):
        arr = np.random.gamma(2, scale=2, size=(5, 15, 20))
        for axis in xrange(3):
            yield (check_normalization_constants, arr, axis)
            yield (check_normalization, arr, axis)


def test_normalize_2x100000():
    """Test stats.normalize for a large array"""
    for i in xrange(1):
        arr = np.random.gamma(2, scale=2, size=(2, 100000))
        for axis in xrange(2):
            yield (check_normalization_constants, arr, axis)
            yield (check_normalization, arr, axis)


######################################################################
# gaussian processes
######################################################################

from stats import GP, gaussian_kernel, circular_gaussian_kernel


@raises(ValueError)
def test_gaussian_kernel_params1():
    gaussian_kernel(0, 1, False)


@raises(ValueError)
def test_gaussian_kernel_params2():
    gaussian_kernel(1, 0, False)


# def test_gaussian_kernel_jit():
#     gaussian_kernel(1, 1, True)


def test_gaussian_kernel():
    x = np.linspace(-2, 2, 10)
    dx = x[:, None] - x[None, :]
    pdx = scipy.stats.norm.pdf(dx, loc=0, scale=1)
    pdx *= np.sqrt(2 * np.pi)
    kernel = gaussian_kernel(1, 1, jit=False)
    K = kernel(x, x)

    diff = abs(pdx - K)
    if not (diff < 1e-8).all():
        print pdx
        print K
        raise AssertionError("invalid kernel matrix")
