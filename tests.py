import numpy as np
np.seterr(all='raise')

N_big = 20
N_small = 5
thresh = 1e-6

from safemath import normalize
from safemath import safe_log
from safemath import safe_multiply


######################################################################
# normalize
######################################################################

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
# safe_log
######################################################################

def test_safe_log():
    arr = np.random.rand(10, 10) + 1
    log_arr1 = np.log(arr)
    log_arr2 = safe_log(arr)
    assert (log_arr1 == log_arr2).all()


######################################################################
# safe_multiply
######################################################################

def test_safe_multiply():
    arr1 = np.random.randn(10, 10)
    arr2 = np.random.randn(10, 10)
    prod1 = arr1 * arr2
    prod2 = safe_multiply(arr1, arr2)
    diff = np.abs(prod1 - prod2)
    if not (diff < 1e-8).all():
        print prod1
        print prod2
        raise ValueError
