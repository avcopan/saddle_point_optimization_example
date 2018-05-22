import numpy
from typing import Iterable
from itertools import starmap


A = numpy.array([-200, -100, -170, 15])
B = numpy.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
C = numpy.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                 [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])


def func(x):
    eqfs = starmap(_exponential_quadratic_function, zip(A, B, C))
    return sum(eqf(x) for eqf in eqfs)


def grad(x):
    eqgs = starmap(_exponential_quadratic_gradient, zip(A, B, C))
    return sum(eqg(x) for eqg in eqgs)


def hess(x):
    eqhs = starmap(_exponential_quadratic_hessian, zip(A, B, C))
    return sum(eqh(x) for eqh in eqhs)


def _exponential_quadratic_function(a, b, c):

    def _f(x):
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return a * numpy.exp(numpy.sum(dx * cdx, axis=0))

    return _f


def _exponential_quadratic_gradient(a, b, c):

    def _g(x):
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return 2 * a * numpy.exp(numpy.sum(dx * cdx, axis=0)) * cdx

    return _g


def _exponential_quadratic_hessian(a, b, c):

    def _h(x):
        dx = x - _cast(b, 0, numpy.ndim(x))
        cdx = numpy.tensordot(c, dx, axes=(1, 0))
        return (2 * a * numpy.exp(numpy.sum(dx * cdx, axis=0)) *
                (_cast(c, (0, 1), numpy.ndim(x) + 1)
                 + 2 * _insert(cdx, 1) * _insert(cdx, 0)))

    return _h


def _cast(a, ax, ndim=None):
    ax = tuple(ax) if isinstance(ax, Iterable) else (ax,)
    assert numpy.ndim(a) == len(ax)
    ndim = max(ax) + 1 if ndim is None else ndim
    ix = (slice(None) if i in ax else None for i in range(ndim))
    at = numpy.transpose(a, numpy.argsort(ax))
    return at[tuple(ix)]


def _insert(a, ax):
    ax = tuple(ax) if isinstance(ax, Iterable) else (ax,)
    ndim = numpy.ndim(a) + 1
    ix = (None if i in ax else slice(None) for i in range(ndim))
    return a[tuple(ix)]
