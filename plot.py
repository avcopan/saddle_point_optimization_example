import numpy
import matplotlib.pyplot

from surf import func


def local_min(a):
    return numpy.logical_and(numpy.greater(numpy.roll(a, -1), a),
                             numpy.greater(numpy.roll(a, +1), a))


def local_max(a):
    return numpy.logical_and(numpy.less(numpy.roll(a, -1), a),
                             numpy.less(numpy.roll(a, +1), a))


def circle(theta):
    return numpy.vstack((numpy.cos(theta), numpy.sin(theta)))


xmin, xmax = -1.50, 0.75
ymin, ymax = -0.50, 2.00
X = numpy.linspace(xmin, xmax)
Y = numpy.linspace(ymin, ymax)
Z = func(numpy.array(numpy.meshgrid(X, Y)))

R = 0.4
P = numpy.array((-1., 1.25))
T = numpy.linspace(0, 2 * numpy.pi, 100)
C = numpy.add(P[:, None], R * circle(T))
F = func(C)
MINS = local_min(F)
MAXS = local_max(F)
C_MINS = C[:, MINS]
C_MAXS = C[:, MAXS]
F_MINS = func(C_MINS)
F_MAXS = func(C_MAXS)
T_MINS = T[MINS]
T_MAXS = T[MAXS]

# contour plot with circle
matplotlib.pyplot.figure(figsize=(10, 5))
matplotlib.pyplot.subplot(1, 2, 1)
matplotlib.pyplot.contour(X, Y, Z, 100)
matplotlib.pyplot.scatter(P[0], P[1], zorder=10)
matplotlib.pyplot.plot(C[0], C[1])
matplotlib.pyplot.scatter(C_MINS[0], C_MINS[1], color='r', zorder=10)
matplotlib.pyplot.scatter(C_MAXS[0], C_MAXS[1], color='g', zorder=10)
matplotlib.pyplot.axis('equal')

matplotlib.pyplot.subplot(1, 2, 2)
matplotlib.pyplot.plot(T, func(C))
matplotlib.pyplot.scatter(T_MINS, F_MINS, color='r', zorder=10)
matplotlib.pyplot.scatter(T_MAXS, F_MAXS, color='g', zorder=10)
matplotlib.pyplot.savefig('figs/potential.png')
matplotlib.pyplot.savefig('figs/potential{:.1f}.png'.format(R))
