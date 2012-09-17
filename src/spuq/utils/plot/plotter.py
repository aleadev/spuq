from numpy import unique, nonzero

try:
    import matplotlib
    matplotlib.use('WxAgg')
#    matplotlib.interactive(True)
    import pylab
    # import plot, show, axis, subplot, xlabel, ylabel, hist, figure
    HAS_PYLAB = True
except Exception, e:
    HAS_PYLAB = False

try:
    from mayavi import mlab
    HAS_MAYAVI = True
except Exception, e:
#    import traceback
#    print traceback.format_exc()
    print "mayavi is not available"
    HAS_MAYAVI = False
except (Exception, SystemExit) as e:
#    import traceback
#    print traceback.format_exc()
    print "mayavi is not available"
    HAS_MAYAVI = False

#from spuq.fem.fenics.fenics_vector import FEniCSVector
#from spuq.utils.type_check import takes, anything, optional


class Plotter(object):
    def __init__(self):
        pass
    
    @staticmethod
    def hasMayavi():
        return HAS_MAYAVI
    
    @staticmethod
    def hasPylab():
        return HAS_PYLAB

    
    # ========================================
    # ============ pylab methods =============
    # ========================================
    
    @staticmethod
    def scatterplot(self, indices, data, withgrid=False):
        x = unique(indices[:, 0])
        y = unique(indices[:, 1])
        M = len(y)
        N = len(x)
        for ai, bi in indices:
            xi = nonzero(x == ai)[0][0]
            yi = nonzero(y == bi)[0][0]
            print 'subplot: ', (ai, bi), (xi, yi), (M, N), N * (M - 1) + xi + 1 - yi * N
            pylab.subplot(M, N, N * (M - 1) + xi + 1 - yi * N)
            pylab.plot(data[ai, :], data[bi, :], 'b.')
            if xi == 0:
                pylab.ylabel('y[' + str(bi) + ']')
            if yi == 0:
                pylab.xlabel('x[' + str(ai) + ']')
        pylab.axis('equal')
        if withgrid:
            pylab.grid(True)
        pylab.show()

    @staticmethod
    def histplot(data, bins=10, normed=False, weights=None, cumulative=False, bottom=None, histtype='bar'):
        M = data.shape[0]
        N = 1
        for i in range(M - 1):
            pylab.subplot(M, N, i + 1)
            pylab.hist(data[i, :], bins=bins, normed=normed, weights=weights, cumulative=cumulative, bottom=bottom, histtype=histtype)

    @staticmethod
    def figure(num=None, mayavi=True, **kwargs):
        if not mayavi:
            assert HAS_PYLAB
            return pylab.figure(num)
        else:
            assert HAS_MAYAVI
            return mlab.figure(num, **kwargs)


    # ========================================
    # =========== mayavi methods =============
    # ========================================

    @staticmethod
    def plotMesh(coordinates, triangles, values=None, **kwargs):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.triangular_mesh
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        if values is None:
            values = 0 * x
        mlab.triangular_mesh(x, y, values, triangles, **kwargs)

    @staticmethod
    def labels(xlabel="x", ylabel="y", zlabel="z", obj=None):
        if xlabel:
            mlab.xlabel(xlabel, obj)
        if ylabel:
            mlab.ylabel(ylabel, obj)
        if zlabel:
            mlab.zlabel(zlabel, obj)

    @staticmethod
    def show(func=None, stop=False):
        mlab.show(func, stop)

    @staticmethod
    def axes(*args, **kwargs):
        mlab.show(args, kwargs)

    @staticmethod
    def title(*args, **kwargs):
        mlab.title(*args, **kwargs)

    @staticmethod
    def close(scene=None, allfig=True):
        mlab.close(scene=scene, all=allfig)
