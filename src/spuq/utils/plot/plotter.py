from numpy import unique, nonzero, zeros, sum

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

from dolfin import Function

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
            fig = mlab.figure(num, bgcolor=(0.75, 0.75, 0.75), size=(800, 600), **kwargs)


    # ========================================
    # =========== mayavi methods =============
    # ========================================

    @classmethod
    def plotMesh(cls, coordinates, triangles=None, values=None, axes=True, displacement=False, newfigure=True, scale=1, **kwargs):
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.triangular_mesh
        if newfigure:
            cls.figure()
        if isinstance(coordinates, Function) and triangles is None:
            coordinates, triangles, values = cls._function_data(coordinates)
        else:
            assert triangles is not None
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        representation = "surface"
        scalars = None
        if displacement:
            representation = "wireframe"
            assert values.shape[1] == 2
            x = coordinates[:, 0] + scale * values[:, 0]
            y = coordinates[:, 1] + scale * values[:, 1]
            scalars = sum(values, axis=1)
        else:
            assert values is None or len(values.shape) == 1
        if values is None or displacement:
            values = 0 * x
        mlab.triangular_mesh(x, y, values, triangles, representation=representation, scalars=scalars, **kwargs)
        if axes:
            cls.axes()

    @staticmethod
    def _function_data(f):
        mesh = f.function_space().mesh()
        coordinates = mesh.coordinates()
        N = coordinates.shape[0]
        cells = mesh.cells()
        nss = f.function_space().num_sub_spaces()
        # NOTE: since these are nodal values, the coefficients would just have to be assigned appropriately
        if nss == 2:
            values = zeros((N, 2))
        else:
            values = zeros(N)
        for i, c in enumerate(coordinates):
            values[i] = f(c)
        return coordinates, cells, values

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
        mlab.axes(color=(1, 1, 1), line_width=2, nb_labels=3)

    @staticmethod
    def title(*args, **kwargs):
        mlab.title(*args, **kwargs)

    @staticmethod
    def close(scene=None, allfig=True):
        mlab.close(scene=scene, all=allfig)
