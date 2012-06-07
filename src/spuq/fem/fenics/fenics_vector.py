from dolfin import Function, FunctionSpace, plot, File, Mesh, norm

from spuq.utils.type_check import takes, anything, sequence_of, set_of
from spuq.linalg.vector import Scalar
from spuq.linalg.basis import check_basis
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fem_vector import FEMVector

import pickle
import os
import logging
logger = logging.getLogger(__name__)

class FEniCSVector(FEMVector):
    '''Wrapper for FEniCS/dolfin Function.

        Provides a FEniCSBasis and a FEniCSFunction (with the respective coefficient vector).'''

    @takes(anything, Function)
    def __init__(self, fefunc):
        '''Initialise with coefficient vector and Function.'''
        self._fefunc = fefunc

    @classmethod
    @takes(anything, FEniCSBasis)
    def from_basis(cls, basis):
        f = Function(basis._fefs)
        return FEniCSVector(f)

    def copy(self):
        return self._create_copy(self.coeffs.copy())

    @property
    def basis(self):
        '''return FEniCSBasis'''
        return FEniCSBasis(self._fefunc.function_space())

    @property
    def coeffs(self):
        '''Return FEniCS coefficient vector of Function.'''
        return self._fefunc.vector()

    @coeffs.setter
    def coeffs(self, val):
        '''Set FEniCS coefficient vector of Function.'''
        self._fefunc.vector()[:] = val

    def array(self):
        '''Return copy of coefficient vector as numpy array.'''
        return self._fefunc.vector().array()

    def eval(self, x):
        return self._fefunc(x)

    def _create_copy(self, coeffs):
        # TODO: remove create_copy and retain only copy()
        new_fefunc = Function(self._fefunc.function_space(), coeffs)
        return self.__class__(new_fefunc)

    @takes(anything, (set_of(int), sequence_of(int)))
    def refine(self, cell_ids=None, with_prolongation=False):
        (new_basis, prolongate, _) = self.basis.refine(cell_ids)
        if with_prolongation:
            return prolongate(self)
        else:
            return FEniCSVector(Function(new_basis._fefs))

    def interpolate(self, f):
        self._fefunc.interpolate(f)
        return self

    def __eq__(self, other):
        """Compare vectors for equality.

        Note that vectors are only considered equal when they have
        exactly the same type."""
#        print "************* EQ "
#        print self.coeffs.array()
#        print other.coeffs.array()
#        print (type(self) == type(other),
#                self.basis == other.basis,
#                self.coeffs.size() == other.coeffs.size())
        return (type(self) == type(other) and
                self.basis == other.basis and
                self.coeffs.size() == other.coeffs.size() and
                (self.coeffs == other.coeffs).all())

    @takes(anything)
    def __neg__(self):
        return self._create_copy(-self.coeffs)

    @takes(anything, "FEniCSVector")
    def __iadd__(self, other):
        check_basis(self.basis, other.basis)
        self.coeffs += other.coeffs
        return self

    @takes(anything, "FEniCSVector")
    def __isub__(self, other):
        check_basis(self.basis, other.basis)
        self.coeffs -= other.coeffs
        return self

    @takes(anything, Scalar)
    def __imul__(self, other):
        self.coeffs *= other
        return self

    @takes(anything, "FEniCSVector")
    def __inner__(self, other):
        v1 = self._fefunc.vector()
        v2 = other._fefunc.vector()
        return v1.inner(v2)

    def plot(self, **kwargs):
        func = self._fefunc
        # fix a bug in the fenics plot function that appears when 
        # the maximum difference between data values is very small 
        # compared to the magnitude of the data 
        values = func.vector().array()
        diff = max(values) - min(values)
        magnitude = max(abs(values))
        if diff < magnitude * 1e-8:
            logger.warning("PLOT: function values differ only by tiny amount -> plotting as constant")
            func = Function(func.function_space())
            func.vector()[:] = values[0]
        plot(func, **kwargs)

    def norm(self, norm_type="L2"):
        return norm(self._fefunc, norm_type)
        
    @property
    def min_val(self):
        return min(self._fefunc.vector().array())
        
    @property
    def max_val(self):
        return max(self._fefunc.vector().array())

    @property
    def degree(self):
        return self.basis.degree

    def pickle(self, outdir, filename):
        """construct representation suitable for pickling"""
        logger.info("pickling data (and mesh) to file %s", os.path.join(outdir, 'DATA_' + filename + '.pkl'))
        basis = self.basis._fefs
        ufl = basis.ufl_element()
        mesh = basis.mesh()
        meshfile = File(os.path.join(outdir, 'MESH_' + filename + '.xml'))
        meshfile << mesh
        data = (self.array(), (ufl.family(), ufl.degree()))
        with open(os.path.join(outdir, 'DATA_' + filename + '.pkl'), 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def from_pickle(cls, indir, filename):
        """unpickle object"""
        logger.info("unpickling data (and mesh) from file %s", os.path.join(indir, 'DATA_' + filename + '.pkl'))
        mesh = Mesh(os.path.join(indir, 'MESH_' + filename + '.xml'))
        with open(os.path.join(indir, 'DATA_' + filename + '.pkl'), "rb") as f:
            data = pickle.load(f) 
        fs = FunctionSpace(mesh, data[1][0], data[1][1])
        f = Function(fs)
        f.coeffs = data[0]
        return cls(f)
