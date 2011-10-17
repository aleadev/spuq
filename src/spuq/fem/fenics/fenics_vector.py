from spuq.fem.fem_vector import FEMVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from dolfin import Function, FunctionSpaceBase, GenericVector
from dolfin.cpp import GenericFunction
from numpy import array, empty

class FEniCSVector(FEMVector):
    '''Wrapper for FEniCS/dolfin Function'''
    
    def __init__(self, coeffs=None, basis=None, function=None):
        '''Initialise with coefficient vector and FEMBasis'''
        if basis is not None:
            assert(function is None)
            assert(isinstance(coeffs, GenericVector))
            if isinstance(basis, GenericFunction):
                basis = FEniCSBasis(functionspace=basis.function_space())
            elif isinstance(basis, FunctionSpaceBase):
                basis = FEniCSBasis(functionspace=basis)
            assert(isinstance(basis,FEniCSBasis))
            self._basis = basis
            self._coeffs = coeffs
            self._F = Function(basis.functionspace, coeffs)
        else:
            assert(function is not None)
            assert(isinstance(function, GenericFunction))
            self._basis = FEniCSBasis(functionspace=function.function_space())
            self._coeffs = function.vector()
            self._F = function

    @property        
    def basis(self):
        return self._basis
    
    @property
    def coeffs(self):
        return self._coeffs
    
    @property
    def F(self):
        return self._F
    
    def evaluate(self, x):
        val = empty([0,0])
        self._F.eval(val, x)
        return val
    