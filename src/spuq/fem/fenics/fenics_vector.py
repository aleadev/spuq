from spuq.fem.fem_vector import FEMVector
# TODO: eliminate circular inclusion (by importing spuq.fem.fenics module/package instead)
#from spuq.fem.fenics.fenics_basis import FEniCSBasis
from dolfin import Function, FunctionSpace, FunctionSpaceFromCpp, Expression
from numpy import array

class FEniCSVector(FEMVector):
    '''wrapper for FEniCS/dolfin Function'''
    
    def __init__(self, coeffs=None, basis=None, function=None):
        '''initialise with coefficient vector and FEMBasis'''
        if basis is not None:
            assert(function is None)
#            assert(isinstance(basis,FEniCSBasis))
# TODO: assume FEniCSBasis for now - FunctionSpace should be supported as well
            self._basis = basis
            self._coeffs = coeffs
            if not (isinstance(basis, FunctionSpace) or isinstance(basis, FunctionSpaceFromCpp)):    # TODO: check for FEniCSBasis instead!
                basis = basis.basis
            self._F = Function(basis, coeffs)
        else:
            assert(function is not None)
            assert(isinstance(function, Function))
            self._basis = function.function_space()
            self._coeffs = function.vector()
            self._F = function

    @property        
    def basis(self):
        return self._basis
    
    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, val):
        self._coeffs = val
        self._F = Function(self._basis, self._coeffs)
    
    @property
    def F(self):
        return self._F
    
    @F.setter
    def F(self, val):
        self._F = val
    
    def evaluate(self, x):
        val = array([])
        self._F.eval(val, x)
        return val
    