from spuq.fem.fem_vector import FEMVector
# TODO: eliminate circular inclusion (by importing spuq.fem.fenics module/package instead)
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from dolfin import Function, FunctionSpaceBase, GenericVector
from dolfin.cpp import GenericFunction
from numpy import array

class FEniCSVector(FEMVector):
    '''wrapper for FEniCS/dolfin Function'''
    
    def __init__(self, coeffs=None, basis=None, function=None):
        '''initialise with coefficient vector and FEMBasis'''
        if basis is not None:
            assert(function is None)
            assert(isinstance(coeffs, GenericVector))
            if isinstance(basis, GenericFunction):
                basis = FEniCSBasis(functionspace=basis.function_space())
            elif isinstance(basis, FunctionSpaceBase):
                basis = FEniCSBasis(functionspace=basis)
            assert(isinstance(basis,FEniCSBasis))
            self.__basis = basis
            self.__coeffs = coeffs
            self.__F = Function(basis.functionspace, coeffs)
        else:
            assert(function is not None)
            assert(isinstance(function, GenericFunction))
            self.__basis = FEniCSBasis(functionspace=function.function_space())
            self.__coeffs = function.vector()
            self.__F = function

    @property        
    def basis(self):
        return self.__basis
    
    @property
    def coeffs(self):
        return self.__coeffs
    
    @property
    def F(self):
        return self.__F
    
    def evaluate(self, x):
        val = array([])
        self.__F.eval(val, x)
        return val
    