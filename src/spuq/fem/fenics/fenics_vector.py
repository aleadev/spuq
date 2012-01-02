from spuq.fem.fem_vector import FEMVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from dolfin import Function, FunctionSpaceBase, GenericVector
from dolfin.cpp import GenericFunction
from numpy import empty

class FEniCSVector(FEMVector):
    '''Wrapper for FEniCS/dolfin Function'''
    
    def __init__(self, coeffs=None, basis=None, function=None):
        '''Initialise with coefficient vector and FEMBasis'''
        if basis != None:
            assert function==None and isinstance(coeffs, GenericVector)
            if not isinstance(basis, FEniCSBasis):
                assert isinstance(basis, FunctionSpaceBase)
                basis = FEniCSBasis(functionspace=basis)
            self._basis = basis
            self._coeffs = coeffs
            self._F = FEniCSFunction(Function(basis.functionspace, coeffs))
        else:
            assert function!=None and isinstance(function, (Function, FEniCSFunction))
            if isinstance(function, FEniCSFunction):
                self._F = function
            else:
                assert isinstance(function, Function)
                self._F = FEniCSFunction(function)
            self._basis = FEniCSBasis(functionspace=self._F.f.function_space())
            self._coeffs = self._F.f.vector()
    
    @property
    def F(self):
        return self._F
    
    @property
    def function(self):
        return self._F.f

    def evaluate(self, x):
        val = empty([0,0])
        self._F.eval(val, x)
        return val
