from spuq.fem.fem_vector import FEMVector
from spuq.fem.fenics.fenics_basis import FEniCSBasis
from spuq.fem.fenics.fenics_function import FEniCSFunction
from dolfin import Function, FunctionSpaceBase, GenericVector, Vector
from dolfin.cpp import GenericFunction
from numpy import empty

class FEniCSVector(FEMVector):
    '''Wrapper for FEniCS/dolfin Function.

        Provides a FEniCSBasis and a FEniCSFunction (with the respective coefficient vector).'''
    
    def __init__(self, coeffs=None, basis=None, function=None):
        '''Initialise with coefficient vector and FEMBasis'''
        if basis:
            assert function==None
            if not isinstance(basis, FEniCSBasis):
                assert isinstance(basis, FunctionSpaceBase)
                basis = FEniCSBasis(functionspace=basis)
            self._basis = basis
            if coeffs == None:
                coeffs = Vector(basis.functionspace.dim)
            assert isinstance(coeffs, GenericVector)
            self._F = FEniCSFunction(Function(basis.functionspace, coeffs))
        else:
            assert function!=None and isinstance(function, (Function, FEniCSFunction))
            if isinstance(function, FEniCSFunction):
                self._F = function
            else:
                assert isinstance(function, Function)
                self._F = FEniCSFunction(function)
            self._basis = FEniCSBasis(functionspace=self._F.function_space())
            self._coeffs = self._F.f.vector()
    
    @property
    def F(self):
        '''return FEniCSFunction'''
        return self._F
    
    @property
    def function(self):
        '''return underlying fenics Function'''
        return self._F.f

    @property
    def basis(self):
        '''return FEniCSBasis'''
#        if not hasattr(self, '_FBasis'):
#            self._FBasis = FEniCSBasis(functionspace=self.functionspace)
        return self._basis

    @property
    def functionspace(self):
        '''return fenics FunctionSpace'''
        return self._F.function_space()

    @property
    def dim(self):
        '''return dimension of function space'''
        return self.functionspace.dim()

    @property
    def coeffs(self):
        '''return (assignable) fenics coefficient vector of Function'''
        return self._F.coeffs

    def array(self):
        '''return copy of coefficient vector as numpy array'''
        return self._F.array()

    def evaluate(self, x):
        val = empty([0,0])
        self._F.eval(val, x)
        return val
