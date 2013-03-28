from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class Covariance:
    __metaclass__ = ABCMeta
    
    def __init__(self, isisotropic=False, ishomogeneous=False):
        self._isisotropic = isisotropic
        self._ishomgeneous = ishomogeneous

    @property
    def isotropic(self):
        return self._isisotropic
    
    @property
    def homogeneous(self):
        return self._ishomogeneous
    
    def __call__(self, r):
        return self.evaluate(r)

    @abstractmethod
    def evaluate(self, r):
        raise NotImplementedError
    

class GaussianCovariance(Covariance):
    def __init__(self, sigma, a):
        self.sigma = sigma
        self.a = a
        
    def evaluate(self, r):
        return self.sigma**2*np.exp(-r**2/self.a**2)
    

class ExponentialCovariance(Covariance):
    def __init__(self, sigma, a):
        self.sigma = sigma
        self.a = a
        
    def evaluate(self, r):
        return self.sigma**2*np.exp(-np.abs(r)/self.a)

    
class TransformedCovariance(Covariance):
    def __init__(self):
        pass

    
class InterpolatedCovariance(Covariance):
    def __init__(self):
        pass

    
class LognormalTransformedCovariance(Covariance):
    def __init__(self):
        pass
        