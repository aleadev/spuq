"""Implements a class for generating and storing sets of multiindices"""

import numpy as np
import scipy as sp


class MultiindexSet(object):
    def __init__(self, arr):
        self.arr = arr
        self.m = arr.shape[1]
        self.p = max(arr.sum(1))
        self.count = arr.shape[0]
        pass
    
    def power(self,vec):
        for i in xrange(vec.size):
            if i==0:
                res = vec[i]**self.arr[:,i]
            else:
                res = res*vec[i]**self.arr[:,i]
        return res
        
    def factorial(self):
        return sp.factorial(self.arr).prod(1)

    @property
    def is_zero(self):
        return bool(len(sp.nonzero(self.arr)[0])) 

    @staticmethod
    def createCompleteOrderSet(m,p):
        def createMultiindexSet(m, p):
            #from numpy import int8,  zeros,  vstack, hstack
            if m==0:
                return np.zeros( (1, 0),  np.int8 )
            else:
                I = np.zeros( (0, m),  np.int8 )
                for q in xrange(0, p+1):
                    J = createMultiindexSet(m-1, q)
                    Jn = q-J.sum(1).reshape((J.shape[0],1))
                    I = np.vstack( (I,  np.hstack( (J, Jn))))
                return I
        arr=createMultiindexSet(m,p)
        return MultiindexSet(arr)

    def __getitem__(self,i):
        return self.arr[i]
        
    def __repr__(self):
        return "MI(m={0},p={1},arr={2})".format(self.m,self.p,self.arr)
