from spuq.bases.polynomial_basis import PolynomialBasis

class GPCBasis(PolynomialBasis):
    def __init__(self, I,  rvs):
        assert( I.m==len(rvs) )
        self.I=I
        self.rvs=rvs
        
    def sample(self, n):
        from numpy import array,  zeros,  ones, ix_
        S=ones((self.I.count, n))
        for i,  rv in enumerate(self.rvs):
            theta=rv.sample(n)
            Phi=rv.getOrthogonalPolynomials()
            Q=zeros((self.I.p+1, n))
            for q in xrange(self.I.p+1):
                Q[q, :]=Phi.eval( q,  theta )
            S=S*Q[self.I.arr[:,i], :]
        return S
        

if __name__=="__main__":
    from spuq.utils.multiindex_set import *
    I=MultiindexSet.createCompleteOrderSet(2, 4)
    print I

    from spuq.statistics import *
    N=NormalDistribution()
    U=UniformDistribution()
    print N,  U

    gpc=GPCBasis(I, [N, U])
    print gpc.sample(3)

# X=U(2,5) (Y=X^2)
# gpc=GPCBasis(MI, {Uniform()})
# x=(3.5, 2, 0, 0, 0, 0)
# (x'*gpc.sample())^2
# M=gpc.mult_tensor()
# y(i)=M(i,j,k)*x(j)*x(k)
# (y'*gpc.sample())



# X=U(2,5) (1/X)
# gpc=GPCBasis(MI, {Uniform()})
# x=(3.5, 2, 0, 0, 0, 0)
# 1/(x'*gpc.sample())
# z=(1,0,0,0,0,0)
# M=gpc.mult_tensor()
# # x(i)=M(i,j,k)*z(j)*y(k)
# D=M(i,j,k)*x(j)
# y=D\z
# y'*gpc.sample()


# multiplication tensor: first as numpy array 3d
# then: 
# class MultiplicationTensor
# def contract( vector ) -> (sparse) matrix
# def contract( vector, vector ) -> vector
# storage: full, sparse matrix, own sparse format?


class MultiplicationTensor:
    def __init__(self):
        #self.M=
        pass
        
    #def contract( self,  vector )
    def contract( self, x, y ):
        # z=array(n)
        for i in xrange(n1):
            for j in xrange(n2):
                for k in xrange(n3):
                    z[i]=self.M[i, j, k]*x[j]*y[k]
                    



# one dimensional PCE or GPC of a random variable
# X in 1d GPC base on distribution D
# distribution of X esp. inverse cdf of X
# and cdf of D
# i.e. GPC is expansion of phi=X.invcdf*D.cdf 
# gpc functions are psi_i
# then i-th gpc coefficient x_i=int( phi psi_i p_D )/int(psi_i^2)

# x_i=int( phi psi_i p_D )/int(psi_i^2)
# if not basis.isorthonormal()
#     G=basis.gramian()
#     x=G\x

# # alternative
# x_i=int( phi psi_i p_D )/int(psi_i^2)
# G=basis.gramian(for_solving=true)
# x=G\x

# def Basis.gramian( self, for_solving=false )
#    if for_solving and isorthonogmal
#        return linear_operator.from_scalar( 1, dim, dim )
#    elif for_solving and isdiagonal
#        return linear_operator.from_diagonal( d )
#    else 
#        return full_gramian(...)

# check whether basis is orthonormal (assert, or use different algorithm)
# gpc basis could call normalise for non-normalised bases




