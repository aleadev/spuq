v = SomeVector(FooBasis)

# transpose a vector gives a vector in the dual space 
vt = v.T

# same class as SomeVector but "dual"
# coefficients are coefficients of v * FooBasis.gramian
# vt.coeffs = FooBasis.gramian *
# vt.basis = v.basis.dual  

vt * v # scalar, inner product
# same as inner(v,v)
v * vt # operator from FooBasis to FooBasis


# Basis gets a flag to indicate whether its dual or not

# Gramian must be an operator from CanonicalBasis to CanonicalBasis since 
# it works on the stripped coefficient vectors

# For canonical or orthonormal bases it can be IdentityOperator or MultOperator(1)   

