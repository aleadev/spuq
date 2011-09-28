


w = MultiVector()

m = MultiIndex( () )

T0 = IntialMesh()

w[m] = FenicsVector( T0 )

...

# sigma_mu

# a = (Function, Function, Function, ... )

newDelta = extend(Delta)

for mu in newDelta:
    sigma_x = a[0]( w[mu].mesh.nodes ) * w[mu].dx() 
    for m in xrange(1,100):
        mu1 = mu.add( (m,1) )
        if mu1 in Delta:
            sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]+1) *\
                      w[mu1].project( w[mu].mesh ).dx()
        mu2 = mu.add( (m,-1) )
        if mu2 in Delta:
            sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]) *\
                      w[mu2].project( w[mu].mesh ).dx()
                      

# operator A

A = MultiOperator( a, rvs )

def MultiOperator.apply( w ):
    v = MultiVector()
    delta = w.active_set()
    for mu in delta:
        A0 = stiffness( a[0], w[mu].basis )
        v[mu] = A0 * w[mu] 
        for m in xrange(1,100):
            Am = stiffness( a[m], w[mu].basis )
            mu1 = mu.add( (m,1) )
            if mu1 in Delta:
                v[mu] += Am * beta(m, mu[m] + 1) * w[mu1].project(w[mu].mesh)
            mu2 = mu.add( (m,-1) )
            if mu2 in Delta:
                v[mu] += Am * beta(m, mu[m]) * w[mu2].project(w[mu].mesh)
    return v

