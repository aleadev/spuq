"""EGSZ a posteriori residual estimator"""

class ResidualEstimator(object):
    """ """
    
    def evaluateVolumeEstimator(self):
        pass

    def evaluateEdgeEstimator(self):
        pass

    def evaluateFlux(self):
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

    def evaluateProjectionError(self):
        pass
