
    @takes((tuple,list,ndarray), RandomVariable, MultiVector, MultiindexSet, CoefficientField, MultiVector, int)
    def evaluateFlux(self, x, rv, wN, mu, CF, projected_wN, maxm=10, pt=FEniCSBasis.PROJECTION.INTERPOLATION):
        """Evaluate numerical flux :math:`\sigma_\mu(w_N)` according to EGSZ (5.1)

            ..math:: \sigma_\mu(w_N) := \overline{a}\nabla w_{N,\mu} + \sum_{m=1}^\infty a_m\nabla(\alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m}w_{N,\mu+e_m}
                                                    -\alpha_{\mu_m}^m}w_{N,\mu} + \alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m}w_{N,\mu-e_m})
        """
#        newDelta = extend(Delta)
#
#        for mu in newDelta:
#            sigma_x = a[0]( w[mu].mesh.nodes ) * w[mu].dx()
#            for m in xrange(1,100):
#                mu1 = mu.add( (m,1) )
#                if mu1 in Delta:
#                    sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]+1) *\
#                        w[mu1].project( w[mu].mesh ).dx()
#                    mu2 = mu.add( (m,-1) )
#                if mu2 in Delta:
#                    sigma_x += a[m]( w[mu].mesh.nodes ) * beta(m, mu[m]) *\
#                        w[mu2].project( w[mu].mesh ).dx()
        # TODO: Lambda and extension of Lambda?!?
        if isinstance(x, tuple) or isinstance(list):
            x = array(x)

        # prepare cache data structures
        if not projected_wN:
            projected_wN = MultiVector()
        if mu not in projected_wN.keys():
            projected_wN[mu] = MultiVector()

        sigma = MultiVector()
        a0_f, a0_rv = CF[0]
        for mu in wN.active_set():
            val = a0_f*wN[mu].dx()(x)
            for m in range(1,maxm):
                # prepare polynom coefficients
                am_f, am_rv = CF[m]
                p = am_rv.orth_poly
                (a, b, c) = p.recurrence_coefficients(mu[m])
                beta = (a/b, 1/b, c/b)

                # prepare projections of wN
                # mu+1
                mu1 = mu.add( (m,1) )
                if mu1 not in projected_wN[mu].keys():
                    projected_wN[mu][mu1] = wN[mu].functionspace.project(wN[mu1], pt)
                # mu-1
                mu2 = mu.add( (m,-1) )
                if mu2 not in projected_wN[mu].keys():
                    projected_wN[mu][mu2] = wN[mu].functionspace.project(wN[mu2], pt)

                # mu+1
                val += am_f*beta[1]*projected_wN[mu][mu1].dx()(x)
                # mu+1
                val -= am_f*beta[0]*wN[mu].dx()(x)
                # mu-1
                val += am_f*beta[-1]*projected_wN[mu][mu2].dx()(x)
                sigma[mu] = val
        return sigma


    @takes((tuple,list,ndarray), RandomVariable, MultiVector, MultiindexSet, CoefficientField, MultiVector, int)
    def evaluateDivFlux(self, x, rv, wN, mu, CF, projected_wN, maxm=10, pt=FEniCSBasis.PROJECTION.INTERPOLATION):
        """Evaluate divergence of numerical flux :math:`\nabla\cdot\sigma_\mu(w_N)` according to EGSZ (5.4)

            ..math:: \nabla\cdot\sigma_\mu(w_N) = \nabla\overline{a}\cdot\nabla w_{N,\mu} + \sum_{m=1}^\infty \nabla a_m\cdot\nabla(
                                                    \alpha_{\mu_m+1}^m\Pi_\mu^{\mu+e_m}w_{N,\mu+e_m} - \alpha_{\mu_m}^m w_{N,\mu}
                                                    +\alpha_{\mu_m-1}^m\Pi_\mu^{\mu-e_m}w_{N,\mu-e_m})

            Note that this form is only valid for conforming piecewise affine approximation spaces (P1 FEM).
        """
        # TODO: Lambda and extension of Lambda?!?
        if isinstance(x, tuple) or isinstance(x, list):
            x = array(x)

        # prepare cache data structures
        if not projected_wN:
            projected_wN = MultiVector()
        if mu not in projected_wN.keys():
            projected_wN[mu] = MultiVector()

        sigma = MultiVector()
        a0_f, a0_rv = CF[0]
        for mu in wN.active_set():
            val = inner(a0_f.diff(x), wN[mu].dx()(x))
            for m in range(1,maxm):
                # prepare polynom coefficients
                am_f, am_rv = CF[m]
                p = am_rv.orth_poly
                (a, b, c) = p.recurrence_coefficients(mu[m])
                beta = (a/b, 1/b, c/b)

                # prepare projections of wN
                # mu+1
                mu1 = mu.add( (m,1) )
                if mu1 not in projected_wN[mu].keys():
                    projected_wN[mu][mu1] = wN[mu].functionspace.project(wN[mu1], pt)
                # mu-1
                mu2 = mu.add( (m,-1) )
                if mu2 not in projected_wN[mu].keys():
                    projected_wN[mu][mu2] = wN[mu].functionspace.project(wN[mu2], pt)

                # mu+1
                val += beta[1]*inner(am_f.diff(x), projected_wN[mu][mu1].dx()(x))
                # mu+1
                val -= beta[0]*inner(am_f.diff(x), wN[mu].dx()(x))
                # mu-1
                val += beta[-1]*inner(am_f.diff(x), projected_wN[mu][mu2].dx()(x))
                sigma[mu] = val
        return sigma

