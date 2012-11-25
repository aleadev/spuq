class FEMNavierLame(FEMDiscretisationBase):
    """FEM discrete Navier-Lame equation (linearised elasticity) with parameters :math:`E` and :math:`\nu` with provided boundary conditions.

        ..math:: -\mathrm{div}a \nabla u = 0 \qquad\textrm{in }\Omega
        ..math:: u = 0 \qquad\textrm{on }\partial\Omega

        ..math:: \int_D a\nabla \varphi_i\cdot\nabla\varphi_j\;dx
    """

    @property
    def norm(self):
        return self.get_norm()
    
    def get_norm(self, mesh=None):
        '''Energy norm wrt operator, i.e. (\sigma(v),\eps(v))=||C^{1/2}\eps(v)||.'''
        if mesh is None:
            return lambda v: np.sqrt(assemble(inner(self.sigma(self.lmbda0, self.mu0, v), sym(nabla_grad(v))) * dx))
        else:
            DG = FunctionSpace(mesh, "DG", 0)
            s = TestFunction(DG)
            def energy_norm(v):
                ae = np.sqrt(assemble(inner(self.sigma(self.lmbda0, self.mu0, v), sym(nabla_grad(v))) * s * dx))
                # reorder DG dofs wrt cell indices
                dofs = [DG.dofmap().cell_dofs(c.index())[0] for c in cells(mesh)]
                norm_vec = ae[dofs]
                return norm_vec
            return energy_norm



    def assemble_rhs(self, lmbdamu, basis, withDirichletBC=True, withNeumannBC=True, f=None):
        """Assemble the discrete right-hand side."""
        if f is None:
            f = self._f
        Dirichlet_boundary = self._dirichlet_boundary
        uD = self._uD

        # get FEniCS function space
        V = basis._fefs
        # define linear form
        u = TrialFunction(V)
        v = TestFunction(V)

        lmbda = lmbdamu[0] 
        mu = lmbdamu[1]
        a = inner(self.sigma(lmbda, mu, u), sym(nabla_grad(v))) * dx
        l = inner(f, v) * dx
        
        # treat Neumann boundary
        if withNeumannBC and self._neumann_boundary is not None:
            Ng, ds = self._prepareNeumann(V.mesh())            
            for j in range(len(Ng)):
                l += dot(Ng[j], v) * ds(j + 1)
                        
        if withDirichletBC:
            bcs = self.create_dirichlet_bcs(V, self._uD, self._dirichlet_boundary)
        else:
            bcs = []

        # assemble linear form
        _, F = assemble_system(a, l, bcs)
        return F
            
