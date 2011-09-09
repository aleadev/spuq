"""Demo program that solves a simple elliptic SPDE on a rectangular domain that is decomposed
into two subdomains where the diffusion coefficient varies homogeneously but independent of the 
other subdomain respectively. This demo doesn't actually run, but is an attempt at how writing 
code with the finished library could look like."""


# define the first paramter to be normally distributed with mean 2 and variance 0.3
# and expand in a pce
param1_dist=NormalDistribution(2,0.3);
rv1=expand_pce_1d( param1_dist, degree=3 )
# rv1 is a random variable, contains the basis and the expansion coefficients
# can get them via rv1.basis, rv1.coeffs. The coefficients rv1.coeffs should 
# be something like [2,0.3,0,0]

# expand the second parameter into a gpc based on a 
# uniform distribution up to order p1
param2_dist=BetaDistribution(2,3);
rv2=expand_gpc_1d(var2, UniformDistribution(), degree=2 )

# create the full tensor product basis
(basis,proj1,proj2)=create_full_tensor_basis( rv1.basis, rv2.basis )

# or something like (does not make full sense as it stands here)
MI=create_complete_multiindex_set( 2, (p1, p2) )
(basis,proj1,proj2)=create_sparse_tensor_basis( rv1.basis, rv2.basis, MI )

# projects the random variables into the bigger space
rv1=project_random_variable( rv1, proj1 )
rv2=project_random_variable( rv2, proj2 )

# some how define the geometry of the problem [-1,1]x[0,1]
# geom=..., not much clue about that

# define the random field as I(x<=0)*rv1+I(x>=0)*rv2, i.e. both side vary independently
# with the distributions specified in param1_dist and param2_dist
spatial_function1="x<=0"
random_field1=create_simple_random_field( geom.evaluate( spatial_function1 ), rv1 )

spatial_function2="x>=0"
random_field2=create_simple_random_field( geom.evaluate( spatial_function2 ), rv2 )

# or maybe
random_field1=create_simple_random_field( geom.get_subdomain_indicator_function(1), rv1 )
random_field2=create_simple_random_field( geom.get_subdomain_indicator_function(2), rv2 )

# add the fields (should generate a seperated representations if the addends are, otherwise 
# add "pointwise"; further check that the bases match)
coefficient_field=random_field1+random_field2

# generate the linear operator from this information
# something like that would be nice
operator=create_differential_operator( -div*coefficient_field*grad, geom ) 
# that would be acceptable, too (at least for the beginning)
operator=create_operator_for_diffusion_eq( coefficient_field, geom ) 

# deterministic right hand side (so this is a random field with rv==1)
rhs_field=create_deterministic_field( geom.evaluate( "sin(x)*cos(y)"), basis )

# solve it (direct solve would create a big matrix from the operator definition and
# a large vector from the rhs, then throw it at some direct solver, and wrap it up later in 
# a random field class)
solution_field=direct_solve( operator, rhs_field )

# plot mean and variance of the solution
plot_field( geom, solution_field.mean() )
plot_field( geom, solution_field.var() )


