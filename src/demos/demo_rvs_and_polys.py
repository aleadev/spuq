# Import the random variables
import spuq.stochastics.random_variable as rvs
import numpy as np

# Now create e.g. a Beta distributed variable
rv = rvs.BetaRV(alpha=1.3, beta=2.2)

# We frequently need the 'x' polynomial, so lets define it here
x = np.poly1d([1, 0])

# The mean of this variable is the same as the integral of x over the domain
# of the rv with the given weight function. Lets check this:   

print "Mean of this rv: %s == %s " % (rv.mean, rv.integrate(x))

# We can do the same with the variance and the skewness   

print "Variance of this rv: %s == %s " % (rv.var, rv.integrate((x-rv.mean)**2))
print "Skewness of this rv: %s == %s " % (rv.skew, rv.integrate((x-rv.mean)**3/rv.var**1.5))

# Get the orthonormal polynomials of this variable
p = rv.orth_polys

# Lets see whether they are really orthonormal

print "int(p_2 p_2 w) == %s ~= 1" % rv.integrate(p[2]*p[2])
print "int(p_3 p_3 w) == %s ~= 1" % rv.integrate(p[3]*p[3])
print "int(p_2 p_3 w) == %s ~= 0" % rv.integrate(p[2]*p[3])
print "int(p_1 p_4 w) == %s ~= 0" % rv.integrate(p[1]*p[4])
print "int(p_4 p_2 w) == %s ~= 0" % rv.integrate(p[4]*p[2])

# Now some rather high degree polynomials 
print "int(p_17 p_18 w) == %s ~= 0 (ooh, that's bad)" % rv.integrate(p[17]*p[18])
print "int(p_17 p_18 w) == %s ~= 0 (ok, better)" % \
    rv.integrate(lambda x: p.eval(17,x)*p.eval(18,x))
print "int(p_70 p_71 w) == %s ~= 0 (works too)" % \
    rv.integrate(lambda x: p.eval(70,x)*p.eval(71,x))

# Mow we can check the recurrence coefficients

(a3,b3,c3)=p.recurrence_coefficients(3)
print "p_4 == "
print p[4]
print b3*x*p[3] + a3 * p[3] - c3 * p[2]

