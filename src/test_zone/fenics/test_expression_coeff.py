from dolfin import *

exp_str = "1+sin(A*pi*x[0])*sin(A*pi*x[1])"
M = 20

# construct set of expressions
exp_set = [Expression(exp_str, A=m + 1, degree=3) for m in range(M)]
E1 = sum(exp_set)

# construct long expression string
exp_str_M = "+".join([exp_str.replace('A', 'A%i' % m) for m in range(M)])
init_A = ','.join(['A%i=%f' % (m, m + 1) for m in range(M)])
exec_str = "E2 = Expression('" + exp_str_M + "', %s, degree=3)" % init_A
print "EXEC:", exec_str
exec(exec_str, globals(), locals())

N = 300
mesh = UnitSquare(N, N)
V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(u), nabla_grad(v))
a1 = E1 * a * dx
a2 = E2 * a * dx
L = Constant(1) * v * dx

import time
T = [(time.time(), "assemble A1")]
A1 = assemble(a1)
T.append((time.time(), "assemble A2"))
A2 = assemble(a2)
T.append((time.time(), "assemble L"))
b = assemble(L)
T.append((time.time(), "apply BC"))

tol = 1e-14
def u0_boundary(x, on_boundary):
    return (abs(x[1]) <= tol or abs(x[1] - 1) <= tol) and on_boundary
bc = DirichletBC(V, Constant(0), u0_boundary)

bc.apply(A1, b)
bc.apply(A2)
u1, u2 = Function(V), Function(V)
T.append((time.time(), "solve u1"))
solve(A1, u1.vector(), b)
T.append((time.time(), "solve u2"))
solve(A2, u2.vector(), b)
T.append((time.time(), ""))

T_str = [(T[j + 1][0] - T[j][0], T[j][1]) for j in range(len(T) - 1)]
print "TIMING:"
def pT(x):
    print "\t%s:\t%f" % (x[1], x[0])
map(lambda x:pT(x) , T_str)

plot(u1, title="u1")
plot(u2, title="u2")
interactive()
