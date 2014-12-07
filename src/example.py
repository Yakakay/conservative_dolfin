from dolfin import *

# compile fvm module
header_file = open('masscorrection.hh')
code = '\n'.join(header_file.readlines())
masscorrection = compile_extension_module(code)

import argparse
parser = argparse.ArgumentParser(description = 'demo for mass conservation.')
parser.add_argument('--correct', action='store_true', default=False)
parser.add_argument('--p', type=int, default = 1)
parser.add_argument('--k', type=int, default = 4)
parser.add_argument('--level', type=int, default = 0)
args = parser.parse_args()
print args

set_log_level(ERROR)

N = 2**args.level
mesh = UnitSquareMesh(N, N, 'crossed')

U = Expression(('sin(k*x[0]*pi)*cos(x[1]*pi)', '-k*cos(k*x[0]*pi)*sin(x[1]*pi)'), k=args.k, pi=pi)
F = Expression(('0', '-cos(k*x[0]*pi)*sin(x[1]*pi)*((k*k + 2)*k*k + 1)/k'), k=args.k, pi=pi)

V = VectorFunctionSpace(mesh, 'CG', args.p)
Q = FunctionSpace(mesh, 'CG', 1 if args.p is 1 else 1)
R = FunctionSpace(mesh, 'R',  0)
X = MixedFunctionSpace([V, Q, R])

u, p, m = TrialFunctions(X)
v, q, r = TestFunctions(X)

# boundary conditions (pure Dirichlet)
bc_u = [ DirichletBC(X.sub(0), U, lambda x, on_boundary: on_boundary) ]

# bilinear and linear forms
def a(u,v): return inner(grad(u),grad(v))*dx
def b(v,q): return -div(v)*q*dx
def f(v):   return dot(F, v)*dx

h = CellSize(mesh)
delta = Constant(1./12 if args.p is 1 else 0.)

stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       + m*q*dx + r*p*dx - h**2*dot(delta*grad(p), grad(q))*dx

xh = Function(X)
solve(lhs(stokes) == rhs(stokes), xh, bc_u)
uh, ph, rh = xh.split()

# mass correction
if args.correct:
  div_uh = project(div(uh), FunctionSpace(mesh, 'DG', args.p-1));
  if args.p is 1:
    uh = project(uh - h**2*delta*grad(ph), VectorFunctionSpace(mesh, 'DG', 1));

  masscorrection.compute_correction(mesh, uh, div_uh, Q, args.p)

#plot(ph)
#interactive()