from dolfin import *

set_log_level(ERROR)

from utilities import *
set_dolfin_optimisation()

from dolfin_fvm import *

import numpy

import argparse
parser = argparse.ArgumentParser(description = 'demo with inexact solvers.')
parser.add_argument('--dim', type=int, default = 3)
parser.add_argument('--level', type=int, default = 4)
parser.add_argument('--atol', type=float, default = 1e-20)
args = parser.parse_args()
print args

# parameters
N = 2**args.level
tend = 100
ccfl = .25
Ra = Constant(1.0)
cinit = Constant(1)
theta = Expression('c*cos(k*x[0]*pi)*cos(k*x[1]*pi)*sin(x[2]*pi)', k = 2, c=1) \
        if args.dim is 3 else Expression('c*cos(k*x[0]*pi)*sin(x[1]*pi)', k = 2, c=1)

# mesh
if args.dim is 2:
   mesh = UnitSquareMesh(N, N)
elif args.dim is 3:
   mesh = UnitCubeMesh(N, N, N)

print 'N =', N, 'elements =', mesh.size(args.dim)

# function spaces
V = VectorFunctionSpace(mesh, 'CG', 1)
Vd = VectorFunctionSpace(mesh, 'DG', 1)
Q = FunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(mesh, 'R',  0)
X = MixedFunctionSpace([V, Q, R])

u, p, m = TrialFunctions(X)
v, q, r = TestFunctions(X)

# free-slip boundary conditions for velocity
bc_u = [ DirichletBC(X.sub(0).sub(i), Constant(0.0), \
         'on_boundary && (near(x[{0}], 0.0) || near(x[{0}], 1.0))'.format(i)) \
         for i in xrange(args.dim) ]

# PSPG stabilization
h = CellSize(mesh)
delta = Constant(1./12)*h**2

# forcing
ez = as_vector([0]*(args.dim-1) + [-1])
F = -ez*Ra*theta

# bilinear and linear forms
def a(u,v): return inner(grad(u),grad(v))*dx
def b(v,q): return -div(v)*q*dx
def f(v):   return dot(F, v)*dx

# variational problem
stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       + m*q*dx + r*p*dx - dot(delta*grad(p), grad(q))*dx
stokes_prec = a(u, v) - p*q*dx - f(v)

# set up operators and solver
A, bb = assemble_system(lhs(stokes), rhs(stokes), bc_u)
P, btmp = assemble_system(lhs(stokes_prec), rhs(stokes_prec), bc_u)
amg = 'amg' if not has_krylov_solver_preconditioner('ml_amg') else 'ml_amg'
solver = KrylovSolver('tfqmr', amg)
solver.set_operators(A, P)
solver.parameters['absolute_tolerance'] = args.atol

for rtol in [1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
   
   xh = Function(X)
   solver.parameters['relative_tolerance'] = rtol
   its = solver.solve(xh.vector(), bb)
   uh, ph, rh = xh.split()
   
   # residual
   res = norm(A*xh.vector()-bb)

   # compute corrected flux
   uh_corr = project(uh - delta*grad(ph), Vd)

   # compute CFL
   umax = numpy.max(numpy.abs(uh_corr.vector().array()))
   dt = ccfl*mesh.hmin()/umax
   
   # advect the constant temperature field for
   # uncorrected and corrected fluxes
   err = []
   for U in [uh, uh_corr]:
   
      c, c0 = interpolate(cinit, Q), Function(Q)
      t = 0.0
      
      while t < tend + DOLFIN_EPS:

         c0.assign(c)

         advance(c, c0, U, dt)
         c.vector().apply('insert')

         t += dt
      
      err.append(errornorm(interpolate(Constant(1), Q), c, 'L2'))
      
   print r'{0} & {1:1.6e} & {2:1.6e} & {3:1.6e} & {4:1.6e} \\'.format(its, rtol, res, err[0], err[1])
