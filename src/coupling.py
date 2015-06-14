#! /usr/bin/env python

from dolfin import *
from utilities import *
set_dolfin_optimisation()

from dolfin_fvm import *

import numpy

import argparse
parser = argparse.ArgumentParser(description = 'coupling demo')
parser.add_argument('--correct', action='store_true', default=False)
parser.add_argument('--limit', action='store_true', default=False)
parser.add_argument('--tend', type=float, default = 1)
parser.add_argument('--Ra', type=float, default = 1e4)
parser.add_argument('--CFL', type=float, default = 1.0)
parser.add_argument('--dim', type=int, default = 2)
parser.add_argument('--n', type=int, default = 40)
parser.add_argument('--sparselu', action='store_true', default=False)
parser.add_argument('--secondorder', action='store_true', default=False)
parser.add_argument('--update_u_every', type=int, default = 1)
parser.add_argument('--adaptlayers', type=int, default=2)
parser.add_argument('--refine', type=int, default=0)
parser.add_argument('--output', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)
parser.add_argument('--a', type=float, default = 0.01)
parser.add_argument('--b', type=float, default = ln(1000))
parser.add_argument('--c', type=float, default = 0.0)
parser.add_argument('--k', type=int, default = 4)
parser.add_argument('--dT', type=float, default = 1)

args = parser.parse_args()
print args

# problem definition
if args.dim is 2:
   mesh = UnitSquareMesh(args.n, args.n, 'crossed')
elif args.dim is 3:
   mesh = UnitCubeMesh(args.n, args.n, args.n)
else:
   print 'dimension {0} not supported!'.format(args.dim)
   exit()

vol = assemble(1*dx(mesh))
dim = mesh.topology().dim()

mesh = refine_boundary_layers(mesh, args.adaptlayers, [dim-1], [0.0], [1.0])

for i in xrange(args.refine):
   mesh = refine(mesh)

#plot(mesh, interactive=True); exit()

ccfl = args.CFL
eps = Constant(1.0/args.Ra)

if dim is 2:
   theta_init_expr = 'dT*(1.0 - x[1]*x[1]) + a*cos(k*pi*x[0])*sin(pi*x[1])'
else:
   theta_init_expr = 'dT*(1.0 - x[2]*x[2]) + a*cos(k*pi*x[0])*cos(k*pi*x[1])*sin(pi*x[2])'
theta_init = Expression(theta_init_expr, a = args.a, dT = args.dT, k = args.k)

#viscosity = lambda xmin, x: exp(-2*(xmin - x))
b = Constant(args.b)
c = Constant(args.c)
nu0 = Constant(1.0)
x = SpatialCoordinate(mesh)
viscosity = lambda deltaT, theta: nu0*exp(-b*theta/deltaT + c*(1.0 - x[dim-1]))

# function spaces
V  = VectorFunctionSpace(mesh, 'CG', 1)
Vd = VectorFunctionSpace(mesh, 'DG', 1)
Q  = FunctionSpace(mesh, 'CG', 1)
R  = FunctionSpace(mesh, 'R', 0)

W = MixedFunctionSpace([V, Q, R])
u, p, m = TrialFunctions(W)
v, q, r = TestFunctions(W)

wh = Function(W)
uh, ph, mh = wh.split()

theta = interpolate(theta_init, Q)
theta_0 = Function(Q)
theta_1 = theta
theta_2 = Function(Q)
theta_min = min(theta.vector()[:])
theta_max = max(theta.vector()[:])

h = CellSize(mesh)

nu = viscosity(theta_max-theta_min, theta)

delta = h**2*Constant(1./12)/nu
ez = as_vector([0]*(dim-1) + [-1])
F = -ez*theta_1

# free-slip boundary conditions for velocity
bc_u = [ DirichletBC(W.sub(0).sub(i), Constant(0.0), \
         'on_boundary && (near(x[{0}], 0.0) || near(x[{0}], 1.0))'.format(i)) \
         for i in xrange(dim) ]

# boundary conditions for temperature
bottom = CompiledSubDomain('on_boundary && near(x[{0}], 0.0)'.format(dim-1))
top    = CompiledSubDomain('on_boundary && near(x[{0}], 1.0)'.format(dim-1))
bc_theta = [ DirichletBC(Q, theta_init, top), DirichletBC(Q, theta_init, bottom) ]

# bilinear and linear forms
def a(u,v): return inner(2*nu*sym(grad(u)),grad(v))*dx
def b(v,q): return -div(v)*q*dx
def f(v):   return dot(F, v)*dx

# variational problem
stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       - dot(delta*grad(p), grad(q))*dx + p*r*dx + m*q*dx
stokes_prec = inner(nu*grad(u),grad(v))*dx - p*q/nu*dx - m*r*dx - f(v)

t, i = 0.0, 0

# for Nusselt
markers = FacetFunction('size_t', mesh)
markers.set_all(0)
bottom.mark(markers, 1)
top.mark(markers, 2)
ds = Measure('ds')[markers]

log = []

U = Function(Vd)

# lumped mass
tmp = Vector(theta.vector())
p, q = TrialFunction(Q), TestFunction(Q)
l_mass = assemble(action(p*q*dx, Constant(1)))

if args.sparselu:
   solver = LUSolver()
else:
   amg = 'amg' if not has_krylov_solver_preconditioner('ml_amg') else 'ml_amg'
   solver = KrylovSolver('tfqmr', amg)
   solver.parameters['nonzero_initial_guess'] = True
   solver.parameters['relative_tolerance'] = 1e-8
   #solver.parameters['monitor_convergence'] = True

# Stokes part
def solve_stokes():
   # solve Stokes part
   if not(args.b == 0.0 and t > 0): # save assembly if mu is only depth-dependent
      A = assemble(lhs(stokes))
   l = assemble(rhs(stokes))
   for bc in bc_u: bc.apply(A, l)

   if args.sparselu:
      solver.set_operator(A)
   else: # assemble preconditioner
      if not(args.b == 0.0 and t > 0):
         P = assemble(lhs(stokes_prec))
         for bc in bc_u: bc.apply(P, l)
         solver.set_operators(A, P)

   its = solver.solve(wh.vector(), l)

   # corrected mass flux
   U.assign(project(uh - delta*grad(ph) if args.correct else uh, Vd))

   return its


# temperature part
def advance_temperature(theta_new, theta_old):

   # diffusion part
   assemble(dot(-eps*grad(theta_old), grad(q))*dx, tensor = tmp)
   for idx, tmpidx in enumerate(tmp):
      tmp[idx] = tmpidx/l_mass[idx]

   # advection part
   theta_new.assign(theta_old)
   advance(theta_new, theta_old, U, dt)
   theta_new.vector().axpy(dt, tmp)
   theta_new.vector().apply('insert')


its = solve_stokes() # initial field

while t < args.tend + DOLFIN_EPS:

   # compute CFL
   umax = numpy.max(numpy.abs(U.vector().array()))
   #umax = sqrt(numpy.max(project(dot(U,U), Q).vector().array()))
   dt = ccfl*min(0.25*args.Ra*mesh.hmin()**2, mesh.hmin()/umax)

   theta_0.assign(theta)

   # theta^(1) = theta^n + dt*L(theta^n)
   advance_temperature(theta_1, theta_0)

   # intermediate solve
   if not (i % args.update_u_every):
      its = solve_stokes()
   else:
      its = 0

   if args.secondorder: # explicit SSP(2,2) Runge-Kutta
   
      # theta^(2) = theta^(1) + dt*L(theta^(1))
      advance_temperature(theta_2, theta_1)
   
      # theta^{n+1} = 0.5*(theta^n + theta^(2))
      theta.vector()[:] = 0.0
      theta.vector().axpy(0.5, theta_0.vector())
      theta.vector().axpy(0.5, theta_2.vector())

   else: # explicit Euler
   
      # theta^{n+1} = theta^(1) = theta^n + dt*L(theta^n)
      theta.assign(theta_1)

   for bc in bc_theta: bc.apply(theta.vector())
   #theta.vector().apply('insert')

   if(args.limit): # limit to physical bounds
      theta.vector()[theta.vector() < theta_min] = theta_min
      theta.vector()[theta.vector() > theta_max] = theta_max

   t += dt; i = i + 1

   rms = norm(U)/vol
   nusselt = assemble(-grad(theta)[dim-1]*ds(2))/assemble(theta*ds(1))

   print 't =', t, 'its({0}/{1}) ='.format(i,args.update_u_every), its, \
         'rms =', rms, 'Nu =', nusselt

   log.append((t, rms, nusselt))

   if args.plot:
      plot(theta, mode='color')

import time
tstr = time.strftime('%Y%m%d-%H%M%S')
basestr = 'output/theta-' + tstr

if args.output:
   File(basestr + '.pvd') << theta

   with open(basestr + '.txt', 'w') as text_file:
      text_file.write(str(args) + '\n\n')
      for l in log:
         text_file.write('{0:.4e}, {1:.4e}, {2:.4e}\n'.format(l[0], l[1], l[2]))

#interactive()
