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
parser.add_argument('--Ra', type=float, default = 1e6)
parser.add_argument('--tol', type=float, default = 1e-8)
parser.add_argument('--n', type=int, default = 40)
parser.add_argument('--CFL', type=float, default = 1.0)
parser.add_argument('--tend', type=float, default = 1)
parser.add_argument('--dim', type=int, default = 2)
parser.add_argument('--sparselu', action='store_true', default=False)
parser.add_argument('--secondorder', action='store_true', default=False)
parser.add_argument('--update_u_every', type=int, default = 1)
parser.add_argument('--adaptlayers', type=int, default=2)
parser.add_argument('--refine', type=int, default=0)
parser.add_argument('--graddiv_pen', type=float, default = 0.0)
parser.add_argument('--output', action='store_true', default=False)
parser.add_argument('--plot', action='store_true', default=False)

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

mesh = refine_boundary_layers(mesh, args.adaptlayers, range(dim), [0.0]*dim, [1.0]*dim)

for i in xrange(args.refine):
   mesh = refine(mesh)

#plot(mesh, interactive=True); exit()

ccfl = args.CFL
eps = Constant(1.0/args.Ra)

theta_init_expr = '1.0 - x[0]'
theta_init = Expression(theta_init_expr)

# function spaces
V  = VectorFunctionSpace(mesh, 'CG', 1)
Vd = VectorFunctionSpace(mesh, 'DG', 1)
Q  = FunctionSpace(mesh, 'CG', 1)
R  = FunctionSpace(mesh, 'R', 0)

W = MixedFunctionSpace([V, Q, R])
u, p, m = TrialFunctions(W)
v, q, r = TestFunctions(W)

wh = Function(W)
uh, ph, rh = wh.split()

theta = interpolate(theta_init, Q)
theta_0 = Function(Q)
theta_1 = theta
theta_2 = Function(Q)
theta_min = min(theta.vector()[:])
theta_max = max(theta.vector()[:])

h = CellSize(mesh)

nu = Constant(1)
ez = as_vector([0]*(dim-1) + [-1])
F = -ez*theta_1

delta = h**2*Constant(1./12)/nu
gamma = Constant(args.graddiv_pen)

# free-slip boundary conditions for velocity
bc_u = [ DirichletBC(W.sub(0), Constant([0.0]*dim), 'on_boundary') ]

# boundary conditions for temperature
left = CompiledSubDomain('on_boundary && near(x[0], 0.0)')
right    = CompiledSubDomain('on_boundary && near(x[0], 1.0)')
bc_theta = [ DirichletBC(Q, theta_init, left), DirichletBC(Q, theta_init, right) ]

# bilinear and linear forms
def a(u,v): return inner(2*nu*sym(grad(u)),grad(v))*dx
def b(v,q): return -div(v)*q*dx
def f(v):   return dot(F, v)*dx

# variational problem
stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       + gamma*div(u)*div(v)*dx \
       - dot(delta*grad(p), grad(q))*dx \
       + p*r*dx + m*q*dx
stokes_prec = inner(nu*grad(u),grad(v))*dx - p*q/nu*dx - m*r*dx - f(v)

t, i = 0.0, 0

log = []

U = Function(Vd)

# lumped mass
tmp = Vector(theta.vector())
p, q = TrialFunction(Q), TestFunction(Q)
l_mass = assemble(action(p*q*dx, Constant(1)))

# operators
A = assemble(lhs(stokes))
l = assemble(rhs(stokes))

if args.sparselu:
   solver = LUSolver()
   solver.set_operator(A)
else:
   amg = 'amg' if not has_krylov_solver_preconditioner('ml_amg') else 'ml_amg'
   solver = KrylovSolver('tfqmr', amg)
   solver.parameters['nonzero_initial_guess'] = True
   solver.parameters['relative_tolerance'] = 1e-8
   #solver.parameters['monitor_convergence'] = True
   P = assemble(lhs(stokes_prec))
   for bc in bc_u: bc.apply(P, l)
   solver.set_operators(A, P)


# Stokes part
def solve_stokes():

   # solve Stokes part
   l = assemble(rhs(stokes))
   for bc in bc_u: bc.apply(A, l)
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

   dudt = errornorm(theta, theta_0, 'L2')/dt
   print 't =', t, 'dt =', dt, 'its({0}/{1}) ='.format(i,args.update_u_every), its, \
         'rms =', rms, '|du/dt| =', dudt

   log.append((t, rms))
   if dudt < args.tol: break

   if args.plot:
      plot(theta, mode='color')

if args.output:
   File('output/cavity-{0}-{1}-{2}-GWW14.pvd'.format(args.Ra, args.n*2**args.refine, args.correct)) << theta

#interactive()
