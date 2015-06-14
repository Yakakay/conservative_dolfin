from dolfin import *
from utilities import *
from utilities.icosahedral_sphere_mesh import *
set_dolfin_optimisation()
from dolfin_fvm import *
import numpy as np

import argparse
parser = argparse.ArgumentParser(description = 'coupling demo')
parser.add_argument('--correct', action='store_true', default=False)
parser.add_argument('--limit', action='store_true', default=False)
parser.add_argument('--tend', type=float, default = 1.0e3)
parser.add_argument('--rmin', type=float, default = 0.55)
parser.add_argument('--rmax', type=float, default = 1.0)
parser.add_argument('--Ra', type=float, default = 1e4)
parser.add_argument('--CFL', type=float, default = 1.0)
parser.add_argument('--level', type=int, default = 4)
parser.add_argument('--secondorder', action='store_true', default=False)
parser.add_argument('--update_u_every', type=int, default = 1)
parser.add_argument('--noslip', action='store_true', default=False)
parser.add_argument('--output', action='store_true', default=False)
parser.add_argument('--output_every', type=float, default = 0.0)
parser.add_argument('--plot', action='store_true', default=False)

parameters['form_compiler']['quadrature_degree'] = 2

args = parser.parse_args()
print args

# spherical shell mesh
layers = np.linspace(args.rmin, args.rmax, 2**args.level+1)
mesh = IcosahedralSphereMesh(args.level, layers)

# some model parameters
ccfl = args.CFL
eps = Constant(1.0/args.Ra)
nu = Constant(1.0)

# spherical coordinates
# note: angles needed to be shifted for some reason, and there was no other elegant way
radius = Expression('sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])')
angle1 = Expression('a < 0.0 ? a + 2*pi : a', a = Expression('atan2(x[1],x[0])'))
angle2 = Expression('a < 0.0 ? a + 2*pi : a', a = Expression('acos(x[2]/r)', r = radius))
s = Expression('(r-rmin)/(rmax-rmin)', rmin = args.rmin, rmax = args.rmax, r = radius)

# initial temperature profile with spherical harmonic perturbation; see Zhong et. al (2008)
theta_expr = 'rmin/r*(r-rmax)/(rmin-rmax) + c*sin(pi*s)*(sin(2*phi)+cos(2*phi))*cos(psi)*sin(psi)*sin(psi)'
theta_init = Expression(theta_expr, \
                        c=0.01*sqrt(7.0/(240*pi))*15, s = s, \
                        phi = angle1, psi = angle2, \
                        rmin = args.rmin, rmax = args.rmax, r = radius)

# function spaces
V  = VectorFunctionSpace(mesh, 'CG', 1)
Vd = VectorFunctionSpace(mesh, 'DG', 1)
Q  = FunctionSpace(mesh, 'CG', 1)
R  = FunctionSpace(mesh, 'R', 0)
W = MixedFunctionSpace([V, Q, R])

# trial and test functions
u, p, m = TrialFunctions(W)
v, q, r = TestFunctions(W)

# solution vector
wh = Function(W)
uh, ph, mh = wh.split()

# temperature vectors
theta = interpolate(theta_init, Q)
theta_0 = Function(Q)
theta_1 = theta
theta_2 = Function(Q)
theta_min = min(theta.vector()[:])
theta_max = max(theta.vector()[:])
#File('output/theta-init.pvd') << theta

h = CellSize(mesh)
x = SpatialCoordinate(mesh)

# PSPG-stabilization parameter
delta = h**2*Constant(1./12)/nu

# forcing for the Stokes part
ez = -x/radius
F = -ez*theta_1

# bilinear and linear forms
def a(u,v): return inner(2*nu*sym(grad(u)),grad(v))*dx
def b(v,q): return -div(v)*q*dx
def f(v):   return dot(F, v)*dx

# variational problem (with free-slip boundary conditions)
stokes = a(u,v) + b(v,p) + b(u,q) - f(v) \
       - dot(delta*grad(p), grad(q))*dx + p*r*dx + m*q*dx
stokes_prec = inner(nu*grad(u),grad(v))*dx - p*q/nu*dx - m*r*dx - f(v)

# boundary conditions for Stokes equations
bc_u = []
if args.noslip:
   # no-slip boundary conditions
   bc_u += [ DirichletBC(W.sub(0), Constant((0.0,0.0,0.0)), 'on_boundary') ]
else:
   # free slip by a Nitsche-approach as described in Freund/Stenberg (1995)
   n = FacetNormal(mesh)
   beta  = Constant(10)
   t = lambda u, p: dot(2*nu*sym(grad(u)),n) - p*n
   stokes += beta/h*dot(u,n)*dot(v,n)*ds \
           - dot(n,t(u,p))*dot(v,n)*ds \
           - dot(u,n)*dot(n,t(v,q))*ds
   # preconditioner needs to be modified as well
   stokes_prec += beta/h*dot(u,n)*dot(v,n)*ds

# boundary conditions for energy equation
bc_theta = [ DirichletBC(Q, theta_init, 'on_boundary') ]

t, i = 0.0, 0
U = Function(Vd)

# lumped mass
tmp = Vector(theta.vector())
p, q = TrialFunction(Q), TestFunction(Q)
l_mass = assemble(action(p*q*dx, Constant(1)))

# explicitly choose ML if available: this worked best for our systems,
# much better than hypre_amg. this may be a configuration issue though...
amg = 'amg' if not has_krylov_solver_preconditioner('ml_amg') else 'ml_amg'
solver = KrylovSolver('tfqmr', amg)
solver.parameters['nonzero_initial_guess'] = True
#solver.parameters['relative_tolerance'] = 1e-8
#solver.parameters['monitor_convergence'] = True

# assemble Stokes operator and preconditioner
A = assemble(lhs(stokes))
l = assemble(rhs(stokes))
for bc in bc_u: bc.apply(A, l)

P = assemble(lhs(stokes_prec))
for bc in bc_u: bc.apply(P, l)
solver.set_operators(A, P)

# solve Stokes part
def solve_stokes():

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
ttmp = 0

import time
tstr = time.strftime('%Y%m%d-%H%M%S')
basestr = 'output/sphere-' + tstr
with open(basestr + '.txt', 'w') as text_file:
   text_file.write(str(args) + '\n\n')

while t < args.tend + DOLFIN_EPS:

   # compute CFL
   umax = np.max(np.abs(U.vector().array()))
   #umax = sqrt(np.max(project(dot(U,U), Q).vector().array()))
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

   print 't =', t, 'its({0}/{1}) ='.format(i,args.update_u_every), its

   if args.plot:
      plot(uh)
      #plot(theta, mode='color')

   ttmp += dt
   if args.output_every > 0 and ttmp > args.output_every:
   
      ttmp -= args.output_every
   
      basestr_i = basestr + '_{0}'.format(i)

      File(basestr_i + '-T.pvd') << theta
      File(basestr_i + '-u.pvd') << uh
      File(basestr_i + '-p.pvd') << ph

if args.output:

   File(basestr + '-T.pvd') << theta
   File(basestr + '-u.pvd') << uh
   File(basestr + '-p.pvd') << ph

