#! /usr/bin/env python

import os
from dolfin import compile_extension_module

# compile the extension modules
fv = {}
for d in [2, 3]:
  dir = os.path.dirname(os.path.realpath(__file__))
  header_file = open('{0}/include/fvm_{1}d.hh'.format(dir,d))
  code = '\n'.join(header_file.readlines())
  fv[d] = compile_extension_module(code)

# wrapper around the different implementations
def advance(c, c0, u, dt):

   Q = c.function_space()
   d = Q.mesh().topology().dim()
   return fv[d].advance(c.vector(), c0.vector(), Q, u, dt)
