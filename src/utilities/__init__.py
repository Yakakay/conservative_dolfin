

def print_convergence_table(h, e, names, tablefmt = 'simple', floatfmt = '.4e'):

   from math import log

   n = len(e[0]); tab = []; headers = []
   for i in xrange(n):
      headers += [names[i], 'rate']
   
   for i in xrange(len(e)):
      tab.append([]);
      ri = h[i-1]/h[i]
      for j in xrange(n):
         tab[-1] += [e[i][j], 0 if i is 0 else log(e[i-1][j]/e[i][j], ri)]

   try:
   
      from tabulate import tabulate
      print tabulate(tab, numalign = 'right', \
         headers = headers, tablefmt = tablefmt, floatfmt = floatfmt) + '\n'

   except ImportError:
   
      print 'warning: module tabulate is not installed! giving plain output instead!\n'
      
      latex = tablefmt == 'latex'
      sep = ' & ' if latex else ' '
      
      if latex: print r'\begin{tabular}{%s}' %(''.join(['r']*len(e[0]*2)))
      line = sep.join(headers)
      if latex:
         print r'\hline'
         print line + r' \\'
         print r'\hline'
      else:
         print line

      for r in tab:
         line = sep.join([('{0:%s}' %(floatfmt)).format(c) for c in r])
         if latex: line += r' \\'
         print line
      if latex: print '\end{tabular}'

def barycentric_refine(mesh):

   from dolfin import Mesh, MeshEditor
   
   # the extension to 3d is straightforward but not yet implemented
   assert mesh.topology().dim() is 2

   # barycentric refinement
   v = mesh.coordinates()
   t = mesh.cells()

   b = (v[t[:,0],:]+v[t[:,1],:]+v[t[:,2],:])/3
   
   mesh = Mesh()
   editor = MeshEditor()
   editor.open(mesh, 2, 2);

   # add vertices to mesh
   nv0 = len(v)
   nv1 = nv0 + len(t)
   editor.init_vertices(nv1)
   for i, vi in enumerate(v):
     editor.add_vertex(i, vi[0], vi[1])
   for i, vi in enumerate(b):
     editor.add_vertex(i + nv0, vi[0], vi[1])

   # add cells to the mesh
   nt1 = 3*len(t)
   editor.init_cells(nt1)
   for i, ti in enumerate(t):
     editor.add_cell(i*3+0, ti[0], ti[1], nv0 + i)
     editor.add_cell(i*3+1, ti[1], ti[2], nv0 + i)
     editor.add_cell(i*3+2, ti[2], ti[0], nv0 + i)

   # done: create and return mesh object
   editor.close()
   return mesh


def refine_boundary_layers(mesh, s, d, x0, x1):

   from dolfin import CellFunction, cells, refine, DOLFIN_EPS

   h = mesh.hmax()
   cell_markers = CellFunction('bool', mesh, mesh.topology().dim())
   cell_markers.set_all(False)

   for cell in cells(mesh):
      x = cell.midpoint()
      for i, d_ in enumerate(d):
         if x[d_] > (x1[i]-s*h-DOLFIN_EPS) or x[d_] < (s*h + x0[i] + DOLFIN_EPS):
            cell_markers[cell] = True
         
   return refine(mesh, cell_markers)


def set_dolfin_optimisation(enable=True):

   from dolfin import parameters
   
   parameters['form_compiler']['optimize'] = enable
   parameters['form_compiler']['cpp_optimize'] = enable

