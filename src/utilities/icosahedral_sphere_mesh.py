# code that allows to generate an icosahedral sphere mesh consisting
# of tetrahedra. using IcosahedralSphereMesh this can be directly used
# to generate meshes for FEniCS.

# authors: B. Gmeiner, C. Waluga(2013)

def spherical_midpoint(x, y):

  # compute projected midpoint in spherical geometry
  h = [(x[0] + y[0]) / 2.0,(x[1] + y[1]) / 2.0,(x[2] + y[2]) / 2.0 ]
  disth = (h[0]*h[0] + h[1]*h[1] + h[2]*h[2])**0.5
  distx = (x[0]*x[0] + x[1]*x[1] + x[2]*x[2])**0.5
  t = distx / disth
  return( t*h[0], t*h[1], t*h[2])

# recursively refine triangular surface meshes
def refine(a, b, c, level):
  
  if level == 0: return [ [a, b, c] ] # end of recursion

  # compute edge-midpoints as nodes for refined elements
  ab = spherical_midpoint( a, b)
  ac = spherical_midpoint( a, c)
  bc = spherical_midpoint( b, c)

  # return a list of nodes for the refined elements
  ret =      refine(a,  ab, ac, level - 1)
  ret.extend(refine(ab,  b, bc, level - 1))
  ret.extend(refine(ac, bc,  c, level - 1))
  ret.extend(refine(ab, ac, bc, level - 1))
  return ret

def generate_icoshedral_sphere_mesh(level, layers):

  nl = len(layers) - 1
  assert level > 0 and nl > 0
  
  # vertices of the icosahedron(12)
  phi =(1.0 + 5.0**0.5) / 2.0 # golden section
  s = layers[-1] * (phi**2 + 1)**(-0.5)
  phi *= s
  vico = [( 0, s,  phi),( s,  phi, 0),(  phi, 0, s),( 0, -s,  phi),( -s,  phi, 0),( phi, 0, -s),
          ( 0, s, -phi),( s, -phi, 0),( -phi, 0, s),( 0, -s, -phi),( -s, -phi, 0),(-phi, 0, -s) ]

  # faces of the icosahedron(20)
  tico = [ [  2, 0, 1], [  0,  1, 4 ], [ 3,  2,  0 ], [  3,  0, 8], [  0, 4,  8 ],
           [  2, 1, 5], [  1,  4, 6 ], [ 1,  6,  5 ], [  7,  3, 2], [  7, 2,  5 ],
           [ 10, 7, 3], [ 10,  3, 8 ], [ 4,  6, 11 ], [  4, 11, 8], [  6, 9,  5 ],
           [  9, 7, 5], [  6, 11, 9 ], [ 9, 10,  7 ], [ 11, 10, 8], [ 11, 9, 10 ] ]

  # refine the faces of the icosahedron
  elements = []
  for t in tico:
    elements.extend(refine(vico[t[0]], vico[t[1]], vico[t[2]], level))

  vertices = { } ; triangles = [ ] ; cells = [ ]

  # generation of a triangluar surface(first layer)
  id_new = 0
  for e in elements:
    ids = []
    for v in e: # find or create
      if vertices.has_key(v): id = vertices[v]
      else: id = vertices[v] = id_new; id_new += 1
      ids.append(id)
    triangles.append(ids)

  # generate additional vertex layers in radial direction
  nv = len(vertices)
  verts = vertices.items()

  nl = len(layers)-1
  for i in range(0, nl):
    for v in verts:
      s = layers[i] / layers[-1]
      coords =(s*v[0][0], s*v[0][1], s*v[0][2])
      vertices[coords] = v[1] +(nl - i)*nv # set id

  # connectivity of tetrahedra
  for t in triangles:
    for i in range(0, nl):
      cells.append( [ t[0]+i*nv, t[1]+i*nv,     t[2]+i*nv,     t[2]+(i+1)*nv ])
      cells.append( [ t[0]+i*nv, t[1]+i*nv,     t[1]+(i+1)*nv, t[2]+(i+1)*nv ])
      cells.append( [ t[0]+i*nv, t[0]+(i+1)*nv, t[1]+(i+1)*nv, t[2]+(i+1)*nv ])

  return( vertices, cells)


def IcosahedralSphereMesh(level, layers):
 
  from dolfin import Mesh, MeshEditor, File

  # generate vertices and cells
  (vertices, cells) = generate_icoshedral_sphere_mesh( level, layers)
  
  # init mesh and mesh editor helper
  mesh = Mesh()
  editor = MeshEditor()
  editor.open(mesh, 3, 3);
  
  # add vertices to mesh
  nVert = len(vertices)
  editor.init_vertices(nVert)

  verts = vertices.items()
  #verts = sorted(verts, key=lambda key: key[1])

  for v in verts:
    editor.add_vertex(v[1], v[0][0], v[0][1], v[0][2])

  ncells = len(cells)
  editor.init_cells(ncells)

  id = 0
  for c in cells:
    editor.add_cell(id, c[0], c[1], c[2], c[3])
    id += 1

  # done: create and return mesh object
  editor.close()
  return mesh

