#include <boost/shared_ptr.hpp>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/fem/GenericDofMap.h>

namespace dolfin
{
   void advance(GenericVector& s,         // new coefficients
                GenericVector const& s0,  // old coefficients
                FunctionSpace const& S,   // function space
                GenericFunction const& u, // velocity
                double dt)                // time-step size
   {
      using namespace std;
      
      Mesh const& mesh = *S.mesh();
      vector<la_index> v2d = vertex_to_dof_map(S);
      
      int d = mesh.topology().dim();

      // iterate over all vertices
      for(VertexIterator vi(mesh); !vi.end(); ++vi)
      {
         Point pi = vi->midpoint();
         
         // determine volume of dual cell
         double vol = 0.0;
         for(CellIterator c(*vi); !c.end(); ++c)
            vol += c->volume()/(d+1);
      
         size_t ii = vi->index();
         double si = s0[v2d[ii]], snew = si;
      
         // iterate over all edges adjacent to this vertex
         for(EdgeIterator e(*vi); !e.end(); ++e)
         {
            Point pe = e->midpoint();
            
            // find second vertex
            unsigned const* ee = e->entities(0);
            size_t ij = (ee[0] == ii) ? ee[1] : ee[0];
            double sj = s0[v2d[ij]];
         
            // iterate over all cells adjacent to this edge
            for(CellIterator c(*e); !c.end(); ++c)
            {
               Point pc = c->midpoint();
               
               // evaluate u at midpoint of dual facet
               Point pce = 0.5*(pc + pe), u_pce;
               ufc::cell ufc_cell;
               c->get_cell_topology(ufc_cell);
               u.evaluate(u_pce.coordinates(), pce.coordinates(), ufc_cell);
               
               // determine normal vector times facet area
               Point t = (pc - pe), n(-t[1], t[0]);
               if(n.dot(pce - vi->midpoint()) < 0) n = -n;
               
               double un = u_pce.dot(n);
               
               // update cell value
               snew -= dt/vol*(un*(si + sj) + abs(un)*(si - sj))/2.0;
            }
         }
         
         // update the value
         s.setitem(v2d[ii], snew);
      }
   }

}