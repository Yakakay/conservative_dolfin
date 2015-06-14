#include <boost/shared_ptr.hpp>
#include <dolfin/fem/fem_utils.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/fem/GenericDofMap.h>
#include <ufc_geometry.h>

namespace dolfin
{
   void advance(GenericVector& s,         // new coefficients
                GenericVector const& s0,  // old coefficients
                FunctionSpace const& S,   // function space
                GenericFunction const& u, // velocity
                double dt)                // time-step size
   {
      // todo: this can be made more efficient if u is not projected beforehand,
      //       but if u and p are both passed and the fluxes are corrected here.
   
      using namespace std;
      
      Mesh const& mesh = *S.mesh();
      vector<la_index> v2d = vertex_to_dof_map(S);
      
      int d = mesh.topology().dim();
      std::vector<double> const& coord = mesh.coordinates();
      
      double J[9], K[9], det, coords[12], u_ij[3], x_ij[3], An[3];

      // iterate over all vertices
      for(VertexIterator vi(mesh); !vi.end(); ++vi)
      {
         size_t ii = vi->index();
         double si = s0[v2d[ii]], dsi = 0.0, vol = 0.0;
      
         // iterate over all cells adjacent to this vertex
         for(CellIterator c(*vi); !c.end(); ++c)
         {
             unsigned const* ce = c->entities(0);
         
             // find index of vertex in cell
             unsigned v0 = 0;
             while(v0 != d + 1)
               if(ce[v0] == ii) break; else ++v0;
            
            // reordering, such that vertex 0 is current one
            int const tet_0 = v0, tet_1 = (v0+1)%4, tet_2 = (v0+2)%4, tet_3 = (v0+3)%4;

            // get coordinates
            coords[ 0] = coord[ce[tet_0]*d];
            coords[ 1] = coord[ce[tet_0]*d+1];
            coords[ 2] = coord[ce[tet_0]*d+2];
            coords[ 3] = coord[ce[tet_1]*d];
            coords[ 4] = coord[ce[tet_1]*d+1];
            coords[ 5] = coord[ce[tet_1]*d+2];
            coords[ 6] = coord[ce[tet_2]*d];
            coords[ 7] = coord[ce[tet_2]*d+1];
            coords[ 8] = coord[ce[tet_2]*d+2];
            coords[ 9] = coord[ce[tet_3]*d];
            coords[10] = coord[ce[tet_3]*d+1];
            coords[11] = coord[ce[tet_3]*d+2];
            
            compute_jacobian_tetrahedron_3d(J, coords);
            compute_jacobian_inverse_tetrahedron_3d(K, det, J);

            ufc::cell ufc_cell;
            c->get_cell_topology(ufc_cell);

            // assemble the equations for the convective operator
            double sign = copysign(1./24, det);
            vol += sign*det;
            
            // precompute some weights
            double const w1 = 17.0/48.0, w2 = 7.0/48.0;

            // assemble fluxes for facet associated with edge(tet_0,tet_1)
            x_ij[0] = w1*coords[0] + w1*coords[3] + w2*coords[6] + w2*coords[ 9];
            x_ij[1] = w1*coords[1] + w1*coords[4] + w2*coords[7] + w2*coords[10];
            x_ij[2] = w1*coords[2] + w1*coords[5] + w2*coords[8] + w2*coords[11];
            u.evaluate(u_ij, x_ij, ufc_cell);
            An[0] = J[3]*J[7] - J[3]*J[8] - J[4]*J[6] + 2*J[4]*J[8] + J[5]*J[6] - 2*J[5]*J[7];
            An[1] = J[0]*J[8] - J[0]*J[7] + J[1]*J[6] - 2*J[1]*J[8] - J[2]*J[6] + 2*J[2]*J[7];
            An[2] = J[0]*J[4] - J[0]*J[5] - J[1]*J[3] + 2*J[1]*J[5] + J[2]*J[3] - 2*J[2]*J[4];
            double area_un = sign*(An[0]*u_ij[0] + An[1]*u_ij[1] + An[2]*u_ij[2]);
            double abs_area_un = std::abs(area_un);
            dsi += 0.5*(area_un+abs_area_un) * s0[v2d[ce[tet_0]]];
            dsi += 0.5*(area_un-abs_area_un) * s0[v2d[ce[tet_1]]];;

            // assemble fluxes for facet associated with edge(tet_0,tet_2)
            x_ij[0] = w1*coords[0] + w2*coords[3] + w1*coords[6] + w2*coords[ 9];
            x_ij[1] = w1*coords[1] + w2*coords[4] + w1*coords[7] + w2*coords[10];
            x_ij[2] = w1*coords[2] + w2*coords[5] + w1*coords[8] + w2*coords[11];
            u.evaluate(u_ij, x_ij, ufc_cell);
            An[0] = J[3]*J[7] - 2*J[3]*J[8] - J[4]*J[6] + J[4]*J[8] + 2*J[5]*J[6] - J[5]*J[7];
            An[1] = J[1]*J[6] + 2*J[0]*J[8] - J[0]*J[7] - J[1]*J[8] - 2*J[2]*J[6] + J[2]*J[7];
            An[2] = J[0]*J[4] - 2*J[0]*J[5] - J[1]*J[3] + J[1]*J[5] + 2*J[2]*J[3] - J[2]*J[4];
            area_un = sign*(An[0]*u_ij[0] + An[1]*u_ij[1] + An[2]*u_ij[2]);
            abs_area_un = std::abs(area_un);
            dsi += 0.5*(area_un+abs_area_un) * s0[v2d[ce[tet_0]]];
            dsi += 0.5*(area_un-abs_area_un) * s0[v2d[ce[tet_2]]];;

            // assemble fluxes for facet associated with edge(tet_0,tet_3)
            x_ij[0] = w1*coords[0] + w2*coords[3] + w2*coords[6] + w1*coords[ 9];
            x_ij[1] = w1*coords[1] + w2*coords[4] + w2*coords[7] + w1*coords[10];
            x_ij[2] = w1*coords[2] + w2*coords[5] + w2*coords[8] + w1*coords[11];
            u.evaluate(u_ij, x_ij, ufc_cell);
            An[0] = 2*J[3]*J[7] - J[3]*J[8] - 2*J[4]*J[6] + J[4]*J[8] + J[5]*J[6] - J[5]*J[7];
            An[1] = 2*J[1]*J[6] + J[0]*J[8] - 2*J[0]*J[7] - J[1]*J[8] - J[2]*J[6] + J[2]*J[7];
            An[2] = 2*J[0]*J[4] - J[0]*J[5] - 2*J[1]*J[3] + J[1]*J[5] + J[2]*J[3] - J[2]*J[4];
            area_un = sign*(An[0]*u_ij[0] + An[1]*u_ij[1] + An[2]*u_ij[2]);
            abs_area_un = std::abs(area_un);
            dsi += 0.5*(area_un+abs_area_un) * s0[v2d[ce[tet_0]]];
            dsi += 0.5*(area_un-abs_area_un) * s0[v2d[ce[tet_3]]];;
         }
         
         // update the value
         s.setitem(v2d[ii], si - dt/vol*dsi);
      }
   }

}