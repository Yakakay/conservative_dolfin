//#include <boost/shared_ptr.hpp>
#include <map>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h> // remove...?
//#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/SimplexQuadrature.h>

namespace dolfin
{
   std::pair<std::vector<double>, std::vector<double> >
     quadrature_rule_triangle(std::size_t order)
   {
     std::vector<double> w, x;

     switch(order) {
      case 1:
         w.assign(1, 1.);
         x.assign(3, 1./3);
         break;
      case 2:
         w.assign(3, 1./3);
         x.assign(9, 1./6);
         x[0] = x[4] = x[8] = 2./3;
         break;
      default:
         dolfin_error("masscorrection.hh",
                      "quadrature rule for triangle",
                      "Not implemented for order ",order);
      }
      return std::make_pair(w,x);
   }
  
   typedef std::pair<std::size_t, std::size_t> facet_index;

   void compute_sub_dual_integrals(std::map<facet_index, double>& kappa,
                                   Cell const& c,
                                   GenericFunction const& div_u,
                                   unsigned const degree)
   {
     double const measT = c.volume();
     
     ufc::cell ufc_cell;
     c.get_cell_topology(ufc_cell);
     
     // first, we need to compute \int_T div u_h * phi_i dx for i=1,2,3
     // todo: how to determine order automatically?
     
     std::vector<double> coords;
     c.get_vertex_coordinates(coords);

     // quadrature weights and points (in barycentric coords)
     std::pair<std::vector<double>, std::vector<double> > quad =
        quadrature_rule_triangle(2);
     std::vector<double> const &w = quad.first, &x = quad.second;
     
     // element vertices: p0, p1, p2
     double const* const p[3] = { &coords[0], &coords[2], &coords[4] };
     
     double r[3] = { 0.0 } /* residuals */, div_u_iq;
     
     for(std::size_t iq = 0; iq < w.size(); ++iq)
     {
       double dx = measT*w[iq];
     
       double const* const phi = &x[iq*3]; // barycentric coords
       // physical coords
       double xq[2] = { p[0][0]*phi[0] + p[1][0]*phi[1] + p[2][0]*phi[2],
                        p[0][1]*phi[0] + p[1][1]*phi[1] + p[2][1]*phi[2]};
       
       div_u.evaluate(&div_u_iq, xq, ufc_cell);
       for(std::size_t i = 0; i < 3; ++i)
         r[i] -= div_u_iq*phi[i]*dx;
     }
     
     
     // element barycenter: b = (p0 + p1 + p2)/3
     double const b[2] = {
        (p[0][0]+p[1][0]+p[2][0])/3,
        (p[0][1]+p[1][1]+p[2][1])/3
     };
     
     // edge midpoints
     double const m[3][2] = {
        { (p[1][0]+p[2][0])/2, (p[1][1]+p[2][1])/2 }, // m0 = (p1 + p2)/2
        { (p[0][0]+p[2][0])/2, (p[0][1]+p[2][1])/2 }, // m1 = (p0 + p2)/2
        { (p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2 }  // m2 = (p0 + p1)/2
     };
     
     // we don't need coords anymore. can be overwritten!
     
     // second, we need to compute \int_{B_i} div u_h dx for i=1,2,3
     // note: we could use a lower degree quadrature here...
     for(std::size_t i = 0; i < 3; ++i)
     {
         // the sub-dual quadrilateral can be decomposed into two triangles
         // (1)  (p_i, m_{i+1}, m_{i+2}) with area |T|/4
         coords[0] = p[i][0];
         coords[1] = p[i][1];
         coords[2] = m[(i+1)%3][0];
         coords[3] = m[(i+1)%3][1];
         coords[4] = m[(i+2)%3][0];
         coords[5] = m[(i+2)%3][1];
       
         for(std::size_t iq = 0; iq < w.size(); ++iq)
         {
           double dx = measT/4*w[iq];
           double const* const phi = &x[iq*3]; // barycentric coords
           // physical coords
           double xq[2] = { p[0][0]*phi[0] + p[1][0]*phi[1] + p[2][0]*phi[2],
                            p[0][1]*phi[0] + p[1][1]*phi[1] + p[2][1]*phi[2]};
           div_u.evaluate(&div_u_iq, xq, ufc_cell);
           r[i] += div_u_iq*dx;
         }
       
         // (2)  (m_{i+1}, m_0, m_{i+2}) with area |T|/12
         coords[0] = m[(i+1)%3][0];
         coords[1] = m[(i+1)%3][1];
         coords[2] = b[0];
         coords[3] = b[1];
         coords[4] = m[(i+2)%3][0];
         coords[5] = m[(i+2)%3][1];
       
         for(std::size_t iq = 0; iq < w.size(); ++iq)
         {
           double dx = measT/12*w[iq];
           double const* const phi = &x[iq*3]; // barycentric coords
           // physical coords
           double xq[2] = { p[0][0]*phi[0] + p[1][0]*phi[1] + p[2][0]*phi[2],
                            p[0][1]*phi[0] + p[1][1]*phi[1] + p[2][1]*phi[2]};
           div_u.evaluate(&div_u_iq, xq, ufc_cell);
           r[i] += div_u_iq*dx;
         }
     }
     
     // add to map (note: we scale kappa with |f_{ij}|)
     unsigned const* vertex_ids = c.entities(0);
     for(std::size_t i = 0; i != 3; ++i) {
        std::size_t j = (i+2)%3, ii = vertex_ids[i], jj = vertex_ids[j];
        double kappa_ij = (r[i] - r[j])/3;
        kappa[facet_index(ii,jj)] = kappa_ij;
        kappa[facet_index(jj,ii)] = -kappa_ij;
     }
   }
  
   void compute_correction(Mesh const& mesh,
                           GenericFunction const& u,
                           GenericFunction const& div_u,
                           FunctionSpace const& Q,
                           unsigned const degree)
   {
      int d = mesh.topology().dim();
      if(d != 2)
        dolfin_error("masscorrection.hh",
                     "flux-correction computation",
                     "not implemented for dimension ", d);

      std::map<facet_index, double> kappa;

      // iterate over all elements and compute corrections
      for(CellIterator c(mesh); !c.end(); ++c) {
      
        compute_sub_dual_integrals(kappa, *c, div_u, degree);
        
        unsigned const* vertex_ids = c->entities(0);
        std::cout << "element (" << vertex_ids[0] << ", "
                  << vertex_ids[1] << ", " << vertex_ids[2] << ")" << std::endl;
       
        for(std::size_t i = 0; i != 3; ++i) {
           std::size_t ii = vertex_ids[i], jj = vertex_ids[(i+2)%3];
           std::cout << "kappa(" << ii << ", " << jj << ") = " << kappa[facet_index(ii,jj)] << std::endl;
           std::cout << "kappa(" << jj << ", " << ii << ") = " << kappa[facet_index(jj,ii)] << std::endl;
        }
      }

      ufc::cell ufc_cell;
      std::vector<double> coords;

      // iterate over all vertices
      for(VertexIterator vi(mesh); !vi.end(); ++vi)
      {
         double sum = 0.0;
         
         size_t ii = vi->index(); // index of center vertex
         
         // iterate over elements in nodal patch
         for(CellIterator c(*vi); !c.end(); ++c)
         {
            unsigned const* vertex_ids = c->entities(0);
            c->get_cell_topology(ufc_cell);
            
            // find index of center vertex in cell
            std::size_t v0 = 0;
            while(v0 < 3)
               if(vertex_ids[v0] == ii) break; else ++v0;
            
            // rearrange other vertices
            std::size_t const v1 = (v0+1)%3, v2 = (v0+2)%3;
            
            // get coordinates
            c->get_vertex_coordinates(coords);
            double const* const p[3] = { &coords[v0*2], &coords[v1*2], &coords[v2*2] };

            // element barycenter: b = (p0 + p1 + p2)/3
            double const b[2] = {
               (p[0][0]+p[1][0]+p[2][0])/3,
               (p[0][1]+p[1][1]+p[2][1])/3
            };
     
            // edge midpoints for the edges adjacent to v0
            double const m[2][2] = {
               { (p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2 }, // m0 = (p0 + p1)/2
               { (p[0][0]+p[2][0])/2, (p[0][1]+p[2][1])/2 }, // m1 = (p0 + p2)/2
            };
            
            // get area facet normal for facet(v0, v1)
            // An_01 = orth(m0-b) = ((m0-b)_1, -(m0-b)_0)
            double An_01[2] = { b[1] - m[0][1], -(b[0] - m[0][0]) }; // todo: sign?
            // todo: replace by quadrature!!!
            double xq01[2] = { (m[0][0] + b[0])/2, (m[0][1] + b[1])/2 };
            
            double sign = std::copysign(1.0, An_01[0]*(xq01[0] - p[0][0]) + An_01[1]*(xq01[1] - p[0][1]));
            
            //if(sign < 0)
            //   std::cout << "warning: n1 in wrong direction" << std::endl;
            
            double u_eval[2] = { 0.0 };
            u.evaluate(u_eval, xq01, ufc_cell);

            // conservative flux
            double j = sign*(u_eval[0]*An_01[0] + u_eval[1]*An_01[1]);
            j -= kappa[facet_index(ii, vertex_ids[v1])];
            sum += j;

            // get area facet normal for facet(v0, v2)
            // An_02 = orth(b-m1) = ((b-m1)_1, -(b-m1)_0)
            double An_02[2] = { m[1][1] - b[1], -(m[1][0] - b[0]) }; // todo: sign?
            // todo: replace by quadrature!!!
            double xq02[2] = { (m[1][0] + b[0])/2, (m[1][1] + b[1])/2 };
            
            sign = std::copysign(1.0, An_02[0]*(xq02[0] - p[0][0]) + An_02[1]*(xq02[1] - p[0][1]));

            //if(sign < 0)
            //   std::cout << "warning: n2 in wrong direction" << std::endl;

            u.evaluate(u_eval, xq02, ufc_cell);
            
            j = sign*(u_eval[0]*An_02[0] + u_eval[1]*An_02[1]);
            j -= kappa[facet_index(ii, vertex_ids[v2])];
            sum += j;
            
            
         
         }
        
         std::cout << sum << std::endl;
      }
   }


}