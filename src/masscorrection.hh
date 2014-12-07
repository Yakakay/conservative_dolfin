//#include <boost/shared_ptr.hpp>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h> // remove...?
//#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/geometry/SimplexQuadrature.h>

namespace dolfin
{
   /*void compute_correction(Mesh const& mesh,
                           GenericFunction const& u,
                           GenericFunction const& div_u)
   {
      using namespace std;
      
      int d = mesh.topology().dim();

      // iterate over all vertices
      for(VertexIterator vi(mesh); !vi.end(); ++vi)
      {
         double sum = 0.0;
      
         Point pi = vi->midpoint();
         
         // iterate over all edges adjacent to this vertex
         for(EdgeIterator e(*vi); !e.end(); ++e)
         {
            Point pe = e->midpoint();
         
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
              
               sum += un;
               
            }
         }
        
         std::cout << sum << std::endl;
      }
   }*/
  
   void compute_sub_dual_integrals(Cell const& c,
                                   GenericFunction const& u,
                                   GenericFunction const& div_u,
                                   FunctionSpace const& Q,
                                   unsigned const degree)
   {
     double measT = c.volume();
     ufc::cell ufc_cell;
     c.get_cell_topology(ufc_cell);
     
     // first, we need to compute \int_T div u_h * phi_i dx for i=1,2,3
     // todo: how to determine order automatically?
     
     // compute quadrature weights and points
     std::vector<double> coords;
     c.get_vertex_coordinates(coords);
          std::pair<std::vector<double>, std::vector<double> > quad =
        SimplexQuadrature::compute_quadrature_rule(coords.data(), 2, 2, degree*2);
     
     FiniteElement const& fe = *Q.element();
     
     double r[3] = { 0.0 }; // residuals
     std::vector<double> phi(fe.space_dimension(), 0.0);
     
     for(size_t iq = 0; iq < quad.first.size(); ++iq)
     {
       double dx = measT*quad.second[iq];
     
       double const* const xq = &quad.second[iq*2];
       fe.evaluate_basis_all(phi.data(), xq, coords.data(), 0);

       double div_u_iq = 0.0;
       div_u.evaluate(&div_u_iq, xq, ufc_cell);
       for(size_t i = 0; i < 3; ++i)
         r[i] -= div_u_iq*phi[i]*dx;
     }
     
     // element vertices: p0, p1, p2
     double const* const p[3] = { &coords[0], &coords[2], &coords[4] };
     
     // element barycenter: b = (p0 + p1 + p2)/3
     double const b[2] = {
        (p[0][0]+p[1][0]+p[2][0])/3,
        (p[0][1]+p[1][1]+p[2][1])/3
     };
     
     // edge midpoints
     double const m[3][2] = {
        { (p[1][0]+p[2][0])/2, (p[1][1]+p[2][1])/2 }, // m0 = (p1 + p2)/2,
        { (p[0][0]+p[2][0])/2, (p[0][1]+p[2][1])/2 }, // m1 = (p0 + p2)/2
        { (p[0][0]+p[1][0])/2, (p[0][1]+p[1][1])/2 }  // m2 = (p0 + p1)/2
     };
     
     // we don't need coords anymore. can be overwritten!
     
     // second, we need to compute \int_{B_i} div u_h dx for i=1,2,3
     for(size_t i = 0; i < 3; ++i)
     {
         // the sub-dual quadrilateral can be decomposed into two triangles
         // (1)  (p_i, m_{i+1}, m_{i+2}) with area |T|/4
         coords[0] = p[i][0];
         coords[1] = p[i][1];
         coords[2] = m[(i+1)%3][0];
         coords[3] = m[(i+1)%3][1];
         coords[4] = m[(i+2)%3][0];
         coords[5] = m[(i+2)%3][1];
       
         quad = SimplexQuadrature::compute_quadrature_rule(coords.data(), 2, 2, degree);
         for(size_t iq = 0; iq < quad.first.size(); ++iq)
         {
           double dx = measT/4*quad.second[iq];
           double const* const xq = &quad.second[iq*2];
           double div_u_iq = 0.0;
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
       
         quad = SimplexQuadrature::compute_quadrature_rule(coords.data(), 2, 2, degree);
         for(size_t iq = 0; iq < quad.first.size(); ++iq)
         {
           double dx = measT/12*quad.second[iq];
           double const* const xq = &quad.second[iq*2];
           double div_u_iq = 0.0;
           div_u.evaluate(&div_u_iq, xq, ufc_cell);
           r[i] += div_u_iq*dx;
         }
     }
     
     std::cout << "res: " << r[0] << " " << r[1] << " " << r[2] << std::endl;
     for(size_t i = 0; i != 3; ++i) {
        std::cout << "kappa = " << (r[i] - r[(i+2)%3])/3 << std::endl;
     }
     
   }
  
   void compute_correction(Mesh const& mesh,
                           GenericFunction const& u,
                           GenericFunction const& div_u,
                           FunctionSpace const& Q,
                           unsigned const degree)
   {
      using namespace std;
      
      int d = mesh.topology().dim();

      // iterate over all elements
      for(CellIterator c(mesh); !c.end(); ++c) {
        
        compute_sub_dual_integrals(*c, u, div_u, Q, degree);
        
        
      }

     
     
      for(VertexIterator vi(mesh); !vi.end(); ++vi)
      {
         double sum = 0.0;
      
         Point pi = vi->midpoint();
         
         // iterate over all edges adjacent to this vertex
         for(EdgeIterator e(*vi); !e.end(); ++e)
         {
            Point pe = e->midpoint();
         
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
              
               sum += un;
               
            }
         }
        
         std::cout << sum << std::endl;
      }
   }


}