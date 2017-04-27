// Copyright (C) 2011-2013 Vincent Heuveline
//
// HiFlow3 is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//  
// HiFlow3 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with HiFlow3.  If not, see <http://www.gnu.org/licenses/>.

// Copyright (C) 2014-2016 Nicolai Schoch
//
// The HiFlow3-based Elasticity Simulation is free software and may be used, 
// redistributed and/or modified under the terms of the HiFlow3 conditions.
//
// If you use this implementation and/or results obtained from using it, please 
// cite the following references either available on the HiFlow3 Website http://www.hiflow3.org/
// - Vincent Heuveline et al: "HiFlow3: A Hardware-Aware Parallel Finite Element Package", 
//   in Tools for High Performance Computing 2011, page 139-151, Publisher: Springer 
//   Berlin Heidelberg, 2012. 
//   http://dx.doi.org/10.1007/978-3-642-31476-6_12
// - Nicolai Schoch, Fabian Kissler: "Elasticity Tutorial for Soft Tissue Simulation",
//   in HiFlow3 Tutorials, available online, 2014. 
// or available on Springer Online
// - Nicolai Schoch, Sandy Engelhardt, Raffaele de Simone, Ivo Wolf, Vincent Heuveline:
//   "High Performance Computing for Cognition-Guided Cardiac Surgery: Soft Tissue Simulation 
//   for Mitral Valve Reconstruction in Knowledge-based Surgery Assistance",
//   in "Modeling, Simulation and Optimization of Complex Processes", the Proceedings of 
//   the 6th International Conference on High-Performance Scientific Computing 2015, Publisher: 
//   Springer Berlin Heidelberg, 2015.

/// \author Nicolai Schoch, Fabian Kissler

/// main author: Nicolai Schoch (general model, implementation, main loop (stationary and instationary), setup, assembly, bcs, linear formulation)
/// additions by: Fabian Kissler (corotational formulation, external rotation class)
/// additions to be made by: Suranita Kanjilal (optimization-based contact formulation -> t.b.d. at 2017-05 ff)

/// Additional information on the implementation can be found either in the source code itself, or in
/// - Nicolai Schoch, PhD Thesis, 2017-01-23.
/// - Georgii and Westermann: "Corotated Finite Elements Made Fast and Stable", VRIPHYS, 2008.

/// NOTE: This code does not contain the entire set of functionalities for Mitral Valve Reconstruction Simulation.
///       Parts of the complete algorithm are not (yet) open-source, but available at Nicolai Schoch.

#ifndef ELASTICITY_H
#define ELASTICITY_H

#include <mpi.h>

#include <exception>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <math.h> // needed only for sin()-function to represent the heart beat / pressure profile.

#include "hiflow.h"

#include "rotation.h"

using namespace hiflow;
using namespace hiflow::doffem;
using namespace hiflow::la;
using namespace hiflow::mesh;

#define DIMENSION 3
// #define WITH_ILUPP
#define WITH_METIS

// Linear Algebra type renaming.
typedef LADescriptorCoupledD LAD;
typedef LAD::DataType Scalar;
typedef LAD::VectorType CVector;
typedef LAD::MatrixType CMatrix;

typedef std::vector<double> Coord;

// Exception types
struct UnexpectedParameterValue : public std::runtime_error {
    UnexpectedParameterValue(const std::string& name,
                             const std::string& value)
        : std::runtime_error(
            "Unexpected value '" + value + "' for parameter " + name) {}
};

/// The default quadrature selection chooses a quadrature rule that is accurate to 2 * max(fe_degree).
struct QuadratureSelection {
    QuadratureSelection(int order) : order_(order) {}

    void operator()(const Element<double>& elem, Quadrature<double>& quadrature) {
        const FEType<double>::FiniteElement fe_id = elem.get_fe_type(0)->get_my_id();

        switch (fe_id) {
        case FEType<double>::LAGRANGE_TRI:
            quadrature.set_cell_type(2);
            quadrature.set_quadrature_by_order("GaussTriangle", order_);
            break;
        case FEType<double>::LAGRANGE_QUAD:
            quadrature.set_cell_type(3);
            quadrature.set_quadrature_by_order("GaussQuadrilateral", order_);
            break;
        case FEType<double>::LAGRANGE_TET:
            quadrature.set_cell_type(4);
            quadrature.set_quadrature_by_order("GaussTetrahedron", order_);
            break;
        case FEType<double>::LAGRANGE_HEX:
            quadrature.set_cell_type(5);
            quadrature.set_quadrature_by_order("GaussHexahedron", order_);
            break;
        default:
            assert(false);
        };
    }

    int order_;
};

struct StationaryElasticity_DirichletBC_3D {
    // Parameters:
    // bdy - material number of boundary
    // NOTE: this struct might be deprecated when using pointwise boundary conditions.

	StationaryElasticity_DirichletBC_3D(int var, int bdy1, int bdy2, int bdy3)
        : var_(var), bdy1_(bdy1), bdy2_(bdy2), bdy3_(bdy3) {
        assert(var_ == 0 || var_ == 1 || var_ == 2);
        assert(DIMENSION == 3);
    }

    std::vector<double> evaluate(const Entity& face, const std::vector<Coord>& coords_on_face) const {
	//Return array with Dirichlet values for dof:s on boundary face.
        std::vector<double> values;

        const int material_num = face.get_material_number();

        if (material_num == bdy1_) { // the fixed Dirichlet BC part.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
            }
        } else if (material_num == bdy2_) { // a displacement Dirichlet BC, and holds ONLY for the time dt*timestep <= 1.0.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
                if (var_ == 0) {
                	values[i] = 0.;
                } else if (var_ == 1) {
                	values[i] = 0.;
                } else if (var_ == 2) {
                	values[i] = 0.15;
                }
            }

        } else if (material_num == bdy3_) { // a displacement Dirichlet BC, and holds ONLY for the time dt*timestep <= 1.0.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
                if (var_ == 0) {
                	values[i] = 0.;
                } else if (var_ == 1) {
                	values[i] = 0.;
                } else if (var_ == 2) {
                	values[i] = 0.15;
                }
            }

        }

        return values;
    }

    const int var_;
    const int bdy1_, bdy2_, bdy3_;
};

struct InstationaryElasticity_DirichletBC_3D {
    // Parameters:
    // var - vector component, in [0,1,2].
    // bdy - material number of boundary
    // 
    // Note: 
    // the InstationaryElasticity_DirichletBC_3D-struct assumes that the given DirichletBC-values 
    // are reached linearly continuously within the first second of the simulation,
    // i.e.: ts_ * delta_t_ * dirichlet_values_
    // (except for fixed boundaries with u_D = 0.0 = const.).

	InstationaryElasticity_DirichletBC_3D(int var, int bdy1, int bdy2, int bdy3, int ts, double delta_t)
        : var_(var), bdy1_(bdy1), bdy2_(bdy2), bdy3_(bdy3), ts_(ts), delta_t_(delta_t) {
        assert(var_ == 0 || var_ == 1 || var_ == 2);
        assert(DIMENSION == 3);
    }

    std::vector<double> evaluate(const Entity& face, const std::vector<Coord>& coords_on_face) const {
        //evaluate() loops over all boundary facets.
        
	//Return array with Dirichlet values for dof:s on boundary face.
        std::vector<double> values;

        const int material_num = face.get_material_number();

        if (material_num == bdy1_) { // the fixed Dirichlet BC part, and holds during all of the simulation.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
            }
        } else if (material_num == bdy2_) { // a displacement Dirichlet BC, and holds ONLY for the time dt*timestep <= 1.0.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
                if (var_ == 0) {
                	values[i] = (ts_ * delta_t_) * 0.;
                } else if (var_ == 1) {
                	values[i] = (ts_ * delta_t_) * 0.; // for Bunny: 0.02 on ears
                } else if (var_ == 2) {
                	values[i] = (ts_ * delta_t_) * 0.05; // for MV: 0.15 along slit
                }
            }

        } else if (material_num == bdy3_) { // a displacement Dirichlet BC, and holds ONLY for the time dt*timestep <= 1.0.
            values.resize(coords_on_face.size());

            // loop over dof points on the face
            for (int i = 0; i < coords_on_face.size(); ++i) {
                // evaluate dirichlet function at each point
                const Coord& pt = coords_on_face[i];
                values[i] = 0.;
                if (var_ == 0) {
                	values[i] = (ts_ * delta_t_) * 0.;
                } else if (var_ == 1) {
                	values[i] = (ts_ * delta_t_) * 0.; // for Bunny: 0.02 on ears
                } else if (var_ == 2) {
                	values[i] = (ts_ * delta_t_) * 0.05; // for MV: 0.15 along slit
                }
            }
        }
        
        // return the computed Dirichlet values (vector "values") for the dofs on the boundary facet.
        return values;
    }

    const int var_;
    const double delta_t_;
    const int ts_;
    const int bdy1_, bdy2_, bdy3_;
};

struct StationaryElasticity_RadialNeumannBC_3D { // MAYBE_DEPRECATED. // Radial dependency can be switched on/off below. Currently ON.
	// ------------------------------------------------------------
	// Parameters: selfexplaining.
	// Explanations: see InstationaryElasticity_RadialNeumannBC_3D.
	// ------------------------------------------------------------
        // NOTE: this struct is not used anymore, but may be used to implement radial dependencies.

	StationaryElasticity_RadialNeumannBC_3D(int var, /*int bdy1,*/ double force_at_surface_facet)
	: var_(var), /*bdy1_(bdy1),*/ force_at_surface_facet_(force_at_surface_facet) {
		assert(var_ == 0 || var_ == 1 || var_ == 2);
		assert(DIMENSION == 3);
	}
	
	// get material number of current facet -> better get it directly from AssemblyAssistant-derived-NBC-class!
	// evaluate(const Entity& face, ...)
	// const int material_num = face.get_material_number();
	
	double operator()(const Vec<DIMENSION, double>& phys_coord) const {
	        
		double value = 0;
		
		// Midpoint of circle/membrane/MV
		const double mid[3] = {0., 0., 0.01};
		const double radius_MVobject = 1.0;
		// Vector from Midpoint to phys_coord
		const double mid2phys[3] = {phys_coord[0]-mid[0], phys_coord[1]-mid[1], phys_coord[2]-mid[2]};
		// Vector length in x-y-plane (from membrane midLINE(along z-direction) to phys_coord-point)
		const double cyl_dist = sqrt( mid2phys[0]*mid2phys[0] + mid2phys[1]*mid2phys[1]);
		// Inverse point-distance-ratio from phys_coord-point to midpoint w.r.t. midpoint:
		// Midpoint = 1.0, Points on Rim = 0.0, Other Points > 0.0 && < 1.0;
		const double inv_distance_ratio = (radius_MVobject - cyl_dist)/radius_MVobject;
		// Compute orthogonal force/pressure in z-direction.
		if (var_ == 0) {
		  value = 0.0;
		} else if (var_ == 1) {
		  value = 0.0;
		} else if (var_ == 2) {
		  // switch on/off radial dependency:
		  // ON:
		  value = force_at_surface_facet_ * inv_distance_ratio; // simplification similar to "Poisseuille Profile".
		  // OFF:
		  //value = force_at_surface_facet_; // const force/pressure profile on corresponding boundary.
		}
		
		return value;
	}
	
	const int var_;
	/*const int bdy1_*/;
	const double force_at_surface_facet_;
};

struct InstationaryElasticity_RadialNeumannBC_3D { // MAYBE_DEPRECATED. // Radial dependency can be switched on/off below. Currently ON.
	// ---------------------------------------------------------------------------------------------------------------------------
	// Parameters:
	// self-explaining.
	// ---------------------------------------------------------------------------------------------------------------------------
	// Explanations:
	// the InstationaryElasticity_RadialNeumannBC_3D assumes that the force/pressure (in z-direction only) on the surface is radially dependent,
	// i.e. strongest in the center of the valve, and weekest near the boundary/annulus.
	// The radial dependency can be switched on/off below.
	// 
	// Assuming two simplifications: // TASK: check and adapt modelling assumptions. // TODO.
	// - time-independent pressure cross-sectional profile is assumed to 
	//   either be similar to "Poisseuille Profile" (radial function)
	//   or just be constant all over the body's boundary part / surface;
	// - pressure-time-profile similar to "Wigger's Diagram"
	//   is assumed to be represented by some sinus function or a Taylor Series approximation of a sinus function
	//   which looks like: p(t) [in Pascal] = 60mmHg + 60mmHg * sin(t * pi) [in Pascal];
	//   --> Note: maybe take away the initial 60mmHg in order to represent inflow AND outflow pressure balancing out each other.
	// ---------------------------------------------------------------------------------------------------------------------------
	
	InstationaryElasticity_RadialNeumannBC_3D(int var, /*int bdy1,*/ double force_at_surface_facet, int ts, double delta_t)
	: var_(var), /*bdy1_(bdy1),*/ force_at_surface_facet_(force_at_surface_facet), ts_(ts), delta_t_(delta_t) {
		assert(var_ == 0 || var_ == 1 || var_ == 2);
		assert(DIMENSION == 3);
	}
	
	double operator()(const Vec<DIMENSION, double>& phys_coord) const {
		double value = 0;
		
		// Midpoint of circle/membrane/MV
		const double mid[3] = {0., 0., 0.01};
		const double radius_MVobject = 1.0;
		// Vector from Midpoint to phys_coord
		const double mid2phys[3] = {phys_coord[0]-mid[0], phys_coord[1]-mid[1], phys_coord[2]-mid[2]};
		// Vector length in x-y-plane (from membrane midLINE(along z-direction) to phys_coord-point)
		const double cyl_dist = sqrt( mid2phys[0]*mid2phys[0] + mid2phys[1]*mid2phys[1]);
		// Inverse point-distance-ratio from phys_coord-point to midpoint w.r.t. midpoint:
		// Midpoint = 1.0, Points on Rim = 0.0, Other Points > 0.0 && < 1.0;
		const double inv_distance_ratio = (radius_MVobject - cyl_dist)/radius_MVobject;
		
		double real_time_ = ts_ * delta_t_;
		
		double real_time_when_force_starts_to_act = 0.0; // TODO: make this flexible with xml-input-file.
		double real_time_when_force_ends_to_act = 1.0; // TODO: make this flexible with xml-input-file.
		
// 		double real_time_when_counterforce_starts_to_act = 1.5; // TODO: make this flexible with xml-input-file.
// 		double real_time_when_counterforce_ends_to_act = 2.0; // TODO: make this flexible with xml-input-file.
		
		// Compute orthogonal time-dependent force/pressure in z-direction.
		  // assuming two modelling simplifications:
		  // - time-independent pressure cross-sectional profile is assumed to 
		  //   either be similar to "Poisseuille Profile" (radial function)
		  //   or just be constant all over the body's boundary part / surface;
		  // - pressure-time-profile similar to "Wigger's Diagram"
		  //   is assumed to be represented by some sinus function or a Taylor Series approximation of a sinus function
		  //   which looks like: p(t) [in Pascal] = 60mmHg + 60mmHg * sin(t * pi) [in Pascal];
		if (var_ == 0)
		  value = 0.0;
		else if (var_ == 1)
		  value = 0.0;
		else if (var_ == 2) {
		  // in case of radially-dependent profile:
		  if (real_time_ >= real_time_when_force_starts_to_act && real_time_ <= real_time_when_force_ends_to_act) {
		    value = sin((real_time_ - real_time_when_force_starts_to_act) * 3.14) * (force_at_surface_facet_ * inv_distance_ratio);
		    //value = /*60.0 * 133,322 +*/ sin(real_time_ * 3.14) * (force_at_surface_facet_ * inv_distance_ratio);
		  }/* else if (real_time_ >= real_time_when_counterforce_starts_to_act && real_time_ <= real_time_when_counterforce_ends_to_act) {
		       value = sin((real_time_ - real_time_when_force_starts_to_act) * 3.14) * (force_at_surface_facet_ * inv_distance_ratio);
		  }*/ else {
		    //std::cout << "No NeumannBCs acting anymore. - Hence, assigning Zero-Pressure-NeumannBCs." << std::endl;
		    value = 0.0; //force_at_surface_facet_ * inv_distance_ratio;
		  }
		  // in case of constant, i.e. non-radially-dependent profile:
// 		  value = force_at_surface_facet_; // const force/pressure profile on corresponding boundary facet
		}
		
		return value;
	}
	
	const int var_;
	const double delta_t_;
	const int ts_;
	double real_time_;
	/*const int bdy1_;*/
	const double force_at_surface_facet_;
};

struct InstationaryElasticity_sinusoidalNeumannBC_3D { // sinusoidal Neumann BC (acting orthogonally on facet).
        // this struct computes the time-dependent (sinusoidal) force value (in orthogonal sense, but yet without any direction).
	// Parameters:
	// bdy - boundary identifier(s)
	// force_at_surface_facet - force(s) (i.e. pressure) value at surface facet(s)
	// ts, delta_t - timestep and delta_t for Newmark time stepping scheme.
	
	InstationaryElasticity_sinusoidalNeumannBC_3D(int bdy1, int bdy2, double force1_at_surface_facet, 
						      double force2_at_surface_facet, int ts, double delta_t)
	: bdy1_(bdy1), bdy2_(bdy2), force1_at_surface_facet_(force1_at_surface_facet), 
	  force2_at_surface_facet_(force2_at_surface_facet), ts_(ts), delta_t_(delta_t) {
		assert(DIMENSION == 3);
	}
	
	double operator()(const int bdyNumber_from_assembler) const {
	  
	  double value = 0.0;
	  double force_from_assembler;
	  
	  if ( bdyNumber_from_assembler == bdy1_ ) { // bdy1_ represents the lower(!) body boundary.
	    force_from_assembler = force1_at_surface_facet_;
	  } else if ( bdyNumber_from_assembler == bdy2_ ) { // bdy2_ represents the upper(!) body boundary.
	    force_from_assembler = force2_at_surface_facet_;
	  } else {
	    std::cout << "ControlOutput: Problem processing NeumannBCs w.r.t. bdy_ numbers. " << std::endl;
	    here();
	    assert(0);
	  }
	  
	  // simplifying model assumption:
	  // - pressure-time-profile similar to "Wigger's Diagram"
	  //   is assumed to be represented by some sinus function or a Taylor Series approximation of a sinus function
	  //   which looks like: p(t) [in Pascal] = 60mmHg + 40mmHg * sin(2*pi*t) [in Pascal];
	  // Additionally, we might assume a spacial distribution of the pressure profile:
	  // - time-independent cross-sectional pressure profile similar to "Poisseuille Profile" (radial/etc function).
	  
	  double real_time_ = ts_ * delta_t_;
	  double real_time_when_force_starts_to_act = 0.0; // TODO: make this flexible with xml-input-file.
	  double real_time_when_force_ends_to_act = 10.0; // TODO: make this flexible with xml-input-file.
	  
	  // Compute time-dependent (sinusial) force value (here without any direction):
	  if (real_time_ >= real_time_when_force_starts_to_act && real_time_ <= real_time_when_force_ends_to_act) {
	    
	    // Approximate Wiggers Diagram representation of time-dependent periodic pressure profile (for systole/diastole):
	    
	    // Option 1: approximate pressure profile by means of a sinus-function:
// 	    value = force_from_assembler * sin( 2.0 * 3.14 * (real_time_ - real_time_when_force_starts_to_act) );
	    // depending on the modelling assumption, apply the following computation:
	    //value = 60.0 * 133,322 * 0.01 + force_from_assembler * sin( 2.0 * 3.14 * (real_time_ - real_time_when_force_starts_to_act) );
	    
	    // Option 2: approximate pressure profile by means of a polynomial-function:
	    while ( real_time_ >= 1.0 ) { real_time_ = real_time_ - 1.0; }
	    if ( bdyNumber_from_assembler == bdy1_ ) { // for pressure from below
	      value = force_from_assembler * -1920.0 * (real_time_ - 0.25)*(real_time_ - 0.25) + 120.0;
	    } else if ( bdyNumber_from_assembler == bdy2_ ) { // for pressure from above
	      value = force_from_assembler * -1920.0 * (real_time_ - 0.75)*(real_time_ - 0.75) + 120.0;
	    }
	    if ( value < 0.0 ) { value = 0.0; }
	    
	    // Option 3: Interpolation of arbitrary physiological curve by means of trigonometric Interpolation:
	    // see AddInfo pdf at 2014-10-08:
	    // 1. scan t/p(t)-curve.
	    // 2. make csv-file with t/p(t)-values by means of engauge-digitizer software.
	    // 3. implement trigonometric interpolation function,
	    //    according to e.g. http://de.wikipedia.org/wiki/Trigonometrische_Interpolation
	  }
	  else {
	    value = 0.0;
	    //std::cout << "No NeumannBCs acting anymore. - Hence, assigning Zero-Pressure-NeumannBCs." << std::endl;
	  }
	  
	  return (-1.0) * value;
	}
	
	//const int var_;
	const int bdy1_, bdy2_;
	const double delta_t_;
	const int ts_;
	double real_time_;
	const double force1_at_surface_facet_, force2_at_surface_facet_;
};
//////////////// End boundary conditions /////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// --------------------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


struct InstationaryElasticity_contactBC_3D { // DEPRECATED. NEW VERSION BELOW.
  // this struct accounts for the penalty counter force which "punishes" in the weak formulation of the boundary value problem 
  // any breaking of the contact conditions (i.e. there must always be a gap >= 0.0 between two bodies.
  // the struct computes the force_scaling_factor, depending on
  // - the current timestep's blood pressure (nBC), and
  // - the current size of the gap between the bodies.
  //
  // Parameters: t.b.d.
  
  InstationaryElasticity_contactBC_3D(int cbdy1, int cbdy2, int ts, double delta_t)
	: cbdy1_(cbdy1), cbdy2_(cbdy2), ts_(ts), delta_t_(delta_t) {
		assert(DIMENSION == 3);
	}
  
  double operator()(const int current_bdyNumber_from_assembler/*, const int opposing_bdyNumber_from_assembler*/) const {
    
    double real_time_ = ts_ * delta_t_;
    double value = 1.0;
    
    // compute scaling_force_factor for penalty counter force:
    // - similar to the above "InstationaryElasticity_sinusoidalNeumannBC_3D" (Option 2):
    //   approximate (time-dependent) pressure profile by means of a polynomial-function, but with a negative sign.
    // - moreover, with a distance-dependent scaling-/penalty-factor.
    
    if ( current_bdyNumber_from_assembler == cbdy1_ ) {
      // algorithms not yet published // TODO.
    } else if ( current_bdyNumber_from_assembler == cbdy2_ ) {
      // algorithms not yet published // TODO.
    } else {
      std::cout << "ERROR: Problem processing ContactBCs w.r.t. bdy_ numbers. " << std::endl;
      assert(0);
    }
    
    return value; // this is the scaling_force_factor;
  }
  
  // Parameters:
  const int cbdy1_, cbdy2_;
  const int ts_;
  double delta_t_;
  double real_time_;
};

// struct InstationaryElasticity_contactBC_3D { // NEW. // ContactDistance-dependent PenaltyContactBC-struct.
//   // This struct computes the scaling_penalty_force_factor for the contact boundary assembler, and takes into account:
//   // - the current_bdyNumber_from_assembler -> so far used for double-checks only; 
//   //                                           may be used for anterior/posterior-leaflet-specific penalty contact forces.
//   // - the curr_ContactDistance             -> for computing a distance-dependent scaling_penalty_force_factor.
//   // Note: In an optimized version, the struct might be set up according to the current timestep's blood pressure (nBC).
//   // Note: In an optimized version, the struct might consider a distance-dependent AND damping-related scaling-/penalty-factor (see notes below).
//   // 
//   // Parameters: selfexplaining.
//   
//   InstationaryElasticity_contactBC_3D(int cbdy1, int cbdy2, int ts, double delta_t, double ContactToleranceThreshold)
// 	: cbdy1_(cbdy1), cbdy2_(cbdy2), ts_(ts), delta_t_(delta_t), ContactToleranceThreshold_(ContactToleranceThreshold) {
// 		assert(DIMENSION == 3);
// 	}
//   
//   double operator()(const int current_bdyNumber_from_assembler, const double curr_ContactDistance) const {
//     
//     double real_time_ = ts_ * delta_t_;
//     double scaling_penalty_force_factor = 1.0;
//     
//     // compute scaling_penalty_force_factor for counter/penalty force: // TASK: improve this!
//     // - possibly similar to the above "InstationaryElasticity_SinusoidalNeumannBC_3D" (Option 2):
//     //   approximate (time-dependent) pressure profile by means of a polynomial-function, but with a negative sign.
//     // - moreover, not only with a distance-dependent scaling-/penalty-factor,
//     //   but also with a distance-dependent AND damping-related scaling-/penalty-factor,
//     //   according to acceleration and velocity: x = 0.5*a*t^2+b*t+x_0, or
//     //   according to force-spring-laws (cp. discussion with Prof. Heuveline on 2015-01-29).
//     
//     // compute scaling_penalty_force_factor as:
//     //  EITHER
//     // (ContactToleranceThreshold_/curr_ContactDistance) - ContactToleranceThreshold_; (which might be too intense),
//     //  OR
//     // (ContactToleranceThreshold_/curr_ContactDistance); (which might be too intense when very close to "real contact"),
//     //  OR 
//     // negative proportional linear relation w.r.t. curr_ContactDistance; (where it seems a proper scaling_factor is 1.5):
//     if ( current_bdyNumber_from_assembler == cbdy1_ ) {
//       
//       scaling_penalty_force_factor = (ContactToleranceThreshold_ / curr_ContactDistance);// - ContactToleranceThreshold_;
//       // OR:
// //       scaling_penalty_force_factor = 1.5 * (ContactToleranceThreshold_ - curr_ContactDistance); // works with xml-ContactPenaltyFactor 1500.
//       
//     } else if ( current_bdyNumber_from_assembler == cbdy2_ ) {
//       
//       scaling_penalty_force_factor = (ContactToleranceThreshold_ / curr_ContactDistance);// - ContactToleranceThreshold_;
//       // OR:
// //       scaling_penalty_force_factor = 1.5 * (ContactToleranceThreshold_ - curr_ContactDistance); // works with xml-ContactPenaltyFactor 1500.
//       
//     } else {
//       std::cout << "ERROR: Problem in ContactBCs-Struct w.r.t. bdy_ numbers. " << std::endl;
//       assert(0);
//     }
//     
//     return scaling_penalty_force_factor;
//   }
//   
//   // Parameters:
//   const int cbdy1_, cbdy2_;
//   const int ts_;
//   const double delta_t_;
//   double real_time_;
//   const double ContactToleranceThreshold_;
// };


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// --------------------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////// Stationary assembler (SystemMatrix and RHS) ////////////////////////////////
class StationaryElasticityAssembler : private AssemblyAssistant<DIMENSION, double> {
public:
    StationaryElasticityAssembler(double lambda, double mu, double rho, double gravity)
        : lambda_(lambda), mu_(mu), rho_(rho), gravity_(gravity) {}

    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalMatrix& lm) {
        AssemblyAssistant<DIMENSION, double>::initialize_for_element(element, quadrature);

        const int num_q = num_quadrature_points();

        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // implement the weak formulation of the elasticity boundary value problem in the stationary case:

            // assemble \int {lambda \div(u) \div(phi)}
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
            				wq * lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * dJ;
            		}
		    }
            	}
            }
            
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)^T) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            // which corresponds to:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            for (int var = 0; var < DIMENSION; ++var) {
                   for (int i = 0; i < num_dofs(var); ++i) {
                       for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lm(dof_index(i, var), dof_index(j, var)) +=
                                    wq * mu_ * (grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob] * dJ;
                            lm(dof_index(i, var), dof_index(j, var_frob)) +=
                                    wq * mu_ * (grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob] * dJ;
                        }
                    }
                }
            }
            
        }
    }
    
    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalVector& lv) {
        AssemblyAssistant<DIMENSION, double>::initialize_for_element(element, quadrature);
	
        const int num_q = num_quadrature_points();
	
	const double source[3] = {0., 0., gravity_ * rho_};
	
        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // implement the weak formulation of the elasticity boundary value problem in the stationary case:
            // assemble l(v) = \rho * \int( f_ext \phi )
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
                    lv[dof_index(i, test_var)] +=
                        wq * source[test_var] * phi(i, q, test_var) * dJ;
                }
	    }
        }
    }

private:
    double lambda_, mu_, rho_, gravity_;
};

//////////////// Stationary assembler (Neumann BC) ////////////////////////////////
template<class Elasticity_Neumann_Evaluator> 
class StationaryElasticityNeumannAssembler : private AssemblyAssistant<DIMENSION, double> {
public:
	StationaryElasticityNeumannAssembler(const Elasticity_Neumann_Evaluator* elast_neum_eval, int bdy)
	: elast_neum_eval_(elast_neum_eval), bdy_(bdy) {}
	
	void operator()(const Element<double>& element, int facet_number, const Quadrature<double>& quadrature, LocalVector& lv) {
		
		// Procedure to get the facet entity
		mesh::IncidentEntityIterator facet = element.get_cell().begin_incident(DIMENSION-1);
		for (int i=0; i<facet_number; ++i, ++facet ) {}

		// Check if it belongs to the boundary of the respective boundary_number bdy_.
		if (facet->get_material_number() != bdy_) return;

		// Initialize the quadrature for integration over the facet which is subject to Neumann BCs
		AssemblyAssistant<DIMENSION, double>::initialize_for_facet(element, quadrature, facet_number);

		const int num_q = num_quadrature_points();
		
// 		// Constant defined Neumann BC Source:
// 		// -> Underlying modelling assumption: pressure gradient on MV = 120mmHg(SystoleVentricle) - 20mmHg(DiastoleVentricle) = 100mmHg.
// 		// -> Underlying modelling assumption: pressure gradient on MV = 120mmHg(SystoleVentricle) - 80mmHg(DiastoleAorta) = 40mmHg.
// 		double nbc_source[3] = {0., 0., 40.}; // assuming the force acts in z-direction only and is constant.
// 		
// 		// loop over quadrature points.
// 		for (int q = 0; q < num_q; ++q) {
// 		  const double wq = w(q);
// 		  const double dsurf = ds(q);
// 		  
// 		  // implement the weak formulation of the elasticity boundary value problem in the stationary case:
// 		  // assemble \int_bdy{ nbc_surface_force * v }
// 		  for (int v_var = 0; v_var < DIMENSION; ++v_var) {
// 		    for (int i = 0; i < num_dofs(v_var); ++i) {
// 			lv[dof_index(i, v_var)] +=  wq * (nbc_source[v_var] * phi(i, q, v_var)) * dsurf;
// 		    }
// 		  }
// 		}
		
		// Flexibly defined Neumann BC Source (w.r.t. Neumann BC struct):
		// loop over quadrature points.
		for (int q = 0; q < num_q; ++q) {
		  const double wq = w(q);
		  const double dsurf = ds(q);
		  const Vec<DIMENSION, double> xq = x(q);
		  
		  // implement the weak formulation of the elasticity boundary value problem in the stationary case:
		  // assemble \int_bdy{ nbc_surface_force * v }
		  for (int v_var = 0; v_var < DIMENSION; ++v_var) {
		    for (int i = 0; i < num_dofs(v_var); ++i) {
			lv[dof_index(i, v_var)] +=  wq * (elast_neum_eval_[v_var](xq) * phi(i, q, v_var)) * dsurf;
		    }
		  }
		}
	}

private:
	const Elasticity_Neumann_Evaluator* elast_neum_eval_; // Template in order to allow implementation of different nBC-types (radial, facet, etc.).
	//const double force_vec_[3];
	const int bdy_;
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// --------------------------------------------------------------------------------------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//////////////// Instationary assembler (SystemMatrix and RHS) COROTATIONAL FORMULATION //////////////////////////////
class InstationaryCorotElasticityAssembler : private AssemblyAssistant<DIMENSION, double> {
 public:
    InstationaryCorotElasticityAssembler(double lambda, double mu, double rho, double gravity, 
				         double dampingFactor, double rayleighAlpha, double rayleighBeta)
        : lambda_(lambda), mu_(mu), rho_(rho), gravity_(gravity), 
        dampingFactor_(dampingFactor), rayleighAlpha_(rayleighAlpha), rayleighBeta_(rayleighBeta), time_(0.0) {}
    // TODO: make materials element-specific, by means of handing over element-specific parameters 
    //       based on a element-ID in the geometry input file.
    
    
    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalMatrix& lm) {
        AssemblyAssistant<DIMENSION, double>::initialize_for_element( element, quadrature );
	
        /*
	 * I) Compute standard local stiffness matrix K_elem as usual (i.e. as in the linear elasticity formulation):
	 */
	
        const int num_q = num_quadrature_points();
	
        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // implement the weak formulation of the elasticity boundary value problem in the instationary case:
	    
	    // Compute Stiffness Matrix K.
            // assemble \int {lambda \div(u) \div(phi)}
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
            				wq * lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * dJ;
//             				* (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
            		}
		    }
            	}
            }
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)^T) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            // which corresponds to:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] } 
            // yields:
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lm(dof_index(i, var), dof_index(j, var)) +=
                                    wq * mu_ * (grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob] * dJ;
//                                     * (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                            lm(dof_index(i, var), dof_index(j, var_frob)) +=
                                    wq * mu_ * (grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob] * dJ;
//                                     * (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                        }
                    }
                }
            }
            // End of K.
            
//             // Compute Mass Matrix part.
//             // assemble \int {Nalpha0_ * (rho_ * dot(phi_j, phi_i) ) } 
//             // PLUS (applying Rayleigh)
//             // yields:
//             // assemble \int {(Nalpha0_ + RayleighAlpha * Nalpha1_) * (rho_ * dot(phi_j, phi_i) ) }
//             for (int var = 0; var < DIMENSION; ++var) {
// 	        // Note: only one loop needed since dot-product / scalar product / symmetry.
// 		const int n_dofs = num_dofs(var);
// 		for (int i = 0; i < n_dofs; ++i) { // num_dofs(test_var)
// 		    for (int j = 0; j < n_dofs; ++j) { // num_dofs(trial_var)
// 			lm(dof_index(i, var), dof_index(j, var)) +=
// 			     (Nalpha0_  + dampingFactor_ * rayleighAlpha_ * Nalpha1_) * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
// 			//std::cout << "TestOutput: phi(j,q,var) = " << phi(j,q,var) << " and phi(i,q,var) = " << phi(i,q,var) << std::endl;
// 		    }
// 		}
// 	       }
// 	       // End of Mass Matrix part.
// 	    
// 	    // Compute Damping Matrix part:
// 	    // assemble \int {dampingFactor_ * Nalpha1_ * (rayleighAlpha_*M + rayleighBeta_*K) } // by default dampingFactor_ = 1.0;
// 	    // This is done above in the respective Stiffness and Mass Matrices part, using the Rayleigh approach).
// 	    // End of Damping Matrix part.
        
        } // end of assembly of local stiffness matrix K_e
        
        /*
	 * I) Compute rotation Q_elem_T from deformed to initial element
	 */
	
	// Preliminary computation of previous displacement vectors.
	for (int d = 0; d < DIMENSION; ++d) {
	    sol_prev_c_[d].clear();
	    evaluate_fe_function(*prev_ts_sol_, d, sol_prev_c_[d]);
	}
	
	// compute intpts of initial and of deformed element.
// 	std::vector<double> initial_intpts_coords;
// 	initial_intpts_coords.reserve( num_q * DIMENSION );
// 	std::vector<double> deformed_intpts_coords;
// 	deformed_intpts_coords.reserve( num_q * DIMENSION );
	
	Vec<DIMENSION, double> sol_prev;
	
	// Initialize new Eigen::MatrixXd type to store initial and deformed coords.
	Eigen::Matrix<double, DIMENSION, 4, Eigen::ColMajor> initial_coords; // TODO TODO TODO replace 4 by constant expression (at compile time) for num_q
	Eigen::Matrix<double, DIMENSION, 4, Eigen::ColMajor> deformed_coords; // TODO TODO TODO replace 4 by constant expression (at compile time) for num_q
// 	Eigen::MatrixXd deformed_coords( DIMENSION, num_q );
	
	// loop over all quadrature points (intpts).
        for (int q = 0; q < num_q; ++q) { // only for DIMENSION == 3 // x(q) is a std::vector<value_type>
	    
	    std::vector<double> quad_coords { x(q)[0], x(q)[1], x(q)[2] }; // this is for DIMENSION 3 only  // TODO: Optimize.
	    std::copy( quad_coords.begin(), quad_coords.end(), initial_coords.data() + (q * DIMENSION) ); // if matrix is column major
	    
// 	    std::copy( x(q).begin(), x(q).end(), initial_coords.data() + (q * DIMENSION) ); // if matrix is column major
//          initial_intpts_coords[0 + q*DIMENSION] = x(q)[0];
// 	    initial_intpts_coords[1 + q*DIMENSION] = x(q)[1];
// #if DIMENSION == 3
// 	    initial_intpts_coords[2 + q*DIMENSION] = x(q)[2];
// #endif
	    
	    for (int var = 0; var < DIMENSION; ++var) {
		sol_prev[var] = sol_prev_c_[var][q];
            }
            
            std::vector<double> disp_field { x(q)[0] + sol_prev[0], x(q)[1] + sol_prev[1], x(q)[2] + sol_prev[2] }; // this is for DIMENSION 3 only  // TODO: Optimize.
            
            std::copy( disp_field.begin(), disp_field.end(), deformed_coords.data() + (q * DIMENSION) ); // if matrix is column major
//          deformed_intpts_coords[0 + q*DIMENSION] = x(q)[0] + sol_prev[0];
// 	    deformed_intpts_coords[1 + q*DIMENSION] = x(q)[1] + sol_prev[1];
// #if DIMENSION == 3
// 	    deformed_intpts_coords[2 + q*DIMENSION] = x(q)[2] + sol_prev[2];
// #endif
	}
	
	// Eigen::MatrixXd initial_coords( DIMENSION, num_q );
	// Eigen::MatrixXd deformed_coords( DIMENSION, num_q );
	Eigen::MatrixXd Q_elem( DIMENSION, num_q ); // Element rotation matrix.
	
	// Note: replace this by some internal "reshape()" function in order to avoid the for loops. // Todo: Optimize. Not needed anymore.
// 	for( int i = 0; i < DIMENSION; ++i ) {
// 	  for( int j = 0; j < num_q; ++j ) {
// 	    deformed_coords( i,j ) = deformed_intpts_coords[ j*DIMENSION + i ];
// 	    initial_coords( i,j ) = initial_intpts_coords[ j*DIMENSION + i ];
// 	  }
// 	}
	
	Eigen::MatrixXd initial_coords2 = initial_coords;
	Eigen::MatrixXd deformed_coords2 = deformed_coords;
	// Move coords to origin in order to minimize in "find_rotation()" below.
	move_center_to_origin( initial_coords2 );
	move_center_to_origin( deformed_coords2 );
	
	// Compute Q_elem (to rotate the initial element into the deformed configuration)
	find_rotation( initial_coords2, deformed_coords2, Q_elem );
	// Compute Q_elem_T (to rotate the deformed element into the initial configuration).
	// Eigen::MatrixXd Q_elem_T = Q_elem.transpose();
	
	/*
	 * II) Transform local stiffness matrix accordingly in order to obtain the corotated local stiffness matrix
	 * lm = Q * lm * Q.trans
	 * MatrixMult(in, out) Rechtsmult.
	 */
	
	// Therefore, first perform permutation in order to get from HiFlow3-DOF-numbering to Georgii's DOF numbering.
	// FOR THE PERMUTATION: HERE BELOW. /////////////////////////////////////////////////////////////
	
	Eigen::MatrixXd Q_e_G; // rotation matrix according to Georgii's article
	Q_e_G = Eigen::MatrixXd::Zero( num_q * DIMENSION, num_q * DIMENSION );
	
	for( int b = 0; b < num_q; ++b )
	  for( int i = 0; i < DIMENSION; ++i )
	    for( int j = 0; j < DIMENSION; ++j )
	      Q_e_G( b*DIMENSION + i, b*DIMENSION + j ) = Q_elem( i, j ); // TODO: Optimize. (e.g. using Eigen lib, to copy blockwise)
	
	// Define Permutation Matrix 
	// to transform the rotation matrix so that it can be used 
	// together with the HiFlow3 internal enumeration scheme.
	Eigen::VectorXi per( num_q * DIMENSION ); // this vector defines the permutation i -> per[i]
// 	per << 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11; // permutation form Georgii's enum. to hiflow enum.
// 	per << 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11; // permutation from hiflow enumeration to Georgii's enum.
	int counter = 0;
	// permutation from hiflow enumeration to Georgii's enum.
	for( int i = 0; i < DIMENSION; ++i ) {
	  for (int j = 0; j < num_q; ++j) {
	    per( counter ) = j * DIMENSION + i; // TODO optimize. maybe use std::fill with lambda expression
	    ++counter;
	  }
	}
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P( per );
	Eigen::MatrixXi PHG( P.toDenseMatrix() ); // permutation from hiflow enum. to Georgii's enum. as dense matrix
	//PHG = P.toDenseMatrix(); // TODO check if this step is necessary
	Eigen::MatrixXd PHGd;
	PHGd = PHG.cast <double> (); // need to cast for matrix product below
	
	Eigen::MatrixXd Q_e_H; // permutation matrix compatible with hiflow enumeration
	Q_e_H = PHGd.transpose() * Q_e_G * PHGd;
	
        SeqDenseMatrix<double> Q_e, Q_e_T;
	Q_e.Resize( DIMENSION * num_q, DIMENSION * num_q );
	Q_e.Zeros();
	Q_e_T.Resize( DIMENSION * num_q, DIMENSION * num_q );
	Q_e_T.Zeros();
	
	for(int i = 0; i < num_q * DIMENSION; ++i) { // TODO Optimize. see above.
	  for(int j = 0; j < num_q * DIMENSION; ++j) {
	    Q_e  ( i, j ) = Q_e_H( i, j );
	    Q_e_T( i, j ) = Q_e_H( j, i );
	  }
	}
	// FOR THE PERMUTATION: HERE ABOVE. /////////////////////////////////////////////////////////////
	
// 	SeqDenseMatrix<double> Q_e, Q_e_T;
// 	Q_e.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	Q_e.Zeros();
// 	Q_e_T.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	Q_e_T.Zeros();
// 	
// // 	// Option 1: num_q blocks of DIM x DIM matrices. OLD! -> WRONG.
// // 	for( int b = 0; b < num_q; ++b ) {
// // 	  for( int i = 0; i < DIMENSION; ++i ) {
// // 	    for( int j = 0; j < DIMENSION; ++j ) {
// // 	      Q_e  ( b*DIMENSION + i, b*DIMENSION + j ) = Q_elem  ( i, j );
// // 	      Q_e_T( b*DIMENSION + i, b*DIMENSION + j ) = Q_elem_T( i, j );
// // 	    }
// // 	  }
// // 	}
// 	// Option 2: resort Q_elem into Q_e by means of using the dof_index()-numbering as in the hf3 strd assembly. NEW!
// 	for (int test_var = 0; test_var < DIMENSION; ++test_var) {
//             	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
// 		    for (int i = 0; i < num_dofs(test_var); ++i) {
//             		for (int j = 0; j < num_dofs(trial_var); ++j) {
//             			Q_e  ( dof_index(i, test_var), dof_index(j, trial_var) ) = Q_elem  ( test_var, trial_var );
// 				Q_e_T( dof_index(i, test_var), dof_index(j, trial_var) ) = Q_elem_T( test_var, trial_var );
//             		}
// 		    }
//             	}
//         }
	
	SeqDenseMatrix<double> tmp;
	tmp.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	SeqDenseMatrix<double> K_tilde;
// 	K_tilde.Resize( DIMENSION * num_q, DIMENSION * num_q );
	lm.MatrixMult( Q_e_T, tmp );
	Q_e.MatrixMult( tmp, lm ); // K_tilde
	
	/*
	 * III) Compute local K_eff using the above local matrices
	 * lm gets updated:
	 * lm = K_corot + Nalpha0_ * M + Nalpha1_ * C
	 * lm = lm + Nalpha0_ * M + Nalpha1_ * C
	 * where C = RayleighAlpha * M + RayleighBeta * K
	 */

	// loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // Compute Mass Matrix part.
            // assemble \int {Nalpha0_ * (rho_ * dot(phi_j, phi_i) ) } 
            // PLUS (applying Rayleigh)
            // yields:
            // assemble \int {(Nalpha0_ + RayleighAlpha * Nalpha1_) * (rho_ * dot(phi_j, phi_i) ) }
            for (int var = 0; var < DIMENSION; ++var) {
	        // Note: only one loop needed since dot-product / scalar product / symmetry.
		const int n_dofs = num_dofs(var);
		for (int i = 0; i < n_dofs; ++i) { // num_dofs(test_var)
		    for (int j = 0; j < n_dofs; ++j) { // num_dofs(trial_var)
			lm(dof_index(i, var), dof_index(j, var)) +=
			     //Nalpha0_ * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
			     (Nalpha0_  + dampingFactor_ * rayleighAlpha_ * Nalpha1_) * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
			//std::cout << "TestOutput: phi(j,q,var) = " << phi(j,q,var) << " and phi(i,q,var) = " << phi(i,q,var) << std::endl;
		    }
		}
	    }// End of Mass Matrix part.
	    
	    // Compute Damping Matrix part:
	    // 1. part: Compute Stiffness Matrix K.
            // assemble \int {lambda \div(u) \div(phi)}
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
            				wq * lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * dJ
            				* (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
            		}
		    }
            	}
            }
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)^T) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            // which corresponds to:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] } 
            // yields:
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lm(dof_index(i, var), dof_index(j, var)) +=
                                    wq * mu_ * (grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob] * dJ
                                    * (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                            lm(dof_index(i, var), dof_index(j, var_frob)) +=
                                    wq * mu_ * (grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob] * dJ
                                    * (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                        }
                    }
                }
            }
            // End of K.
            // 2. part: Compute Mass Matrix M is done above.
        
            // End of K_eff.
	} // end of loop over intpts.
	
    } // end of lm assembly.
    
    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalVector& lv) {
        
	AssemblyAssistant<DIMENSION, double>::initialize_for_element(element, quadrature);
	
	const int num_q = num_quadrature_points();
	
	// Compute sol_prev_, dt_sol_prev_, and dt2_sol_prev_ values at quadrature points
	// (in order to have them available in the loop over the quadrature points below):
	for (int d = 0; d < DIMENSION; ++d) {
	    sol_prev_c_[d].clear();
	    dt_sol_prev_c_[d].clear();
	    dt2_sol_prev_c_[d].clear();
	    
	    evaluate_fe_function(*prev_ts_sol_, d, sol_prev_c_[d]);
	    evaluate_fe_function(*dt_prev_ts_sol_, d, dt_sol_prev_c_[d]);
	    evaluate_fe_function(*dt2_prev_ts_sol_, d, dt2_sol_prev_c_[d]);
	}
	
        
        /*
	 * Find rotation matrix as above (in order for the computation of the corotational force correction term below).
	 */
	
	// compute intpts of initial and of deformed element.
	std::vector<double> initial_intpts_coords;
	initial_intpts_coords.reserve( num_q * DIMENSION );
	std::vector<double> deformed_intpts_coords;
	deformed_intpts_coords.reserve( num_q * DIMENSION );
	Vec<DIMENSION, double> sol_prev;
	// loop over all quadrature points (intpts).
        for (int q = 0; q < num_q; ++q) { // only for DIMENSION == 3
            initial_intpts_coords[0 + q*DIMENSION] = x(q)[0];
	    initial_intpts_coords[1 + q*DIMENSION] = x(q)[1];
#if DIMENSION == 3
	    initial_intpts_coords[2 + q*DIMENSION] = x(q)[2];
#endif
	    
	    for (int var = 0; var < DIMENSION; ++var) {
		sol_prev[var] = sol_prev_c_[var][q];
            }
            
            deformed_intpts_coords[0 + q*DIMENSION] = x(q)[0] + sol_prev[0];
	    deformed_intpts_coords[1 + q*DIMENSION] = x(q)[1] + sol_prev[1];
#if DIMENSION == 3
	    deformed_intpts_coords[2 + q*DIMENSION] = x(q)[2] + sol_prev[2];
#endif
	}
	
	// Initialize new Eigen::MatrixXd type to store initial and deformed coords.
	Eigen::MatrixXd initial_coords( DIMENSION, num_q );
	Eigen::MatrixXd deformed_coords( DIMENSION, num_q );
	Eigen::MatrixXd Q_elem( DIMENSION, num_q ); // Element rotation matrix.
	
	// Note: replace this by some internal "reshape()" function in order to avoid the for loops. // TODO: Optimize.
	for( int i = 0; i < DIMENSION; ++i ) {
	  for( int j = 0; j < num_q; ++j ) {
	    deformed_coords( i,j ) = deformed_intpts_coords[ j*DIMENSION + i ];
	    initial_coords( i,j ) = initial_intpts_coords[ j*DIMENSION + i ];
	  }
	}
	
	// Move coords to origin in order to minimize in "find_rotation()" below.
	move_center_to_origin( initial_coords );
	move_center_to_origin( deformed_coords );
	
	// Compute Q_elem 
	find_rotation( initial_coords, deformed_coords, Q_elem );
	
	/*
	 * II) Transform local stiffness matrix accordingly in order to obtain the corotated local stiffness matrix
	 * lm = Q * lm * Q.trans
	 * MatrixMult(in, out) Rechtsmult.
	 */
	
	// Therefore, first perform permutation in order to get from HiFlow3-DOF-numbering to Georgii's DOF numbering.
	// FOR THE PERMUTATION: HERE BELOW. /////////////////////////////////////////////////////////////
	
	Eigen::MatrixXd Q_e_G; // Rotation matrix accoring to Georgii
	Q_e_G = Eigen::MatrixXd::Zero( num_q * DIMENSION, num_q * DIMENSION );
	
	for( int b = 0; b < num_q; ++b ) {
	  for( int i = 0; i < DIMENSION; ++i ) {
	    for( int j = 0; j < DIMENSION; ++j ) {
	      Q_e_G( b * DIMENSION + i, b * DIMENSION + j ) = Q_elem( i, j );
	    }
	  }
	}
	
	// Define Permutation Matrix to transform the rotation matrix so that it
	// can be used together with the HiFlow3 internal enumeration scheme
	Eigen::VectorXi per( num_q * DIMENSION );
// 	for( int b = 0; b < num_q; ++b )
// 	  for( int i = 0; i < DIMENSION; ++i )
// 	    per(  ) = 
// 	per << 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11;
// 	per << 0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11;
	int counter = 0;
	for( int i = 0; i < DIMENSION; ++i ) {
	  for (int j = 0; j < num_q; ++j) {
	    per( counter ) = j * DIMENSION + i;
	    ++counter;
	  }
	}
	Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P( per );
	Eigen::MatrixXi PHG;
	PHG = P.toDenseMatrix();
	Eigen::MatrixXd PHGd;
	PHGd = PHG.cast <double> ();
	
	Eigen::MatrixXd Q_e_H;
	Q_e_H = PHGd.transpose() * Q_e_G * PHGd;
	
	SeqDenseMatrix<double> Q_e;
	Q_e.Resize( DIMENSION * num_q, DIMENSION * num_q );
	Q_e.Zeros();
	
	for(int i = 0; i < num_q * DIMENSION; ++i)
	  for(int j = 0; j < num_q * DIMENSION; ++j)
	    Q_e( i, j ) = Q_e_H( i, j );
        
        // FOR THE PERMUTATION: HERE ABOVE. /////////////////////////////////////////////////////////////
	
// 	SeqDenseMatrix<double> Q_e;
// 	Q_e.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	Q_e.Zeros();
// 	
// // 	// Option 1: num_q blocks of DIM x DIM matrices. OLD! -> WRONG.
// // 	for( int b = 0; b < num_q; ++b ) {
// // 	  for( int i = 0; i < DIMENSION; ++i ) {
// // 	    for( int j = 0; j < DIMENSION; ++j ) {
// // 	      Q_e( b*DIMENSION + i, b*DIMENSION + j ) = Q_elem( i, j );
// // 	    }
// // 	  }
// // 	}
// 	// Option 2: resort Q_elem into Q_e by means of using the dof_index()-numbering as in the hf3 strd assembly. NEW!
// 	for (int test_var = 0; test_var < DIMENSION; ++test_var) {
//             	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
// 		    for (int i = 0; i < num_dofs(test_var); ++i) {
//             		for (int j = 0; j < num_dofs(trial_var); ++j) {
//             			Q_e  ( dof_index(i, test_var), dof_index(j, trial_var) ) = Q_elem  ( test_var, trial_var );
//             		}
// 		    }
//             	}
//         }
        
        /*
	 * Computation of the corotational force correction term.
	 */
	
	// Compute (K_e * x_e):
	//loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // K: Stiffness Matrix part 1:
            // assemble \int {lambda \div(u) \div(phi)}
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lv[dof_index(i, test_var)] += wq * dJ 
            			     * ( lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * x(q)[test_var] );
            		}
		    }
            	}
            }
            // K: Stiffness Matrix part 2:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lv[dof_index(i, var)] +=
                                    wq * ( mu_ * ((grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob]) * x(q)[var] ) * dJ;
                            lv[dof_index(i, var)] +=
                                    wq * ( mu_ * ((grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob]) * x(q)[var] ) * dJ;
                        }
                    }
                }
            }
            // End of Stiffness Matrix part.
	} // end of loop over quadrature points.
	
	// Multiply (K_e * x_e) by Q_e:
	std::vector<double> in( lv );
	Q_e.VectorMult( in, lv ); // hence, lv now corresponds to the corotational force correction term f_0.
	
	// end of Computation of the corotational force correction term.
	
	/*
	 * Computation of the standard R_eff force term (without corotational part).
	 */
	
	const double gravity[3] = {0., 0., gravity_ * rho_};
	
        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // get previous time step solutions in vector form
            Vec<DIMENSION, double> sol_prev_c, dt_sol_prev_c, dt2_sol_prev_c;
            for (int var = 0; var < DIMENSION; ++var) {
		sol_prev_c[var] = sol_prev_c_[var][q];
		dt_sol_prev_c[var] = dt_sol_prev_c_[var][q];
		dt2_sol_prev_c[var] = dt2_sol_prev_c_[var][q];
            }
            
            // implement the weak formulation of the elasticity boundary value problem in the instationary case:
            
	    // Compute R_eff (without corotational correction part yet),
	    // consisting of R (standard RHS gravity term), and K (stiffness), M (mass), and D (damping) related terms.
	    
            // R
            // which is: \int { \grav * phi{v} } = \rho * \int( f_ext \phi )
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		  //for (int j = 0; j < num_dofs(test_var); ++j) { // Additional loop. Does not change anything. Cp. Conv-Diff-Tut.
                    lv[dof_index(i, test_var)] +=
                        wq * dJ * gravity[test_var] * phi(i, q, test_var); // * phi(j, q, test_var);
                        //wq * gravity[test_var] * phi(j, q, test_var) * phi(i, q, test_var) * dJ;
		  //}
                }
	    }
	    
	    // M * {a_0 * U + a_2 * dtU + a_3 * dt2U}
	    // which is: 
	    for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		    for (int j = 0; j < num_dofs(test_var); ++j) {
		        lv[dof_index(i, test_var)] +=
			      wq * (
			      rho_ * phi(j,q,test_var) * phi(i,q,test_var) *
			      (Nalpha0_ * sol_prev_c[test_var] + Nalpha2_ * dt_sol_prev_c[test_var] + Nalpha3_ * dt2_sol_prev_c[test_var])
			      ) * dJ;
		    }
		}
	    }
	    
	    // D * {a_1 * U + a_4 * dtU + a_5 * dt2U} --- which is: ...
	    // RayleighAlpha * M * {a_1 * U + a_4 * dtU + a_5 * dt2U} + RayleighBeta * K * {a_1 * U + a_4 * dtU + a_5 * dt2U}
	    // --- which is: ...
	    
	    // RayleighBeta * K: Stiffness Matrix part 1:
            // assemble \int {lambda \div(u) \div(phi)} PLUS (applying Rayleigh)
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lv[dof_index(i, test_var)] +=
            				dampingFactor_ * rayleighBeta_ * wq * (
            				lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] *
            				(Nalpha1_ * sol_prev_c[test_var] + Nalpha4_ * dt_sol_prev_c[test_var] + Nalpha5_ * dt2_sol_prev_c[test_var])
            				) * dJ;
            		}
		    }
            	}
            }
            // RayleighBeta * K: Stiffness Matrix part 2:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lv[dof_index(i, var)] +=
                                    dampingFactor_ * rayleighBeta_ * wq * (
				     mu_ * ((grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob]) *
				     (Nalpha1_ * sol_prev_c[var] + Nalpha4_ * dt_sol_prev_c[var] + Nalpha5_ * dt2_sol_prev_c[var])
				     ) * dJ;
                            lv[dof_index(i, var)] +=
                                    dampingFactor_ * rayleighBeta_ * wq * (
				     mu_ * ((grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob]) *
				     (Nalpha1_ * sol_prev_c[var] + Nalpha4_ * dt_sol_prev_c[var] + Nalpha5_ * dt2_sol_prev_c[var])
				     ) * dJ;
                        }
                    }
                }
            }
            // End of Stiffness Matrix part.
	    
	    // RayleighAlpha * M: Mass Matrix:
	    // M * {a_0 * U + a_2 * dtU + a_3 * dt2U}
	    // which is: 
	    for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		    for (int j = 0; j < num_dofs(test_var); ++j) {
		        lv[dof_index(i, test_var)] +=
			      dampingFactor_ * rayleighAlpha_ * wq * (
			      rho_ * phi(j,q,test_var) * phi(i,q,test_var) *
			      (Nalpha1_ * sol_prev_c[test_var] + Nalpha4_ * dt_sol_prev_c[test_var] + Nalpha5_ * dt2_sol_prev_c[test_var])
			      ) * dJ;
		    }
		}
	    }
	    
        } // end of loop over quadrature points.
        
        // end of Computation of standard force term (without corotational part).
        
    } // end of lv assembly.
    
    void set_timestep_parameters(double theta, double delta_t);
    void set_time(double time);
    void set_prev_solution(const CVector* sol_prev);
    void set_dt_prev_solution(const CVector* dt_sol_prev);
    void set_dt2_prev_solution(const CVector* dt2_sol_prev);
    
    double get_Newmark_TimeIntegration_Constants(int NewmarkConstantNumber);
    
 private:
    double lambda_, mu_, rho_, gravity_;
    double dampingFactor_, rayleighAlpha_, rayleighBeta_;
    
    // For timestepping:
    double time_;
    
    // These parameters specify which time discretization method you choose
    // For the ThetaFamily-Methods: see function set_timestep_parameters(double theta, double delta_t):
    double ThetaFamAlpha1_;
    double ThetaFamAlpha2_;
    double ThetaFamAlpha3_;
    // For the Newmark-Method: see function set_timestep_parameters(double theta, double delta_t):
    double Nalpha_, Ndelta_;
    double Nalpha0_, Nalpha1_, Nalpha2_, Nalpha3_, Nalpha4_, Nalpha5_, Nalpha6_, Nalpha7_;
    
    // Vectors u, v, a of the previous timestep:
    const CVector* prev_ts_sol_; // solution vector (u) at previous timestep.
    const CVector* dt_prev_ts_sol_; // first derivative dt_u of solution vector (u) at previous timestep.
    const CVector* dt2_prev_ts_sol_; // second derivative dt2_u of solution vector (u) at previous timestep.
    
    FunctionValues<double> sol_prev_c_[DIMENSION]; // static array of DIM number of std::vector<double>
    FunctionValues<double> dt_sol_prev_c_[DIMENSION];
    FunctionValues<double> dt2_sol_prev_c_[DIMENSION];
    
}; // end InstationaryCorotElasticityAssembler for COROTATIONAL FORMULATION.

///////////////////////////////////////////////////////////////////////



//////////////// Instationary assembler (SystemMatrix and RHS) LINEAR FORMULATION //////////////////////////////
class InstationaryLinElasticityAssembler : private AssemblyAssistant<DIMENSION, double> {
 public:
    InstationaryLinElasticityAssembler(double lambda, double mu, double rho, double gravity, 
				    double dampingFactor, double rayleighAlpha, double rayleighBeta)
        : lambda_(lambda), mu_(mu), rho_(rho), gravity_(gravity), 
        dampingFactor_(dampingFactor), rayleighAlpha_(rayleighAlpha), rayleighBeta_(rayleighBeta), time_(0.0) {}
    // TODO: make materials element-specific, by means of handing over element-specific parameters 
    //       based on a element-ID in the geometry input file.
    
    
    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalMatrix& lm) {
        AssemblyAssistant<DIMENSION, double>::initialize_for_element( element, quadrature );
	
        const int num_q = num_quadrature_points();
	
        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // implement the weak formulation of the elasticity boundary value problem in the instationary case:
	    
	    // Compute K_eff,
	    // consisting of K (stiffness), M (mass), and RayleighBeta * D (damping).
	    
	    // Compute Stiffness Matrix part.
            // assemble \int {lambda \div(u) \div(phi)} PLUS (applying Rayleigh)
	    // * (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_) in order to account for Rayleigh Damping.
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
            				wq * lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * dJ
            				* (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
            		}
		    }
            	}
            }
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)^T) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            // which corresponds to:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] } 
            // PLUS (applying Rayleigh)
            // yields:
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lm(dof_index(i, var), dof_index(j, var)) +=
                                    wq * mu_ * (grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob] * dJ
                                    * (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                            lm(dof_index(i, var), dof_index(j, var_frob)) +=
                                    wq * mu_ * (grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob] * dJ
                                    * (1.0 + dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
                        }
                    }
                }
            }
            // End of Stiffness Matrix K part.
            
            // Compute Mass Matrix part.
            // assemble \int {Nalpha0_ * (rho_ * dot(phi_j, phi_i) ) } 
            // PLUS (applying Rayleigh)
            // yields:
            // assemble \int {(Nalpha0_ + RayleighAlpha * Nalpha1_) * (rho_ * dot(phi_j, phi_i) ) }
            for (int var = 0; var < DIMENSION; ++var) {
	        // Note: only one loop needed since dot-product / scalar product / symmetry.
		const int n_dofs = num_dofs(var);
		for (int i = 0; i < n_dofs; ++i) { // num_dofs(test_var)
		    for (int j = 0; j < n_dofs; ++j) { // num_dofs(trial_var)
			lm(dof_index(i, var), dof_index(j, var)) +=
			     (Nalpha0_  + dampingFactor_ * rayleighAlpha_ * Nalpha1_) * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
			//std::cout << "TestOutput: phi(j,q,var) = " << phi(j,q,var) << " and phi(i,q,var) = " << phi(i,q,var) << std::endl;
		    }
		}
	    }
	    // End of Mass Matrix part.
	    
	    // Compute Damping Matrix part:
	    // assemble \int {dampingFactor_ * Nalpha1_ * (rayleighAlpha_*M + rayleighBeta_*K) } // by default dampingFactor_ = 1.0;
	    // This is done above in the respective Stiffness and Mass Matrices part, using the Rayleigh approach).
	    // End of Damping Matrix part.
	    
        } // end of loop over quadrature points.
        
        
//         /*
// 	 * I) Compute rotation from defomed to initial element
// 	 */
// 	
// 	// Preliminary computation of previous displacement vectors.
// 	for (int d = 0; d < DIMENSION; ++d) {
// 	    sol_prev_c_[d].clear();
// 	    evaluate_fe_function(*prev_ts_sol_, d, sol_prev_c_[d]);
// 	}
// 	
// 	// compute intpts of initial and of deformed element.
// 	std::vector<double> initial_intpts_coords;
// 	initial_intpts_coords.reserve( num_q * DIMENSION );
// 	std::vector<double> deformed_intpts_coords;
// 	deformed_intpts_coords.reserve( num_q * DIMENSION );
// 	Vec<DIMENSION, double> sol_prev;
// 	// loop over all quadrature points (intpts).
//         for (int q = 0; q < num_q; ++q) { // only for DIMENSION == 3
//             initial_intpts_coords[0 + q*DIMENSION] = x(q)[0];
// 	    initial_intpts_coords[1 + q*DIMENSION] = x(q)[1];
// 	    initial_intpts_coords[2 + q*DIMENSION] = x(q)[2];
// 	    
// 	    for (int var = 0; var < DIMENSION; ++var) {
// 		sol_prev[var] = sol_prev_c_[var][q];
//             }
//             
//             deformed_intpts_coords[0 + q*DIMENSION] = x(q)[0] + sol_prev[0];
// 	    deformed_intpts_coords[1 + q*DIMENSION] = x(q)[1] + sol_prev[1];
// 	    deformed_intpts_coords[2 + q*DIMENSION] = x(q)[2] + sol_prev[2];
// 	}
// 	
// 	// Initialize new Eigen::MatrixXd type to store initial and deformed coords.
// 	Eigen::MatrixXd initial_coords( DIMENSION, num_q );
// 	Eigen::MatrixXd deformed_coords( DIMENSION, num_q );
// 	Eigen::MatrixXd Q_elem( DIMENSION, num_q ); // Element rotation matrix.
// 	
// 	// Note: replace this by some internal "reshape()" function in order to avoid the for loops. // Todo: Optimize.
// 	for( int i = 0; i < DIMENSION; ++i ) {
// 	  for( int j = 0; j < num_q; ++j ) {
// 	    deformed_coords( i,j ) = deformed_intpts_coords[ j*DIMENSION + i ];
// 	    initial_coords( i,j ) = initial_intpts_coords[ j*DIMENSION + i ];
// 	  }
// 	}
// 	
// 	// Move coords to origin in order to minimize in "find_rotation()" below.
// 	move_center_to_origin( initial_coords );
// 	move_center_to_origin( deformed_coords );
// 	
// 	// Compute Q_elem 
// 	find_rotation( initial_coords, deformed_coords, Q_elem );
// 	// Compute Q_elem_T (to rotate the deformed element into the initial configuration).
// 	Eigen::MatrixXd Q_elem_T = Q_elem.transpose();
// 	
// 	/*
// 	 * II) Transform local stiffness matrix accordingly
// 	 * lm = Q * lm * Q.trans
// 	 * MatrixMult(in, out) Rechtsmult.
// 	 */
// 	
// 	SeqDenseMatrix<double> Q_e, Q_e_T;
// 	Q_e.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	Q_e.Zeros();
// 	Q_e_T.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	Q_e_T.Zeros();
// 	
// 	// Option 2: resort Q_elem into Q_e by means of using the dof_index()-numbering as in the hf3 strd assembly. NEW!
// 	for (int test_var = 0; test_var < DIMENSION; ++test_var) {
//             	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
// 		    for (int i = 0; i < num_dofs(test_var); ++i) {
//             		for (int j = 0; j < num_dofs(trial_var); ++j) {
//             			Q_e  ( dof_index(i, test_var), dof_index(j, trial_var) ) = Q_elem  ( test_var, trial_var );
// 				Q_e_T( dof_index(i, test_var), dof_index(j, trial_var) ) = Q_elem_T( test_var, trial_var );
//             		}
// 		    }
//             	}
//         }
// 	
// 	SeqDenseMatrix<double> tmp;
// 	tmp.Resize( DIMENSION * num_q, DIMENSION * num_q );
// // 	SeqDenseMatrix<double> K_tilde;
// // 	K_tilde.Resize( DIMENSION * num_q, DIMENSION * num_q );
// 	lm.MatrixMult( Q_e_T, tmp );
// 	Q_e.MatrixMult( tmp, lm ); // K_tilde
// 	
// 	/*
// 	 * III) Compute local K_eff using the above local matrices
// 	 * lm gets updated:
// 	 * lm = K_corot + Nalpha0_ * M + Nalpha1_ * C
// 	 * lm = lm + Nalpha0_ * M + Nalpha1_ * C
// 	 * where C = RayleighAlpha * M + RayleighBeta * K
// 	 */
// 
// 	// loop over quadrature points.
//         for (int q = 0; q < num_q; ++q) {
//             const double wq = w(q);
//             const double dJ = std::abs(detJ(q));
// 	    
// 	    // Compute Mass Matrix part.
//             // assemble \int {Nalpha0_ * (rho_ * dot(phi_j, phi_i) ) } 
//             // PLUS (applying Rayleigh)
//             // yields:
//             // assemble \int {(Nalpha0_ + RayleighAlpha * Nalpha1_) * (rho_ * dot(phi_j, phi_i) ) }
//             for (int var = 0; var < DIMENSION; ++var) {
// 	        // Note: only one loop needed since dot-product / scalar product / symmetry.
// 		const int n_dofs = num_dofs(var);
// 		for (int i = 0; i < n_dofs; ++i) { // num_dofs(test_var)
// 		    for (int j = 0; j < n_dofs; ++j) { // num_dofs(trial_var)
// 			lm(dof_index(i, var), dof_index(j, var)) +=
// 			     //Nalpha0_ * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
// 			     (Nalpha0_  + dampingFactor_ * rayleighAlpha_ * Nalpha1_) * wq * rho_ * phi(j,q,var) * phi(i,q,var) * dJ;
// 			//std::cout << "TestOutput: phi(j,q,var) = " << phi(j,q,var) << " and phi(i,q,var) = " << phi(i,q,var) << std::endl;
// 		    }
// 		}
// 	    }// End of Mass Matrix part.
// 	    
// 	    // Compute Damping Matrix part:
// 	    // 1. part: Compute Stiffness Matrix K.
//             // assemble \int {lambda \div(u) \div(phi)}
//             for (int test_var = 0; test_var < DIMENSION; ++test_var) {
//             	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
// 		    for (int i = 0; i < num_dofs(test_var); ++i) {
//             		for (int j = 0; j < num_dofs(trial_var); ++j) {
//             			lm(dof_index(i, test_var), dof_index(j, trial_var)) +=
//             				wq * lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] * dJ
//             				* (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
//             		}
// 		    }
//             	}
//             }
//             // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)^T) + \frob(\nabla(u)^T\nabla(\phi)) ] }
//             // which corresponds to:
//             // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] } 
//             // yields:
//             for (int var = 0; var < DIMENSION; ++var) {
//                 for (int i = 0; i < num_dofs(var); ++i) {
//                     for (int j = 0; j < num_dofs(var); ++j) {
//                         for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
//                             lm(dof_index(i, var), dof_index(j, var)) +=
//                                     wq * mu_ * (grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob] * dJ
//                                     * (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
//                             lm(dof_index(i, var), dof_index(j, var_frob)) +=
//                                     wq * mu_ * (grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob] * dJ
//                                     * (dampingFactor_ * rayleighBeta_ * Nalpha1_); // last line is accounting for Rayleigh Damping.
//                         }
//                     }
//                 }
//             }
//             // End of K.
//             // 2. part: Compute Mass Matrix M is done above.
//         
//             // End of K_eff.
// 	} // end of loop over intpts.
// 	
    } // end of assembly of local stiffness matrix K_eff (lm).
    ///////////////////////////////////////
    
    void operator()(const Element<double>& element, const Quadrature<double>& quadrature, LocalVector& lv) {
        
	AssemblyAssistant<DIMENSION, double>::initialize_for_element(element, quadrature);
	
	const int num_q = num_quadrature_points();
	
	// Compute sol_prev_, dt_sol_prev_, and dt2_sol_prev_ values at quadrature points
	// (in order to have them available in the loop over the quadrature points below):
	for (int d = 0; d < DIMENSION; ++d) {
	    sol_prev_c_[d].clear();
	    dt_sol_prev_c_[d].clear();
	    dt2_sol_prev_c_[d].clear();
	    
	    evaluate_fe_function(*prev_ts_sol_, d, sol_prev_c_[d]);
	    evaluate_fe_function(*dt_prev_ts_sol_, d, dt_sol_prev_c_[d]);
	    evaluate_fe_function(*dt2_prev_ts_sol_, d, dt2_sol_prev_c_[d]);
	}
	
	/*
	 * Computation of the standard R_eff force term (without corotational part).
	 */
	
	const double gravity[3] = {0., 0., gravity_ * rho_};
	
        // loop over quadrature points.
        for (int q = 0; q < num_q; ++q) {
            const double wq = w(q);
            const double dJ = std::abs(detJ(q));
	    
	    // get previous time step solutions in vector form
            Vec<DIMENSION, double> sol_prev_c, dt_sol_prev_c, dt2_sol_prev_c;
            for (int var = 0; var < DIMENSION; ++var) {
		sol_prev_c[var] = sol_prev_c_[var][q];
		dt_sol_prev_c[var] = dt_sol_prev_c_[var][q];
		dt2_sol_prev_c[var] = dt2_sol_prev_c_[var][q];
            }
            
            // implement the weak formulation of the elasticity boundary value problem in the instationary case:
            
	    // Compute R_eff,
	    // consisting of R (standard RHS gravity term), and K (stiffness), M (mass), and D (damping) related terms.
	    
            // R
            // which is: \int { \grav * phi{v} } = \rho * \int( f_ext \phi )
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		  //for (int j = 0; j < num_dofs(test_var); ++j) { // Additional loop. Does not change anything. Cp. Conv-Diff-Tut.
                    lv[dof_index(i, test_var)] +=
                        wq * dJ * gravity[test_var] * phi(i, q, test_var); // * phi(j, q, test_var);
                        //wq * gravity[test_var] * phi(j, q, test_var) * phi(i, q, test_var) * dJ;
		  //}
                }
	    }
	    
	    // M * {a_0 * U + a_2 * dtU + a_3 * dt2U}
	    // which is: 
	    for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		    for (int j = 0; j < num_dofs(test_var); ++j) {
		        lv[dof_index(i, test_var)] +=
			      wq * (
			      rho_ * phi(j,q,test_var) * phi(i,q,test_var) *
			      (Nalpha0_ * sol_prev_c[test_var] + Nalpha2_ * dt_sol_prev_c[test_var] + Nalpha3_ * dt2_sol_prev_c[test_var])
			      ) * dJ;
		    }
		}
	    }
	    
	    // D * {a_1 * U + a_4 * dtU + a_5 * dt2U} --- which is: ...
	    // RayleighAlpha * M * {a_1 * U + a_4 * dtU + a_5 * dt2U} + RayleighBeta * K * {a_1 * U + a_4 * dtU + a_5 * dt2U}
	    // --- which is: ...
	    
	    // RayleighBeta * K: Stiffness Matrix part 1:
            // assemble \int {lambda \div(u) \div(phi)} PLUS (applying Rayleigh)
            for (int test_var = 0; test_var < DIMENSION; ++test_var) {
            	for (int trial_var = 0; trial_var < DIMENSION; ++trial_var) {
		    for (int i = 0; i < num_dofs(test_var); ++i) {
            		for (int j = 0; j < num_dofs(trial_var); ++j) {
            			lv[dof_index(i, test_var)] +=
            				dampingFactor_ * rayleighBeta_ * wq * (
            				lambda_ * grad_phi(j, q, trial_var)[trial_var] * grad_phi(i, q, test_var)[test_var] *
            				(Nalpha1_ * sol_prev_c[test_var] + Nalpha4_ * dt_sol_prev_c[test_var] + Nalpha5_ * dt2_sol_prev_c[test_var])
            				) * dJ;
            		}
		    }
            	}
            }
            // RayleighBeta * K: Stiffness Matrix part 2:
            // assemble \int {mu [ \frob(\nabla(u)\nabla(\phi)) + \frob(\nabla(u)^T\nabla(\phi)) ] }
            for (int var = 0; var < DIMENSION; ++var) {
                for (int i = 0; i < num_dofs(var); ++i) {
                    for (int j = 0; j < num_dofs(var); ++j) {
                        for (int var_frob = 0; var_frob < DIMENSION; ++var_frob) {
                            lv[dof_index(i, var)] +=
                                    dampingFactor_ * rayleighBeta_ * wq * (
				     mu_ * ((grad_phi(j, q, var)[var_frob]) * grad_phi(i, q, var)[var_frob]) *
				     (Nalpha1_ * sol_prev_c[var] + Nalpha4_ * dt_sol_prev_c[var] + Nalpha5_ * dt2_sol_prev_c[var])
				     ) * dJ;
                            lv[dof_index(i, var)] +=
                                    dampingFactor_ * rayleighBeta_ * wq * (
				     mu_ * ((grad_phi(j, q, var_frob)[var]) * grad_phi(i, q, var)[var_frob]) *
				     (Nalpha1_ * sol_prev_c[var] + Nalpha4_ * dt_sol_prev_c[var] + Nalpha5_ * dt2_sol_prev_c[var])
				     ) * dJ;
                        }
                    }
                }
            }
            // End of Stiffness Matrix part.
	    
	    // RayleighAlpha * M: Mass Matrix:
	    // M * {a_0 * U + a_2 * dtU + a_3 * dt2U}
	    // which is: 
	    for (int test_var = 0; test_var < DIMENSION; ++test_var) {
                for (int i = 0; i < num_dofs(test_var); ++i) {
		    for (int j = 0; j < num_dofs(test_var); ++j) {
		        lv[dof_index(i, test_var)] +=
			      dampingFactor_ * rayleighAlpha_ * wq * (
			      rho_ * phi(j,q,test_var) * phi(i,q,test_var) *
			      (Nalpha1_ * sol_prev_c[test_var] + Nalpha4_ * dt_sol_prev_c[test_var] + Nalpha5_ * dt2_sol_prev_c[test_var])
			      ) * dJ;
		    }
		}
	    }
	    
        } // end of loop over quadrature points.
        
        // end of Computation of standard force term (without corotational part).
        
    } // end of lv assembly.
    ////////////////////////
    
    void set_timestep_parameters(double theta, double delta_t);
    void set_time(double time);
    void set_prev_solution(const CVector* sol_prev);
    void set_dt_prev_solution(const CVector* dt_sol_prev);
    void set_dt2_prev_solution(const CVector* dt2_sol_prev);
    
    double get_Newmark_TimeIntegration_Constants(int NewmarkConstantNumber);
    
 private:
    double lambda_, mu_, rho_, gravity_;
    double dampingFactor_, rayleighAlpha_, rayleighBeta_;
    
    // For timestepping:
    double time_;
    
    // These parameters specify which time discretization method you choose
    // For the ThetaFamily-Methods: see function set_timestep_parameters(double theta, double delta_t):
    double ThetaFamAlpha1_;
    double ThetaFamAlpha2_;
    double ThetaFamAlpha3_;
    // For the Newmark-Method: see function set_timestep_parameters(double theta, double delta_t):
    double Nalpha_, Ndelta_;
    double Nalpha0_, Nalpha1_, Nalpha2_, Nalpha3_, Nalpha4_, Nalpha5_, Nalpha6_, Nalpha7_;
    
    // Vectors u, v, a of the previous timestep:
    const CVector* prev_ts_sol_; // solution vector (u) at previous timestep.
    const CVector* dt_prev_ts_sol_; // first derivative dt_u of solution vector (u) at previous timestep.
    const CVector* dt2_prev_ts_sol_; // second derivative dt2_u of solution vector (u) at previous timestep.
    
    FunctionValues<double> sol_prev_c_[DIMENSION]; // static array of DIM number of std::vector<double>
    FunctionValues<double> dt_sol_prev_c_[DIMENSION];
    FunctionValues<double> dt2_sol_prev_c_[DIMENSION];
    
}; // end InstationaryLinElasticityAssembler for LINEAR ELASTICITY FORMULATION.

///////////////////////////////////////////////////////////////////////////////////

//////////////// Instationary assembler (Neumann BC) //////////////////////////////
template<class Elasticity_Neumann_Evaluator> 
class InstationaryElasticityNeumannAssembler : private AssemblyAssistant<DIMENSION, double> {
 public:
    InstationaryElasticityNeumannAssembler(const Elasticity_Neumann_Evaluator* elast_neum_eval, int bdy1, int bdy2, 
					    double ts, double delta_t)
	: elast_neum_eval_(elast_neum_eval), bdy1_(bdy1), bdy2_(bdy2), ts_(ts), delta_t_(delta_t) {}
    
    void operator()(const Element<double>& element, int facet_number, const Quadrature<double>& quadrature, LocalVector& lv) {
      
      // Procedure to get the facet entity
      mesh::IncidentEntityIterator facet = element.get_cell().begin_incident(DIMENSION-1);
      for (int i=0; i<facet_number; ++i, ++facet ) {} // this for-loop only performs the ++facet task until the right facet is reached.
      
      // Check if it belongs to the boundary specified by the material number "bdy_"
      if ( facet->get_material_number() == bdy1_ || facet->get_material_number() == bdy2_ ) {
         
         // Initialize the quadrature for integration over the facet which is subject to Neumann BCs
         AssemblyAssistant<DIMENSION, double>::initialize_for_facet(element, quadrature, facet_number);
         
         const int num_q = num_quadrature_points();
	 
         double scaling_force_factor = 0.0;
	 
         // Model assumption for Neumann BC force/pressure: // OLD
         // -> pressure gradient on MV = 120mmHg(SystoleVentricle) - 20mmHg(DiastoleVentricle) = 100mmHg.
         // -> pressure gradient on MV = 120mmHg(SystoleVentricle) - 80mmHg(DiastoleAorta) = 40mmHg.
         // double nbc_source[3] = {0., 0., 40.}; // assuming the force acts in z-direction only and is constant.
	 
	 // Get scaling_force_factor for facet normal of facet with respective material number "bdy_":
	 if ( facet->get_material_number() == bdy1_ ) {
	   scaling_force_factor = elast_neum_eval_[0](bdy1_); // = (*elast_neum_eval_)(bdy1_);
	 } else if ( facet->get_material_number() == bdy2_ ) {
	   scaling_force_factor = elast_neum_eval_[0](bdy2_); // = (*elast_neum_eval_)(bdy2_);
	 } else {
	   std::cout << "Error: Problem in NBC-assembler processing NeumannBCs w.r.t. bdy_[.] numbers. " << std::endl;
	   assert(0);
	 }
	 
         // loop over quadrature points.
         for (int q = 0; q < num_q; ++q) {
	   const double wq = w(q);
	   const double dsurf = ds(q);
	   //const Vec<DIMENSION, double> xq = x(q); // OLD
	   const Vec<DIMENSION, double> nq = n(q); // surface normal
	   
	   // implement the weak formulation of the elasticity boundary value problem in the instationary case:
	   // assemble \int_bdy{ nbc_surface_force * v }
	   for (int v_var = 0; v_var < DIMENSION; ++v_var) {
	     for (int i = 0; i < num_dofs(v_var); ++i) {
	       lv[dof_index(i, v_var)] +=  wq * scaling_force_factor * nq[v_var] * phi(i, q, v_var) * dsurf;
	       //lv[dof_index(i, v_var)] +=  wq * (scaled_nbc_facet_normal[v_var] * phi(i, q, v_var)) * dsurf;
	       //lv[dof_index(i, v_var)] +=  wq * (nbc_source[v_var] * phi(i, q, v_var)) * dsurf; // OLD
	       //lv[dof_index(i, v_var)] +=  wq * (elast_neum_eval_[v_var](xq) * phi(i, q, v_var)) * dsurf; // OLD
	     }
	   }
         }
      }
      
    }
    
 private:
    const Elasticity_Neumann_Evaluator* elast_neum_eval_; // Template in order to allow implementation of different nBC-types (radial, orthogonal, facet-wise, etc.).
    const int bdy1_, bdy2_;
    double ts_, delta_t_;
};

//////////////// Instationary assembler (Contact BC) //////////////////////////////
template<class Elasticity_Contact_Evaluator>
class InstationaryElasticityContactAssembler : private AssemblyAssistant<DIMENSION, double> {
  // the InstationaryElasticityContactAssembler implements the penalty term accounting for "contact" 
  // in the weak formulation of the boundary value problem, i.e. it "punishes" by means of a penalty counter force 
  // any breaking of the contact conditions (i.e. there must always be a gap >= 0.0 between two bodies).
  // The referred struct computes the force_scaling_factor, depending on
  // - the current timestep's blood pressure (nBC), and
  // - the current size of the gap between the bodies.
  // The assembler multiplies with suitably chosen test functions and integrates over this force,
  // according to the weak formulation as in the "Signorini Problem".
  // 
  // NOTE: algorithms not yet published // TODO.
  // 
  // Parameters: t.b.d.
  
  public:
    InstationaryElasticityContactAssembler(const Elasticity_Contact_Evaluator* elast_contact_eval, int cbdy1, int cbdy2, 
					   double ts, double delta_t, MeshPtr mesh_ptr, MPI_Comm comm_, 
					   double ContactToleranceThreshold, std::map<int, double> contactDistance_lookup_table)
	: elast_contact_eval_(elast_contact_eval), cbdy1_(cbdy1), cbdy2_(cbdy2), ts_(ts), delta_t_(delta_t),
					    mesh_pointer_(mesh_ptr), communicator_(comm_), 
					    ContactToleranceThreshold_(ContactToleranceThreshold),
					    contactDistance_lookup_table_(contactDistance_lookup_table),
					    contact_facets_counter_(0), facet_counter_(0), counter_pseudo_contact_facets_(0) {}
    
    void operator()(const Element<double>& element, int facet_number, const Quadrature<double>& quadrature, LocalVector& lv) {
      
      // Procedure to get the facet entity
      mesh::IncidentEntityIterator facet = element.get_cell().begin_incident(DIMENSION-1);
      for (int i=0; i<facet_number; ++i, ++facet ) {} // this for-loop only performs the ++facet task until the correct (surface) facet is reached.
      
      facet_counter_++;
      
      // Check if it belongs to the boundary specified by the material number "bdy_"
      if ( facet->get_material_number() == cbdy1_ || facet->get_material_number() == cbdy2_ ) {
         // ATTENTION: be aware that, when being inside this if-loop, not all processes might be available anymore!!!
	 // this causes MPI problems, e.g., for geometric search, etc.
	 // therefore use map, and pre-searched facet-list.
	 // NOTE: algorithms not yet published // TODO.
	 
	 std::map< int, double >::iterator map_iter = contactDistance_lookup_table_.begin();
	 while ( map_iter != contactDistance_lookup_table_.end() ) {
	   
	   if ( map_iter->first == facet_counter_ ) {
	     //std::cout << ".h --> ControlOutput: Warning: The current element's facet of Leaflet A (or B) is very close (in an epsilon-environment) to a facet of Leaflet B (or A)." << std::endl;
	     counter_pseudo_contact_facets_++;
	     
	     // Initialize the quadrature for integration over the facet which is subject to Neumann BCs
	     AssemblyAssistant<DIMENSION, double>::initialize_for_facet(element, quadrature, facet_number);
	     
	     const int num_q = num_quadrature_points();
	     
	     double penalty_force_factor = -1.0; // negative, in order to have the penalty-force act into counter (anti-normal) direction;
	     double force_scaling_factor = 1.0; // scaling not yet accounted for (but instead by means of a factor in "assembly()" method).
	     
	     // Get force_scaling_factor for facet normal of facet with respective material number "cbdy_":
	     if ( facet->get_material_number() == cbdy1_ ) {
	       force_scaling_factor = elast_contact_eval_[0](cbdy1_);//, cbdy2_); // = (*elast_contact_eval_)(cbdy2_);
	     } else if ( facet->get_material_number() == cbdy2_ ) {
	       force_scaling_factor = elast_contact_eval_[0](cbdy2_);//, cbdy2_); // = (*elast_contact_eval_)(cbdy1_);
	     } else {
	       std::cout << "Error: Problem in ContactBC-assembler processing ContactBCs w.r.t. cbdy_[.] numbers. " << std::endl;
	       assert(0);
	     }
	     penalty_force_factor *= force_scaling_factor;
	     
	     // loop over quadrature points.
	     for (int q = 0; q < num_q; ++q) {
	       const double wq = w(q); // quadrature weight for q-th quadrature point on surface
	       const double dsurf = ds(q);
	       const Vec<DIMENSION, double> nq = n(q); // surface normal in q-th quadrature point on surface
	       
	       // implement the weak formulation of the elasticity boundary value problem in the instationary case:
	       // introduce penalty counter force...
	       // NOTE: algorithms not yet published // TODO.
	       // assemble \int_bdy{ cbc_surface_penalty_force * v }
	       for (int v_var = 0; v_var < DIMENSION; ++v_var) {
	         for (int i = 0; i < num_dofs(v_var); ++i) {
		   lv[dof_index(i, v_var)] +=  wq * penalty_force_factor * nq[v_var] * phi(i, q, v_var) * dsurf;
	         }
	       }
	       
	     } // end of loop over quadrature points of respective surface facet.
	     
	   } // end if (facet_number in map, i.e. ContactToleranceThreshold is reached).
	   
	   ++map_iter;
	   
	 } // end for map_iter while loop.
	 
      } // end if ( facet->get_material_number() == cbdy1_ || facet->get_material_number() == cbdy2_ ).
      
    }
    
 private:
    const Elasticity_Contact_Evaluator* elast_contact_eval_;
    const int cbdy1_, cbdy2_;
    double ts_, delta_t_;
    MeshPtr mesh_pointer_;
    MPI_Comm communicator_;
    double ContactToleranceThreshold_;
    int contact_facets_counter_;
    std::map<int, double> contactDistance_lookup_table_;
    int facet_counter_;
    int counter_pseudo_contact_facets_;
};

//////////////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------------------------------------------

// For "InstationaryLinElasticityAssembler" for LINEAR ELASTICITY. ///////////////
double InstationaryLinElasticityAssembler::get_Newmark_TimeIntegration_Constants(int NewmarkConstantNumber) {
  if (NewmarkConstantNumber == 0)
    return Nalpha0_;
  else if (NewmarkConstantNumber == 1)
    return Nalpha1_;
  else if (NewmarkConstantNumber == 2)
    return Nalpha2_;
  else if (NewmarkConstantNumber == 3)
    return Nalpha3_;
  else if (NewmarkConstantNumber == 4)
    return Nalpha4_;
  else if (NewmarkConstantNumber == 5)
    return Nalpha5_;
  else if (NewmarkConstantNumber == 6)
    return Nalpha6_;
  else if (NewmarkConstantNumber == 7)
    return Nalpha7_;
  else
    assert(0); //std::cout << "Something went wrong with getting the Newmark Time Integration Constants. \n";
}

void InstationaryLinElasticityAssembler::set_prev_solution(const CVector* sol_prev) {
    prev_ts_sol_ = sol_prev;
}

void InstationaryLinElasticityAssembler::set_dt_prev_solution(const CVector* dt_sol_prev) {
  dt_prev_ts_sol_ = dt_sol_prev;
}

void InstationaryLinElasticityAssembler::set_dt2_prev_solution(const CVector* dt2_sol_prev) {
  dt2_prev_ts_sol_ = dt2_sol_prev;
}

void InstationaryLinElasticityAssembler::set_timestep_parameters(double theta, double delta_t) {
    // Specify which time discretization method you choose
    // theta = 0: ExplicitEuler, theta=0.5: CrankNicolson, theta=1: ImplicitEuler
    
    // Theta-Family-Integration Constants:
    ThetaFamAlpha1_ = theta * delta_t;
    ThetaFamAlpha2_ = delta_t;
    ThetaFamAlpha3_ = (1. - theta) * delta_t;
    
    // Newmark parameters Nalpha_ and Ndelta_:
    Ndelta_ = 0.5;
    Nalpha_ = 0.25*(0.5+Ndelta_)*(0.5+Ndelta_);
    // Newmark Time Iteration Constants:
    Nalpha0_ = 1./(Nalpha_ * delta_t * delta_t);
    Nalpha1_ = Ndelta_/(Nalpha_ * delta_t);
    Nalpha2_ = 1./(Nalpha_ * delta_t);
    Nalpha3_ = 1./(2.*Nalpha_) - 1.;
    Nalpha4_ = Ndelta_/Nalpha_ - 1.;
    Nalpha5_ = 0.5 * delta_t * (Ndelta_/Nalpha_ - 2.);
    Nalpha6_ = delta_t * (1. - Ndelta_);
    Nalpha7_ = Ndelta_ * delta_t;
}

void InstationaryLinElasticityAssembler::set_time(double time) {
    time_ = time;
}

// -------------------------------------------------------------------

// For "InstationaryCorotElasticityAssembler" for COROTATIONAL ELASTICITY. ///////////////
double InstationaryCorotElasticityAssembler::get_Newmark_TimeIntegration_Constants(int NewmarkConstantNumber) {
  if (NewmarkConstantNumber == 0)
    return Nalpha0_;
  else if (NewmarkConstantNumber == 1)
    return Nalpha1_;
  else if (NewmarkConstantNumber == 2)
    return Nalpha2_;
  else if (NewmarkConstantNumber == 3)
    return Nalpha3_;
  else if (NewmarkConstantNumber == 4)
    return Nalpha4_;
  else if (NewmarkConstantNumber == 5)
    return Nalpha5_;
  else if (NewmarkConstantNumber == 6)
    return Nalpha6_;
  else if (NewmarkConstantNumber == 7)
    return Nalpha7_;
  else
    assert(0); //std::cout << "Something went wrong with getting the Newmark Time Integration Constants. \n";
}

void InstationaryCorotElasticityAssembler::set_prev_solution(const CVector* sol_prev) {
    prev_ts_sol_ = sol_prev;
}

void InstationaryCorotElasticityAssembler::set_dt_prev_solution(const CVector* dt_sol_prev) {
  dt_prev_ts_sol_ = dt_sol_prev;
}

void InstationaryCorotElasticityAssembler::set_dt2_prev_solution(const CVector* dt2_sol_prev) {
  dt2_prev_ts_sol_ = dt2_sol_prev;
}

void InstationaryCorotElasticityAssembler::set_timestep_parameters(double theta, double delta_t) {
    // Specify which time discretization method you choose
    // theta = 0: ExplicitEuler, theta=0.5: CrankNicolson, theta=1: ImplicitEuler
    
    // Theta-Family-Integration Constants:
    ThetaFamAlpha1_ = theta * delta_t;
    ThetaFamAlpha2_ = delta_t;
    ThetaFamAlpha3_ = (1. - theta) * delta_t;
    
    // Newmark parameters Nalpha_ and Ndelta_:
    Ndelta_ = 0.5;
    Nalpha_ = 0.25*(0.5+Ndelta_)*(0.5+Ndelta_);
    // Newmark Time Iteration Constants:
    Nalpha0_ = 1./(Nalpha_ * delta_t * delta_t);
    Nalpha1_ = Ndelta_/(Nalpha_ * delta_t);
    Nalpha2_ = 1./(Nalpha_ * delta_t);
    Nalpha3_ = 1./(2.*Nalpha_) - 1.;
    Nalpha4_ = Ndelta_/Nalpha_ - 1.;
    Nalpha5_ = 0.5 * delta_t * (Ndelta_/Nalpha_ - 2.);
    Nalpha6_ = delta_t * (1. - Ndelta_);
    Nalpha7_ = Ndelta_ * delta_t;
}

void InstationaryCorotElasticityAssembler::set_time(double time) {
    time_ = time;
}

// -------------------------------------------------------------------

#endif
