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

/// \author: Nicolai Schoch.

#include "elasticity.h"

#include <iomanip>
#include <iterator>
#include <algorithm>
#include <vector>

// NOTATION:
// TASK: has do be done at some point;
// TODO: has to be done urgently;
// NOTE: is there to remind of something.
// NOTE: naming conventions: class names "capital, no underscores"; class methods "small, with underscores"; member variables "with underscores at end of name";

/// NOTE: This code does not contain the entire set of functionalities for Mitral Valve Reconstruction Simulation.
///       Parts of the complete algorithm are not (yet) open-source, but available at Nicolai Schoch.

#define HEART_MV_SCENARIO // this only affects the scaling of the NeumannBCs (w.r.t. mmHg/Pa-unit-conversion for the heart scenario).

namespace {
    static const char* DATADIR = ""; //MESHES_DATADIR;
    static const int MASTER_RANK = 0;
    static const char* PARAM_FILENAME = "param.xml"; // is replaced if function main is called with argument "elasticity.xml".
    static bool CONSOLE_OUTPUT_ACTIVE = true;
    static const int CONSOLE_THRESHOLD_LEVEL = 3;
}
#define CONSOLE_OUTPUT(lvl, x) { if (CONSOLE_OUTPUT_ACTIVE &&            \
                                    lvl <= CONSOLE_THRESHOLD_LEVEL) {   \
            for (int i = 0; i < lvl; ++i) { std::cout << "  "; }        \
            std::cout << x << "\n"; }}

struct TimingData {
    double time_elapsed;
};

class TimingScope {
 public:
    TimingScope(const std::string& name) {
        if (report_) {
            report_->begin_section(name);
        }
    }

    TimingScope(int iteration) {
        if (report_) {
            std::stringstream sstr;
            sstr << "Iteration " << iteration;
            report_->begin_section(sstr.str());
            timer_.reset();
            timer_.start();
        }
    }

    ~TimingScope() {
        timer_.stop();
        if (report_) {
            TimingData* data = report_->end_section();
            data->time_elapsed = timer_.get_duration();
        }
    }

    static void set_report(HierarchicalReport<TimingData>* report) {
        report_ = report;
    }

 private:
    static HierarchicalReport<TimingData>* report_;
    Timer timer_;
};

HierarchicalReport<TimingData>* TimingScope::report_ = 0;

class TimingReportOutputVisitor {
 public:
    TimingReportOutputVisitor(std::ostream& os)
        : os_(os), level_(0) {
        }

    void enter(const std::string& name, TimingData* data) {
        if (name == "root") {
            os_ << "+++ Timing Report +++\n\n";
        } else {
            for (int l = 0; l < level_; ++l) {
                os_ << "  ";
            }
            os_ << name << " took " << data->time_elapsed << " s.\n";
            ++level_;
        }
    }

    void exit(const std::string& name, TimingData* data) {
        if (name == "root") {
            os_ << "\n+++ End Timing Report +++\n\n";
        } else {
            --level_;
        }
    }

 private:
    std::ostream& os_;
    int level_;
};

class Elasticity {
 public:
	Elasticity(const std::string& param_filename)
        : comm_(MPI_COMM_WORLD),
          params_(param_filename.c_str(), MASTER_RANK, MPI_COMM_WORLD),
          refinement_level_(0),
          is_done_(false)
        {}

    virtual ~Elasticity() {}

    virtual void run() {
        simul_name_ = params_["OutputPathAndPrefix"].get<std::string>();
	u_deg = params_["FiniteElements"]["DisplacementDegree"].get<int>();

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &num_partitions_);

        // Turn off INFO log except on master proc.
        if (rank_ != MASTER_RANK) {
            INFO = false;
            CONSOLE_OUTPUT_ACTIVE = false;
        }

        std::ofstream info_log((simul_name_ + "_info_log").c_str());
        LogKeeper::get_log("info").set_target(&info_log);
        std::ofstream debug_log((simul_name_ + "_debug_log").c_str());
        LogKeeper::get_log("debug").set_target(&debug_log);

        CONSOLE_OUTPUT(0, "");
	CONSOLE_OUTPUT(0, "=========================================================");
        CONSOLE_OUTPUT(0, "==== Elasticity Simulation                            ===");
        CONSOLE_OUTPUT(0, "====       built using HiFlow3.                       ===");
        CONSOLE_OUTPUT(0, "====                                                  ===");
        CONSOLE_OUTPUT(0, "==== Engineering Mathematics and Computing Lab (EMCL) ===");
        CONSOLE_OUTPUT(0, "=========================================================");
        CONSOLE_OUTPUT(0, "");

        // output parameters for debugging
        LOG_INFO("parameters", params_);
	
	// get command if total mesh volume shall be computed, or not:
	calcVolumeSwitch = params_["ElasticityModel"]["calcVolumeSwitch"].get<bool>();
	
        // setup timing report
        TimingScope::set_report(&time_report_);

        {
            TimingScope tscope("Setup");
	    
            setup_linear_algebra();
	    
            read_mesh();
	    
            // The simulation has two modes: stationary and
            // instationary. Which one is used, depends on the parameter
            // Instationary.SolveInstationary. In stationary mode, solve()
            // is only called once, whereas in instationary mode, it is
            // called several times via the time-stepping method implemented in run_time_loop() .
            solve_instationary_ = params_["Instationary"]["SolveInstationary"].get<bool>();

            prepare();
	    
	    if (calcVolumeSwitch) {
	      // Calculate (initial) volume before simulation
	      calc_volume();
	      if ( rank_ == MASTER_RANK ) {
		std::cout << "\n  Initial global mesh volume: " << initial_global_volume << std::endl << std::endl;
	      }
	      
	    }
        }

        if (solve_instationary_) {
	    TimingScope tscope("Complete_instationary_simulation_loop");
	    
            LOG_INFO("simulation", "Solving instationary problem");
	    
	    // Time discretization
	    ts_ = 0;
	    time_ = 0.0;
	    // Number of time steps to be calculated
	    Tmax_ = params_["Instationary"]["MaxTimeStepIts"].get<int>();
	    // Time step size
	    delta_t_ = params_["Instationary"]["DeltaT"].get<double>();
	    
	    dampingFactor_ =  params_["Instationary"]["DampingFactor"].get<double>();
	    rayleighAlpha_ =  params_["Instationary"]["RayleighAlpha"].get<double>(); // for Rayleigh-Scaling of Mass Matrix.
	    rayleighBeta_ =  params_["Instationary"]["RayleighBeta"].get<double>(); // for Rayleigh-Scaling of Stiffness Matrix.
	    
	    method = params_["Instationary"]["Method"].get<std::string>();
	    if (method == "CrankNicolson") {
	      theta_ = 0.5;
	      assert(0); // not_yet_implemented. // TASK.
	    } else if (method == "ImplicitEuler") {
		  theta_ = 1.;
		  assert(0); // not_yet_implemented. // TASK.
		  } else if (method == "ExplicitEuler") {
			 theta_ = 0.;
			 assert(0); // not_yet_implemented. // TASK.
			 } else if (method == "Newmark") {
			        CONSOLE_OUTPUT(0, "Starting simulation using the Newmark time integration scheme.\n");
			        // anything to be initialized?!
				} else {
				  std::cout  << "=================================\n"
				  << "Unknown method for solving instationary problem\n"
				  << "=================================\n"
				  << "Default method: Newmark.\n";
				}
	    
// 	    // set the alpha coefficients correctly for the Crank-Nicolson method.
// 	    alpha1_ = 0.5 * delta_t_;
// 	    alpha2_ = delta_t_;
// 	    alpha3_ = 0.5 * delta_t_;
	    
	    // get command if displacement Dirichlet BCs shall be released after 1.0s again, or not:
	    dispDirBCsReleaseValue = params_["Mesh"]["dispDirBCsRelease"].get<bool>();
	    
	    // get command if contact is possible and shall be considered in simulation, or not:
	    chanceOfContactValue = params_["Mesh"]["chanceOfContact"].get<bool>();
	    
	    // get command if "corotational formulation" shall be used instead of "linear elasticity formulation":
	    corotFormValue = params_["Mesh"]["corotForm"].get<bool>();
	    
	    visPerXtimesteps = params_["Instationary"]["VisPerXTs"].get<int>();
	    extVis = params_["ExtendedOutputVisualization"].get<bool>();
	    
	    // Loop over all time steps
	    for (ts_ = 0; ts_ <= Tmax_; ++ts_) {
	      if (ts_ > 0) {
		
		if (chanceOfContactValue) {
		  // Find potential contact/intersection elements in current time step
		  std::cout << "\n ATTENTION: Contact Search DEPRECATED in this version. Check back with Nicolai Schoch for newer implementations. \n" << std::endl;
		  find_potential_contact_elements();
		}
		
    		// Compute the stiffness matrix and right-hand side
    		assemble_system();
		
    		// Solve the linear system
    		solve_system();
		
		if (calcVolumeSwitch) {
		  // Calculate (current timestep's) volume
		  calc_def_volume();
		  if ( rank_ == MASTER_RANK ) {
		    std::cout << "\n  Global mesh volume of current timestep " << ts_ << " : " << current_timesteps_global_volume << std::endl << std::endl;
		  }
		}
		
	      }
	      
	      // Visualize the solution and the errors.
	      if (ts_ % visPerXtimesteps == 0) {
		visualize(/*ts_*/);
	      }
	      time_ += delta_t_;
	    }
	    
        } else {
	    TimingScope tscope("Complete_stationary_simulation_loop");
	    
            LOG_INFO("simulation", "Solving stationary problem");
	    
	    assemble_system();
	    
	    solve_system();
	    
            visualize();
        }

        CONSOLE_OUTPUT(0, "");

        if (rank_ == MASTER_RANK) {
            // Output time report
            TimingReportOutputVisitor visitor(std::cout);
            time_report_.traverse_depth_first(visitor);
        }

        LogKeeper::get_log("info").flush();
        LogKeeper::get_log("debug").flush();
        LogKeeper::get_log("info").set_target(0);
        LogKeeper::get_log("debug").set_target(0);

        CONSOLE_OUTPUT(0, "============================================================\n");
    }

 private:
    const MPI_Comm& communicator() { return comm_; }
    int rank() { return rank_; }
    int num_partitions() { return num_partitions_; }

    PLATFORM la_platform() const { return la_sys_.Platform; }
    IMPLEMENTATION la_implementation() const { return la_impl_; }
    MATRIX_FORMAT la_matrix_format() const { return la_matrix_format_; }

    // Read, refine and partition mesh.
    void read_mesh();

    // Set up datastructures and read in some parameters.
    void prepare();

    // Set up boundary conditions
    void prepare_bc_stationary();
    void prepare_bc_instationary();
    void process_BCdataInputfile_into_BCdataStructure();
    
    // Find potential contact elements
    void find_potential_contact_elements();
    
    // Assemble system (includes solve_system() for the instationary case).
    void assemble_system();
    
    // Solve System (excludes solver procedure for the instationary case).
    void solve_system();
    
    // Calculate total mesh volume:
    void calc_volume(); // for the initial mesh (w.r.t. timestep 0)
    void calc_def_volume(); // for the deformed mesh (w.r.t. current timestep)

    // Visualize the solution in a file.
    // In stationary mode, the filename contains 'stationary',
    // in instationary mode, it contains the current time-step ts_.
    void visualize(/*ts_*/);
    
    // Helper functions for instationary computation
    void compute_dt2_sol(double Nalpha0, double Nalpha2, double Nalpha3, 
			 const CVector& solution_, const CVector& solution_prev_, const CVector& dt_solution_prev_, const CVector& dt2_solution_prev_, 
			 CVector* dt2_solution_);
    void compute_dt_sol(double Nalpha6, double Nalpha7, 
			const CVector& dt_solution_prev_, const CVector& dt2_solution_prev_, const CVector& dt2_solution_, 
			CVector* dt_solution_);

//     // Helper functions for solver in case of non-linear elasticity (Newton steps):
//     // NOTE: parallel algorithms not yet published // TODO.
//     void compute_RHS(const LAD::VectorType& in, LAD::VectorType* out);
//     void compute_stationary_RHS(const LAD::VectorType& in, LAD::VectorType* out);
//     void compute_instationary_RHS(const LAD::VectorType& in, LAD::VectorType* out);
// 
//     void compute_matrix(const LAD::VectorType& in, LAD::MatrixType* out);
//     void compute_stationary_matrix(const LAD::VectorType& in, LAD::MatrixType* out);
//     void compute_instationary_matrix(const LAD::VectorType& in, LAD::MatrixType* out);

    // Linear algebra set up
    void setup_linear_algebra();

    // MPI stuff
    MPI_Comm comm_;
    int rank_, num_partitions_;

    // Linear algebra stuff
    SYSTEM la_sys_;
    IMPLEMENTATION la_impl_;
    MATRIX_FORMAT la_matrix_format_;

    // Parameter data read in from file.
    PropertyTree params_;

    std::string simul_name_; // parameter 'OutputPathAndPrefix': prefix for output files
    int u_deg;  // Finite Element Degree (of variable u)

    // Time-stepping variables
    double time_; // actual simulation time.
    int ts_; // actual time step (used for visualization name-definition)
    int Tmax_; // number of time iterations.
    double delta_t_;  // size of time step.
    double theta_; // parameter for specifying Theta-Family Time Integration methods.
    int visPerXtimesteps; // visualize sim results every 1 in X timesteps.
    bool extVis; // boolean variable to declare if extended visualization (in order for von-Mises-stress computation) is performed and written out, or not.
    std::string method;  // Defines which method is used for time discretization.
    double Ndelta_, Nalpha_; // Newmark parameters.
    double Nalpha0_, Nalpha1_, Nalpha2_, Nalpha3_, Nalpha4_, Nalpha5_, Nalpha6_, Nalpha7_; // Newmark integration constants.
    double alpha1_, alpha2_, alpha3_; // parameters of Crank-Nicolson method.
    double dampingFactor_, rayleighAlpha_, rayleighBeta_; // Damping factor and Rayleigh Damping constants.
    bool dispDirBCsReleaseValue; // boolean variable to declare if displacement Dirichlet BCs shall be released after 1.0s again, or not.
    bool corotFormValue; // boolean variable to declare if corotational or linear formulation shall be used.
    
    // other variables (w.r.t. contact simulation, volume computation, ...)
    bool chanceOfContactValue; // boolean variable to declare if contact is possible or not; by default without contact.
    bool calcVolumeSwitch; // boolean variable to declare if volume shall be computed or not.
    double initial_global_volume; // initial mesh volume.
    double current_timesteps_global_volume; // current timestep's mesh volume.
    
    // Elasticity model variables
    double lambda_, mu_, rho_, gravity_;
    double n_bdy1_force_in_mmHg_, n_bdy1_force_in_Pascal_, n_bdy2_force_in_mmHg_, n_bdy2_force_in_Pascal_;
    
    // Meshes
    MeshPtr mesh_;
    MeshPtr master_mesh_copy;
    int refinement_level_;

    VectorSpace<double> space_;

    // linear algebra objects
    Couplings<double> couplings_;
    CMatrix matrix_; // Moreover: stiffM_, dampM_, massM_, iterM_;
    CVector sol_, rhs_, nbc_, cbc_; // solution-vector u, rhs-vector f, nbc-vector s, cbc-vector c;
    CVector sol_prev_, dt_sol_prev_, dt2_sol_prev_; // solution-vector u(t-1), dt_solution-vector u(t-1), dt2_solution-vector u(t-1);
    CVector dt2_sol_, dt_sol_; // helper vectors for instationary simulation;

    // linear solver parameters
    std::string solver_name_;
    int lin_max_iter;
    double lin_abs_tol;
    double lin_rel_tol;
    double lin_div_tol;
    int basis_size;
    
    // preconditioner parameters
    MATRIX_FREE_PRECOND matrix_precond_; // preconditioner name
    double omega_; // for SOR and SSOR
    int ilu_p_; // for ILUp
    
    // preconditioner
    PreconditionerBlockJacobiStand<LAD> preconditioner_; // Base class for all block Jacobi preconditioners
    // PreconditionerBlockJacobi: base class for all block Jacobi preconditioners (Jacobi, GaussSeidel, S-GaussSeidel, SOR, SSOR, ...).
#ifdef WITH_ILUPP
    PreconditionerIlupp<LAD> ilupp_;
#endif
    
    // linear solver
    // instead of putting up two solvers, alternatively put up an abstract LinearSolver-Object which is filled during runtime via XML-Input.
    GMRES<LAD> gmres_; // GMRES works best with ILUpp.
    CG<LAD> cg_; // CG works best with BlockJacobi and SymmetricGaussSeidel
    
    StandardGlobalAssembler<double> global_asm_;
    
    bool is_done_, solve_instationary_, use_ilupp_; //use_precond_;
    
    // Dirichlet BC vectors (DoFs and Values).
    std::vector<int> dirichlet_dofs_;
    std::vector<Scalar> dirichlet_values_;
    std::vector<int> fixed_dirichlet_dofs_;
    std::vector<Scalar> fixed_dirichlet_values_;
    std::vector<int> displacement_dirichlet_dofs_;
    std::vector<Scalar> displacement_dirichlet_values_;
    
    // Visualization vector.
    std::vector<std::string> visu_names_;
    
    HierarchicalReport<TimingData> time_report_;
};

// ------------------------------------------------------------------

// program entry point
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    std::string param_filename(PARAM_FILENAME);
    if (argc > 1) {
        param_filename = std::string(argv[1]);
    }

    try {
        Elasticity app(param_filename);
        app.run();
    } catch(std::exception& e) {
        std::cerr << "\nProgram ended with uncaught exception.\n";
        std::cerr << e.what() << "\n";
        return -1;
    }
    MPI_Finalize();

    return 0;
}

// ------------------------------------------------------------------

void Elasticity::read_mesh() {
    TimingScope tscope("read_mesh");
    
    MeshPtr master_mesh;
    MeshPtr local_mesh;
    
    // get mesh_filename:
    const std::string mesh_name = params_["Mesh"]["Filename"].get<std::string>();
    std::string mesh_filename = std::string(DATADIR) + mesh_name;
    
    // find position of last '.' in mesh_filename:
    int dot_pos = mesh_filename.find_last_of(".");
    assert(dot_pos != std::string::npos);
    
    // get suffix of mesh_filename:
    std::string suffix = mesh_filename.substr(dot_pos);
    
    // switch cases according to the suffix of mesh_filename --> pvtu or vtu/inp:
    // if ( suffix == ".pvtu" )
    if ( suffix == std::string(".pvtu") ) {
      // make sure that np = X fits the number of partitions in the pvtu-file: // TODO
      //assert(num_partitions_ != number_of_vtuFiles_linked_in_pvtuFile); // TODO
      
      std::cout << "WARNING: Be aware, that pvtu-Inputfiles with np>1 might not be processed correctly!" << std::endl; // TODO CHECK!
      std::cout << "WARNING: Errors do occur as soon as the pvtu-file has not been built using 'utils/partition_mesh.cc'." << std::endl;
      
      // assign local mesh to the according vtu-files that are linked in the pvtu-file:
      local_mesh = read_mesh_from_file(mesh_filename, DIMENSION, DIMENSION, &comm_);
      
      // in case the input mesh does not specify material_IDs for the boundary facets,
      // the function set_default_material_number_on_bdy(MeshPtr*, int) sets default bdy_facet_IDs
      // in order for (among others) the GridGeometricSearch to work properly
      // especially w.r.t. find_closest_point() -> see also: documentation.
      set_default_material_number_on_bdy(local_mesh, 1);
    }
    // else if ( suffix == ".vtu" or ".inp" )
    else {
      
      if ( rank() == MASTER_RANK ) {
	
	master_mesh = read_mesh_from_file(mesh_filename, DIMENSION, DIMENSION, &comm_);
	CONSOLE_OUTPUT(1, "Read mesh with " << master_mesh->num_entities(DIMENSION) << " cells.");
	
	if ( suffix == std::string(".vtu") ) {
	  // in case the input mesh does not specify material_IDs for the boundary facets,
 	  // the function set_default_material_number_on_bdy(MeshPtr*, int) sets default bdy_facet_IDs
 	  // in order for (among others) the GridGeometricSearch to work properly
 	  // especially w.r.t. find_closest_point() -> see also: documentation.
	  set_default_material_number_on_bdy(master_mesh, 1);
	}
	refinement_level_ = 0;
        const int initial_ref_lvl = params_["Mesh"]["InitialRefLevel"].get<int>();

        for (int r = 0; r < initial_ref_lvl; ++r) {
            master_mesh = master_mesh->refine();
            ++refinement_level_;
        }
        LOG_INFO("mesh", "Initial refinement level = " << refinement_level_);
        CONSOLE_OUTPUT(1, "Refined mesh (level " << refinement_level_ << ") has "
                       <<  master_mesh->num_entities(DIMENSION) << " cells.");
	
      }
      
//       // DEPRECATED:
// #ifdef WITH_METIS
//       MetisGraphPartitioner partitioner;
// #else
//       NaiveGraphPartitioner partitioner;
// #endif
//       
//       const GraphPartitioner* p = (num_partitions() == 1 ? 0 : &partitioner);
      
      local_mesh = partition_and_distribute(master_mesh, MASTER_RANK, comm_); //, p);
      assert(local_mesh != 0);
      
    }
    
    SharedVertexTable shared_verts;
    mesh_ = compute_ghost_cells(*local_mesh, comm_, shared_verts);
    // NOTE: For pvtu-Inputfiles, this somehow destroys the (parallel) mesh...?! // TODO: check this!
    
    master_mesh_copy = compute_ghost_cells(*local_mesh, comm_, shared_verts); // TODO: check this!
    
    std::ostringstream rank_str;
    rank_str << rank();
    
    PVtkWriter writer(comm_);
    std::string output_file = simul_name_ + std::string("_initial_mesh_local.pvtu");
    writer.add_all_attributes(*mesh_, true);
    writer.write(output_file.c_str(), *mesh_);
}

void Elasticity::prepare() {
    TimingScope tscope("prepare");
    
    // prepare modelling problem parameters
    rho_ = params_["ElasticityModel"]["density"].get<double>();
    mu_  = params_["ElasticityModel"]["mu"].get<double>();
    lambda_  = params_["ElasticityModel"]["lambda"].get<double>();
    gravity_  = params_["ElasticityModel"]["gravity"].get<double>();
    
    // prepare space
    std::vector< int > degrees(DIMENSION);
    //const int u_deg = params_["FiniteElements"]["DisplacementDegree"].get<int>();
    for (int c = 0; c < DIMENSION; ++c) {
        degrees.at(c) = u_deg;
    }
    
    // Initialize the VectorSpace object
    space_.Init(degrees, *mesh_);

    CONSOLE_OUTPUT(1, "Total number of dofs = " << space_.dof().ndofs_global());

    for (int p = 0; p < num_partitions(); ++p) {
        CONSOLE_OUTPUT(2, "Num dofs on process " << p << " = " << space_.dof().ndofs_on_sd(p));
    }
    
    // prepare visualization structures
    visu_names_.push_back("u1");
    visu_names_.push_back("u2");
    visu_names_.push_back("u3");
    
    // Setup couplings object and prepare linear algebra structures
    couplings_.Clear();
    couplings_.Init(communicator(), space_.dof());
    
    // prepare global assembler
    QuadratureSelection q_sel(params_["QuadratureOrder"].get<int>());
    global_asm_.set_quadrature_selection_function(q_sel);
    
    // compute matrix graph
    SparsityStructure sparsity;
    global_asm_.compute_sparsity_structure(space_, sparsity);

    couplings_.InitializeCouplings(sparsity.off_diagonal_rows, sparsity.off_diagonal_cols);
    
    // Initialize system matrix (K_eff), solution vector and rhs vector (R_eff) [and possibly nbc vector].
    matrix_.Init(communicator(), couplings_, la_platform(), la_implementation(), la_matrix_format()); // System (Stiffness) Matrix.
    sol_.Init(communicator(), couplings_, la_platform(), la_implementation()); // Solution Vector.
    rhs_.Init(communicator(), couplings_, la_platform(), la_implementation()); // RHS Vector.
    nbc_.Init(communicator(), couplings_, la_platform(), la_implementation()); // NeumannBC Vector.
    cbc_.Init(communicator(), couplings_, la_platform(), la_implementation()); // ContactBC Vector.

    matrix_.InitStructure(vec2ptr(sparsity.diagonal_rows),
                          vec2ptr(sparsity.diagonal_cols),
                          sparsity.diagonal_rows.size(),
                          vec2ptr(sparsity.off_diagonal_rows),
                          vec2ptr(sparsity.off_diagonal_cols),
                          sparsity.off_diagonal_rows.size());
    matrix_.Zeros();

    sol_.InitStructure();
    sol_.Zeros();

    rhs_.InitStructure();
    rhs_.Zeros();
    
    nbc_.InitStructure();
    nbc_.Zeros();
    
    cbc_.InitStructure();
    cbc_.Zeros();
    
    // For instationary simulation:
    // Initialize (helper) vectors dt_sol_(t) and dt2_sol_(t)
    dt_sol_.Init(communicator(), couplings_, la_platform(), la_implementation());
    dt2_sol_.Init(communicator(), couplings_, la_platform(), la_implementation());
    
    // Initialize solution-vector u(t-1), dt_solution-vector u(t-1), dt2_solution-vector u(t-1).
    sol_prev_.Init(communicator(), couplings_, la_platform(), la_implementation()); // solution-vector u(t-1)
    dt_sol_prev_.Init(communicator(), couplings_, la_platform(), la_implementation()); // dt_solution-vector u(t-1)
    dt2_sol_prev_.Init(communicator(), couplings_, la_platform(), la_implementation()); // dt2_solution-vector u(t-1)
    
    dt_sol_.InitStructure(); // helper vector for instationary simulation
    dt_sol_.Zeros();
    
    dt2_sol_.InitStructure(); // helper vector for instationary simulation
    dt2_sol_.Zeros();
    
    sol_prev_.InitStructure(); // previous solution vector for instationary simulation
    sol_prev_.Zeros();
    
    dt_sol_prev_.InitStructure(); // previous dt_solution vector for instationary simulation
    dt_sol_prev_.Zeros();
    
    dt2_sol_prev_.InitStructure(); // previous dt2_solution vector for instationary simulation
    dt2_sol_prev_.Zeros();
    
    // setup linear solver parameters
    solver_name_ = params_["LinearSolver"]["SolverName"].get<std::string>();
    lin_max_iter = params_["LinearSolver"]["MaximumIterations"].get<int>();
    lin_abs_tol = params_["LinearSolver"]["AbsoluteTolerance"].get<double>();
    lin_rel_tol = params_["LinearSolver"]["RelativeTolerance"].get<double>();
    lin_div_tol = params_["LinearSolver"]["DivergenceLimit"].get<double>();
    basis_size = params_["LinearSolver"]["BasisSize"].get<int>();
    
    // prepare preconditioner
    //use_precond_ = params_["LinearSolver"]["Preconditioning"].get<bool>(); // not needed, since setup_linear_algebra() reads "precond_str" from the parameter xml-file.
    
    // Setup Solver and Preconditioner //////////////////////////////////////////////////////////
    if (solver_name_ == "CG") {
      
      // Setup CG Preconditioner //////////////////////////////////////////
      if (matrix_precond_ != NOPRECOND) { // With Preconditioning.
      // if (use_precond_) ...
	  
	  if(matrix_precond_ == JACOBI) {
	    preconditioner_.Init_Jacobi(rhs_);
	  }
	  if(matrix_precond_ == GAUSS_SEIDEL) {
	    preconditioner_.Init_GaussSeidel();
	  }
	  if(matrix_precond_ == SGAUSS_SEIDEL) {
	    preconditioner_.Init_SymmetricGaussSeidel();
	  }
	  if(matrix_precond_ == SOR) {
	    omega_ = params_["LinearSolver"]["Omega"].get<double>();
	    preconditioner_.Init_SOR(omega_);
	  }
	  if(matrix_precond_ == SSOR) {
	    omega_ = params_["LinearSolver"]["Omega"].get<double>();
	    preconditioner_.Init_SSOR(omega_);
	  }
	  if(matrix_precond_ == ILU) {
	    // TASK.
	    std::cout << "By definition: CG is not supposed to pair with ILU as solver-preconditioner-combination." << std::endl;
	  }
	  if(matrix_precond_ == ILU2) { // Note that: ILU2 = ILUpp; handled for GMRES only.
	    // TASK.
 	    // ilu_p_ = params_["LinearAlgebra"]["ILU_p"].get<int>();
 	    // preconditioner_.Init_ILUp(ilu_p_);
	    std::cout << "By definition: CG is not supposed to pair with ILU2 (a.k.a. ILUpp) as solver-preconditioner-combination." << std::endl;
	  }
// 	  if(matrix_precond_ == ILU_p) { // ILU_p not defined here. -> TASK: improve code w.r.t. solver/preconditioner factories.
// 	    ilu_p_ = params_["LinearAlgebra"]["ILU_p"].get<int>();
// 	    preconditioner_.Init_ILUp(ilu_p_);
// 	  }
	  
	  cg_.InitParameter("Preconditioning");
	  cg_.SetupPreconditioner(preconditioner_);
	  
      } else { // Without Preconditioning.
	  
	  cg_.InitParameter("NoPreconditioning");
	  
      }
      
      // Setup CG Solver ////////////////////////////////////////////////
      cg_.InitControl(lin_max_iter, lin_abs_tol, lin_rel_tol, lin_div_tol);
      cg_.SetupOperator(matrix_);
      
    } // end of CG Setup.
    else /*if (solver_name_ == "GMRES")*/ { // if not CG then GMRES by default.
      
      // Setup GMRES Preconditioner //////////////////////////////////////////
      use_ilupp_ = 0; // default.
      if (matrix_precond_ == ILU2) { // Note that: ILU2 = ILU_pp.
	  use_ilupp_ = 1; // otherwise: no preconditioner for GMRES.
      }
#ifdef WITH_ILUPP
      if (use_ilupp_) {
	  ilupp_.InitParameter(params_["ILUPP"]["PreprocessingType"].get<int>(),
                             params_["ILUPP"]["PreconditionerNumber"].get<int>(),
                             params_["ILUPP"]["MaxMultilevels"].get<int>(),
                             params_["ILUPP"]["MemFactor"].get<double>(),
                             params_["ILUPP"]["PivotThreshold"].get<double>(),
                             params_["ILUPP"]["MinPivot"].get<double>());
	  
	  gmres_.SetupPreconditioner(ilupp_);
	  gmres_.InitParameter(basis_size, "RightPreconditioning");
      } else {
	  gmres_.InitParameter(basis_size, "NoPreconditioning");
      }
#else
      gmres_.InitParameter(basis_size, "NoPreconditioning");
#endif
      
      // Setup GMRES Solver ////////////////////////////////////////////////
      gmres_.InitControl(lin_max_iter, lin_abs_tol, lin_rel_tol, lin_div_tol);
      gmres_.SetupOperator(matrix_); // setup only w.r.t. the above ControlParameters.
      // yet, the setup w.r.t. the matrix components and possibly w.r.t. Dirichlet-BC-Setup 
      // is to be done again after the assembly of Dirichlet-BCs.
      
    } // end of GMRES Setup.
    
    // prepare dirichlet BC
    if (!solve_instationary_) {
      prepare_bc_stationary();
    } /*else {
      prepare_bc_instationary();
      // Note: This is done below in assemble_system() in the timestepping loop (once in every timestep).
    }*/
}

void Elasticity::prepare_bc_stationary() {
    TimingScope tscope("prepare_bc_stationary");
    
    dirichlet_dofs_.clear();
    dirichlet_values_.clear();
    
    // >-----------------------------------------------------------------------------------------------------
    // Set DirichletBCs for prescribed facets (i.e. material_IDs given in hiflow3_scene.xml-File): // DEPRECATED, but still working.
    const int dir_bdy1 = params_["Boundary"]["DirichletMaterial1"].get<int>();
    const int dir_bdy2 = params_["Boundary"]["DirichletMaterial2"].get<int>();
    const int dir_bdy3 = params_["Boundary"]["DirichletMaterial3"].get<int>();
    // NOTE: DirichletBCs by means of prescribed facets are deprecated, and replaced by means of pointwise DirichletBCs, see below.
    
    // create InstationaryElasticity_DirichletBC_3D-Object.
    StationaryElasticity_DirichletBC_3D bc[3] = { StationaryElasticity_DirichletBC_3D(0, dir_bdy1, dir_bdy2, dir_bdy3),
    						   StationaryElasticity_DirichletBC_3D(1, dir_bdy1, dir_bdy2, dir_bdy3),
    						   StationaryElasticity_DirichletBC_3D(2, dir_bdy1, dir_bdy2, dir_bdy3)};
    
    // and compute Dirichlet values for pre-set dofs.
    for (int var = 0; var < DIMENSION; ++var) {
        compute_dirichlet_dofs_and_values(bc[var], space_, var,
                                          dirichlet_dofs_, dirichlet_values_);
    }
    
    // >-----------------------------------------------------------------------------------------------------
    // Set pointwise BCs as given in BCdata-Input-File:
    process_BCdataInputfile_into_BCdataStructure();
    // >-----------------------------------------------------------------------------------------------------
}

void Elasticity::process_BCdataInputfile_into_BCdataStructure() {
    TimingScope tscope("process_BCdataInputfile_into_BCdataStructure");
    // doing a geometric search, this function processes the BCdata-Inputfile into the corresponding BCdata-structures
    // (namely DirichletBC structures) which can be handled by the below "assembly()" and "solving()" algorithms.
    
    // >-----------------------------------------------------------------------------------------------------
    // TODO: Include some assert()-tests in order to verify the function works properly.
    
    // In initial time step only, do:
    // Read BCdata-File: >--------------------------------------------------------
    if ( !solve_instationary_ || time_ == delta_t_ ) {
      
      const std::string bc_file_name = params_["Mesh"]["BCdataFilename"].get<std::string>();
      std::string bcdata_filename_ = std::string(DATADIR) + bc_file_name;
      
      PropertyTree bcdata(bcdata_filename_.c_str(), MASTER_RANK, MPI_COMM_WORLD);
      // TODO: Problems may occur when processing input XML file! -> Therefore: ToDo: Parsing Check!
      // e.g.: check for too many / too few components in string, check for wrong separation sign
      // (so far only: "," and ";" but what if blank space, tab, newline, etc.?!)
      // e.g.: by means of "if (string != 0...9 or != , or != ;) { assert(false); assert("Error when parsing BCdata.") }
      // Note: the string might need to be wrapped in "(...)" in order to get one whole line in the xml-file?!
      
      // Initialize GridGeometricSearch:
      //  in order to for a given (BCdata-)point find the cell and its localCellIndex and its globalCellID.
      GridGeometricSearch geom_search(mesh_);
      
      // >-----------------------------------------------------------------------------------------------------
      // Set DirichletBCs for prescribed pointCoords (i.e. given in hiflow3_BCdata.xml-File):
      
      // BEGINNING OF fdPoints: ----------------------------------------
      std::vector<double> curr_BCpoint(DIMENSION, 0.0);
      
      // First: DirichletBC-parts: FixedDirichlet:
      // > Get Fixed Zero Dirichlet Points: --------------------------------------------------
      int numFDpoints = bcdata["BCData"]["FixedConstraintsBCs"]["NumberOfFixedDirichletPoints"].get<int>();
      
      if (numFDpoints > 0) {
	std::vector<double> fdPoints(DIMENSION * numFDpoints, 0.0);
	std::string fdPoints_str = bcdata["BCData"]["FixedConstraintsBCs"]["fDPoints"].get<std::string>();
	
	int indexx = 0;
	for(int i = 0; i < numFDpoints; i++) {
	    //std::cout << "TestOutput: Processing FDpoint " << i << std::endl;
	    for(int j = 0; j < DIMENSION; j++) {
		std::string curr_value = ""; // Alternative: std::ostringstream curr_value;
		bool value_finished = false;
		while(!value_finished && (indexx < fdPoints_str.length())) {
		  // check string components for validity (i.e. numbers, +/-, points, kommas, semicolons):
		  if ( fdPoints_str[indexx] != '0' && fdPoints_str[indexx] != '1' && fdPoints_str[indexx] != '2' && 
			fdPoints_str[indexx] != '3' && fdPoints_str[indexx] != '4' && fdPoints_str[indexx] != '5' && 
			fdPoints_str[indexx] != '6' && fdPoints_str[indexx] != '7' && fdPoints_str[indexx] != '8' && 
			fdPoints_str[indexx] != '9' && fdPoints_str[indexx] != ',' && fdPoints_str[indexx] != ';' && 
			fdPoints_str[indexx] != '-' && fdPoints_str[indexx] != '.' && fdPoints_str[indexx] != '+' ) {
		    std::cout << "Error when parsing BCdata-strings. --> string contains char which is not in {'0'...'9', '+', '-', ',', ';', '.'}" << std::endl;
		    assert(0);
		  }
		  // if string component is not a comma or semicolon, add it to string of curr_value:
		  if( (fdPoints_str[indexx] != ',') && (fdPoints_str[indexx] != ';') ) {
		    curr_value += fdPoints_str[indexx]; // Alternative: curr_value << fdPoints_str[indexx];
		  } else {
		    value_finished = true;
		  }
		  ++indexx;
		}
		// convert string of curr_value into double value and store it into fdPoints[i*DIMENSION + j]
		double temp_fdPoint_value = atof( curr_value.c_str() );
		// function "atof()": double atof (const char* str) parses a string, interprets as double and returns its value.
		fdPoints[i*DIMENSION + j] = temp_fdPoint_value;
		//std::cout << "TestOutput: fdPoints: (" << i << ", " << j << ") = " << fdPoints[i*DIMENSION + j] << std::endl;
	    }
	}
	
	// Get Fixed Zero Dirichlet Displacement Vectors: --------------------------------------------------
	std::vector<double> fDisplacements(DIMENSION * numFDpoints, 0.0);
	
	// Now, process the above read DirichletBCs: -----------------------------------
	for (int BCpointCounter = 0; BCpointCounter < numFDpoints; ++BCpointCounter) {
	  
	  curr_BCpoint[0] = fdPoints[BCpointCounter*DIMENSION + 0];
	  curr_BCpoint[1] = fdPoints[BCpointCounter*DIMENSION + 1];
	  curr_BCpoint[2] = fdPoints[BCpointCounter*DIMENSION + 2];
	  
	  // If curr_BCpoint is not in/on mesh, find_closest_point on boundary of mesh and the appendant cell's facet_ID:
	  int boundary_facet_index;
	  std::vector<double> projected_curr_BCpoint = geom_search.find_closest_point_parallel(curr_BCpoint, boundary_facet_index, comm_);
	  // Note: find_closest_point() works only if the input mesh (which was given to "read_mesh()")
	  // specified material_IDs for the boundary facets other than -1.
	  // Since this is w.log. not the case, "set_default_material_number_on_bdy()" sets default bdy_facet_IDs in "read_mesh()".
	  // Note: "boundary_facet_index" is available on one process only! all other processes have boundary_facet_index = -1.
	  
	  // extract boundary of mesh
	  MeshPtr boundary_mesh_pointer = geom_search.get_boundary_mesh();
	  
	  // get mesh_facet_index:
	  int mesh_facet_index;
	  if (boundary_facet_index != -1) { // Note: "boundary_facet_index" is available on one process only.
	    boundary_mesh_pointer->get_attribute_value("_mesh_facet_index_", DIMENSION-1, boundary_facet_index, &mesh_facet_index);
	    
	    // get_entity(DIM-1)
	    Entity mesh_facet = mesh_->get_entity(DIMENSION-1, mesh_facet_index);
	    assert(mesh_facet.num_incident_entities(DIMENSION) == 1);
	    
	    // get pointer on cell:
	    IncidentEntityIterator cell = mesh_facet.begin_incident(DIMENSION);
	    
	    // get number of facet with respect to cell
	    int facet_number = 0;
	    for (IncidentEntityIterator inc_facet = cell->begin_incident(DIMENSION-1); mesh_facet_index != inc_facet->index();
		    ++inc_facet){
		++facet_number;
	    }
	    
	    const DofPartition<double>& dof = space_.dof();
	    
	    for (int var = 0; var < DIMENSION; ++var) {
		std::vector<int> dofs_on_facet;
		dof.get_dofs_on_subentity(var, cell->index(), DIMENSION-1, facet_number, dofs_on_facet);
		// iterate dofs
		for (std::vector<int>::iterator facet_dof_it = dofs_on_facet.begin(); 
		      facet_dof_it != dofs_on_facet.end(); ++facet_dof_it) {
		  if (dof.owner_of_dof(*facet_dof_it) == rank()) {
		    fixed_dirichlet_dofs_.push_back(*facet_dof_it);
		    fixed_dirichlet_values_.push_back(fDisplacements.at(BCpointCounter*DIMENSION + var));
		  }
		}
	    }
	  }
	}
	
      }
      // END OF fdPoints: ----------------------------------------
      
      // Second: DirichletBC-parts: DisplacementDirichlet:
      // --> compute displacement_dirichlet_dofs_ and final-displacement_dirichlet_values_ here;
      // --> the scaling with ts_ * delta_t_ takes place later.
      
      // get command if displacement Dirichlet BCs shall be released after 1.0s again, or not:
      // bool dispDirBCsReleaseValue = params_["Mesh"]["dispDirBCsRelease"].get<bool>(); // This is done in the very beginning.
      
      // > Get Displaced Dirichlet Points: --------------------------------------------------
      int numDDpoints = bcdata["BCData"]["DisplacementConstraintsBCs"]["NumberOfDisplacedDirichletPoints"].get<int>();
      
      if (numDDpoints > 0) {
	//std::cout << "TestOutput: numDDpoints = " << numDDpoints << std::endl;
	std::vector<double> ddPoints(DIMENSION * numDDpoints, 0.0);
	std::string ddPoints_str = bcdata["BCData"]["DisplacementConstraintsBCs"]["dDPoints"].get<std::string>();
	
	int indexx = 0;
	for(int i = 0; i < numDDpoints; i++) {
	    for(int j = 0; j < DIMENSION; j++) {
		std::string curr_value = ""; // Alternative: std::ostringstream curr_value;
		bool value_finished = false;
		while(!value_finished && (indexx < ddPoints_str.length())) {
		  // check string components for validity (i.e. numbers, +/-, points, kommas, semicolons):
		  if ( ddPoints_str[indexx] != '0' && ddPoints_str[indexx] != '1' && ddPoints_str[indexx] != '2' && 
			ddPoints_str[indexx] != '3' && ddPoints_str[indexx] != '4' && ddPoints_str[indexx] != '5' && 
			ddPoints_str[indexx] != '6' && ddPoints_str[indexx] != '7' && ddPoints_str[indexx] != '8' && 
			ddPoints_str[indexx] != '9' && ddPoints_str[indexx] != ',' && ddPoints_str[indexx] != ';' && 
			ddPoints_str[indexx] != '-' && ddPoints_str[indexx] != '.' && ddPoints_str[indexx] != '+' ) {
		    std::cout << "Error when parsing BCdata-strings. --> string contains char which is not in {'0'...'9', '+', '-', ',', ';', '.'}" << std::endl;
		    assert(0);
		  }
		  // if string component is not a comma or semicolon, add it to string of curr_value:
		  if( (ddPoints_str[indexx] != ',') && (ddPoints_str[indexx] != ';') ) {
		    curr_value += ddPoints_str[indexx]; // Alternative: curr_value << ddPoints_str[indexx];
		  } else {
		    value_finished = true;
		  }
		  ++indexx;
		}
		//convert string curr_value into double value and store it into ddPoints[i*DIMENSION + j]:
		double temp_ddPoint_value = atof( curr_value.c_str() );
		// function "atof()": double atof (const char* str) parses a string, interprets as double and returns its value.
 		ddPoints[i*DIMENSION + j] = temp_ddPoint_value;
		//std::cout << "TestOutput: ddPoints: (" << i << ", " << j << ") = " << ddPoints[i*DIMENSION + j] << std::endl;
	    }
	}
	
	// Get Dirichlet Displacement Vectors: --------------------------------------------------
	std::vector<double> dDisplacements(DIMENSION * numDDpoints, 0.0);
	std::string dDisplacements_str = bcdata["BCData"]["DisplacementConstraintsBCs"]["dDisplacements"].get<std::string>();
	
	indexx = 0;
	for(int i = 0; i < numDDpoints; i++) {
	    //std::cout << "TestOutput: Processing DDpoint " << i << std::endl;
	    for(int j = 0; j < DIMENSION; j++) {
		std::string curr_value = ""; // Alternative: std::ostringstream curr_value;
		bool value_finished = false;
		while(!value_finished && (indexx < dDisplacements_str.length())) {
		  // check string components for validity (i.e. numbers, +/-, points, kommas, semicolons):
		  if ( dDisplacements_str[indexx] != '0' && dDisplacements_str[indexx] != '1' && dDisplacements_str[indexx] != '2' && 
			dDisplacements_str[indexx] != '3' && dDisplacements_str[indexx] != '4' && dDisplacements_str[indexx] != '5' && 
			dDisplacements_str[indexx] != '6' && dDisplacements_str[indexx] != '7' && dDisplacements_str[indexx] != '8' && 
			dDisplacements_str[indexx] != '9' && dDisplacements_str[indexx] != ',' && dDisplacements_str[indexx] != ';' && 
			dDisplacements_str[indexx] != '-' && dDisplacements_str[indexx] != '.' && dDisplacements_str[indexx] != '+' ) {
		    std::cout << "Error when parsing BCdata-strings. --> string contains char which is not in {'0'...'9', '+', '-', ',', ';', '.'}" << std::endl;
		    assert(0);
		  }
		  // if string component is not a comma or semicolon, add it to string of curr_value:
		  if((dDisplacements_str[indexx] != ',') && (dDisplacements_str[indexx] != ';')) { // Try this for ' ' (blankspace).
		    curr_value += dDisplacements_str[indexx]; // Alternative: curr_value << dDisplacements_str[indexx];
		  } else {
		    value_finished = true;
		  }
		  ++indexx;
		}
		// convert string curr_value into double value and store it into dDisplacements[i*DIMENSION + j]:
		double temp_dDisplacement_value = atof( curr_value.c_str() );
		// function "atof()": double atof (const char* str) parses a string, interprets as double and returns its value.
		dDisplacements[i*DIMENSION + j] = temp_dDisplacement_value;
		//std::cout << "TestOutput: dDisplacements: (" << i << ", " << j << ") = " << dDisplacements[i*DIMENSION + j] << std::endl;
	    }
	}
	
	// Now, process the above read DirichletBCs: -----------------------------------
	std::vector<double> curr_dBCpoint(DIMENSION, 0.0);
	
	for (int BCpointCounter = 0; BCpointCounter < numDDpoints; ++BCpointCounter) {
	  
	  curr_dBCpoint[0] = ddPoints[BCpointCounter*DIMENSION + 0];
	  curr_dBCpoint[1] = ddPoints[BCpointCounter*DIMENSION + 1];
	  curr_dBCpoint[2] = ddPoints[BCpointCounter*DIMENSION + 2];
	  
	  // If curr_dBCpoint is not in/on mesh, find_closest_point on boundary of mesh and the appendant cell's facet_ID:
	  int boundary_facet_index;
	  std::vector<double> projected_curr_dBCpoint = geom_search.find_closest_point_parallel(curr_dBCpoint, boundary_facet_index, comm_);
	  // Note: find_closest_point() works only if the input mesh (which was given to "read_mesh()")
	  // specified material_IDs for the boundary facets other than -1.
	  // Since this is w.log. not the case, "set_default_material_number_on_bdy()" sets default bdy_facet_IDs in "read_mesh()".
	  // Note: "boundary_facet_index" is available on one process only! all other processes have boundary_facet_index = -1.
	  
	  // extract boundary of mesh
	  MeshPtr boundary_mesh_pointer = geom_search.get_boundary_mesh();
	  
	  // get mesh_facet_index:
	  int mesh_facet_index;
	  if (boundary_facet_index != -1) { // Note: "boundary_facet_index" is available on one process only.
	    boundary_mesh_pointer->get_attribute_value("_mesh_facet_index_", DIMENSION-1, boundary_facet_index, &mesh_facet_index);
	    
	    // get_entity(DIM-1)
	    Entity mesh_facet = mesh_->get_entity(DIMENSION-1, mesh_facet_index);
	    assert(mesh_facet.num_incident_entities(DIMENSION) == 1);
	    
	    // get pointer on cell:
	    IncidentEntityIterator cell = mesh_facet.begin_incident(DIMENSION);
	    
	    // get number of facet with respect to cell
	    int facet_number = 0;
	    for (IncidentEntityIterator inc_facet = cell->begin_incident(DIMENSION-1); mesh_facet_index != inc_facet->index();
		    ++inc_facet){
		++facet_number;
	    }
	    
	    const DofPartition<double>& dof = space_.dof();
	    
	    for (int var = 0; var < DIMENSION; ++var) {
		std::vector<int> dofs_on_facet;
		dof.get_dofs_on_subentity(var, cell->index(), DIMENSION-1, facet_number, dofs_on_facet);
		// iterate dofs
		for (std::vector<int>::iterator facet_dof_it = dofs_on_facet.begin(); 
		      facet_dof_it != dofs_on_facet.end(); ++facet_dof_it) {
		  if (dof.owner_of_dof(*facet_dof_it) == rank()) {
		    displacement_dirichlet_dofs_.push_back(*facet_dof_it);
		    displacement_dirichlet_values_.push_back(dDisplacements.at(BCpointCounter*DIMENSION + var));
		    // The following is done later outside the initial if (time_ == delta_t_) procedure.
// 		    if (solve_instationary_ == 0) {
// 		      //here(i am the stationary dirichlet bc processing procedure);
// 		      displacement_dirichlet_values_.push_back(dDisplacements.at(BCpointCounter*DIMENSION + var));
// 		    } else {
// 		      //here(i am the instationary dirichlet bc processing procedure);
// 		      if( (ts_ * delta_t_) <= 1.0 ) {
// 			// the prescribed displacement is reached linearly within the first second:
// 			displacement_dirichlet_values_.push_back(dDisplacements.at(BCpointCounter*DIMENSION + var) * ts_ * delta_t_);
// 		      } else {
// 			// after the first second, the prescribed displacement is maintained:
// 			displacement_dirichlet_values_.push_back(dDisplacements.at(BCpointCounter*DIMENSION + var));
// 			// or in case that displacement BCs shall be released again:
// 			// re-assemble stiffness matrix and dirichlet vector newly
// 			// without allocating space in the BC vector, by means of looping only over the fixedDirichlet-BCs.
// 		      }
// 		    }
		  }
		}
	    }
	  }
	}
	
      }
      // END OF ddPoints: ----------------------------------------
      if (rank_ == MASTER_RANK) {
        std::cout << "ControlOutput: Pointwise DirichletBC parts of BCdata: successfully read for simulation in initial step.\n" << std::endl;
      }
    }
    // END OF reading the BCdata-file
    // either for stationary simulation or during the initial if ( time_ == delta_t_ ) loop in instationary simulation.
    
    // ---------------------------------------------------------------------------
    // Now, for stationary simulation, do:
    if (solve_instationary_ == 0) {
      
      // Now, firstly, insert the above computed fixed_dirichlet_ dofs_ and values_ into the general dirichlet-vectors:
      dirichlet_dofs_.insert( dirichlet_dofs_.end(), fixed_dirichlet_dofs_.begin(), fixed_dirichlet_dofs_.end() );
      dirichlet_values_.insert( dirichlet_values_.end(), fixed_dirichlet_values_.begin(), fixed_dirichlet_values_.end() );
      // Then, secondly, insert the above computed displacement_dirichlet_ dofs_ and values_ into the general dirichlet-vectors:
      dirichlet_dofs_.insert( dirichlet_dofs_.end(), displacement_dirichlet_dofs_.begin(), displacement_dirichlet_dofs_.end() );
      dirichlet_values_.insert( dirichlet_values_.end(), displacement_dirichlet_values_.begin(), displacement_dirichlet_values_.end() );
      
    } else { // if (solve_instationary_ == 1)
      
      // Now, in every time step, do:
      // Scale displacement_dirichlet_values_ according to current timestep:
      std::vector<Scalar> scaled_displacement_dirichlet_values_(displacement_dirichlet_values_.size(), 0.0);
      if ( (ts_ * delta_t_) <= 1.0 ) {
        for (int i = 0; i < displacement_dirichlet_values_.size(); ++i) {
	  scaled_displacement_dirichlet_values_[i] = ts_ * delta_t_ * displacement_dirichlet_values_[i];
        }
      } else if ((ts_ * delta_t_) > 1.0 ) {
        for (int i = 0; i < displacement_dirichlet_values_.size(); ++i) {
	  scaled_displacement_dirichlet_values_[i] = displacement_dirichlet_values_[i];
        }
      }
      
      // Now, firstly, insert the above computed fixed_dirichlet_ dofs_ and values_ into the general dirichlet-vectors:
      dirichlet_dofs_.insert( dirichlet_dofs_.end(), fixed_dirichlet_dofs_.begin(), fixed_dirichlet_dofs_.end() );
      dirichlet_values_.insert( dirichlet_values_.end(), fixed_dirichlet_values_.begin(), fixed_dirichlet_values_.end() );
      // consider: std::vector<Scalar>::iterator 
      // InputIterator shall be an input iterator type that points to elements of a type from which value_type objects can be constructed.
      
      // Then, secondly, insert the above computed displacement_dirichlet_ dofs_ and values_ into the general dirichlet-vectors:
      if ( (ts_ * delta_t_) <= 1.0 || !dispDirBCsReleaseValue ) {
        dirichlet_dofs_.insert( dirichlet_dofs_.end(), displacement_dirichlet_dofs_.begin(), displacement_dirichlet_dofs_.end() );
        dirichlet_values_.insert( dirichlet_values_.end(), scaled_displacement_dirichlet_values_.begin(), scaled_displacement_dirichlet_values_.end() );
      }
    }
    // --------------------------------------------------------------------------
    
    if (rank_ == MASTER_RANK) {
       std::cout << "ControlOutput: Pointwise DirichletBC parts of BCdata: successfully read and processed for simulation." << std::endl;
    }
    // --------------------------------------------------------------------------
    
    // Third: NeumannBC-parts: Pressure/ForceNeumann: // TASK.
    // TODO. handle pointwise NeumannBCs (i.e. ForceOrPressureBCPoints from BCdata-InputFile), too.
    if (rank_ == MASTER_RANK) {
       std::cout << "ControlOutput: Pointwise NeumannBC parts of BCdata: not yet implemented." << std::endl;
       // ... successfully read and processed for simulation. -> NOT_YET_IMPLEMENTED." << std::endl;
    }
    
    // -----------------------------------------------------------------------------------------------------<
}

void Elasticity::prepare_bc_instationary() {
    TimingScope tscope("prepare_bc_instationary");
    
    dirichlet_dofs_.clear();
    dirichlet_values_.clear();
    
    // >-----------------------------------------------------------------------------------------------------
    // Set DirichletBCs for prescribed facets (i.e. material_IDs given in hiflow3_scene.xml-File): // DEPRECATED, but still working.
    const int dir_bdy1 = params_["Boundary"]["DirichletMaterial1"].get<int>();
    const int dir_bdy2 = params_["Boundary"]["DirichletMaterial2"].get<int>();
    const int dir_bdy3 = params_["Boundary"]["DirichletMaterial3"].get<int>();
    // NOTE: DirichletBCs by means of prescribed facets are deprecated, and replaced by means of pointwise DirichletBCs, see below.
    
    // create InstationaryElasticity_DirichletBC_3D-Object.
    InstationaryElasticity_DirichletBC_3D bc[3] = { 
			      InstationaryElasticity_DirichletBC_3D(0, dir_bdy1, dir_bdy2, dir_bdy3, ts_, delta_t_),
			      InstationaryElasticity_DirichletBC_3D(1, dir_bdy1, dir_bdy2, dir_bdy3, ts_, delta_t_),
			      InstationaryElasticity_DirichletBC_3D(2, dir_bdy1, dir_bdy2, dir_bdy3, ts_, delta_t_) };
    
    // and compute Dirichlet values for pre-set dofs.
    for (int var = 0; var < DIMENSION; ++var) {
        compute_dirichlet_dofs_and_values(bc[var], space_, var,
                                          dirichlet_dofs_, dirichlet_values_);
    }
    
    // >-----------------------------------------------------------------------------------------------------
    // Set pointwise BCs as given in BCdata-Input-File:
    process_BCdataInputfile_into_BCdataStructure();
    // >-----------------------------------------------------------------------------------------------------
}

void Elasticity::find_potential_contact_elements() { // NOTE: algorithms not yet published // TODO.
  TimingScope tscope("find_potential_contact_elements");
  // this function (rather the assembler which is included) finds a next step's potential contact elements
  // i.e., elements of two different body parts (with different material IDs) 
  // that are within an epsilon-environment of each other, and thus are subject to potential contact.
  
  std::cout << std::endl; // separating the previous time step in the output from the current time step...
  
  // ------------------------------------------------------------------------------------
  // delete contents of former timestep's cbc_-vector.
  cbc_.Zeros(); // maybe use "Zeros()" instead of "Clear()"?!
  
  // ------------------------------------------------------------------------------------
  // get (first/second) ContactBC-materialID.
  const int contact_bdy1 = params_["Boundary"]["ContactMaterial1"].get<int>();
  const int contact_bdy2 = params_["Boundary"]["ContactMaterial2"].get<int>();
  
  // get contact tolerance threshold (i.e., the value as from which on we say "there is contact").
  const double ContactToleranceThreshold = params_["Boundary"]["ContactToleranceThreshold"].get<double>();
  
  // ------------------------------------------------------------------------------------
  // Set up new mesh-pointer w.r.t. the current timestep's deformed state of mesh
  // in order to give it as an argument to the contact-assembler and its internal GridGeometricSearch.
  
  // therefore, copy the last timestep's solution-vector 
  // (i.e. the last timestep's displacement w.r.t. the initial state):
  std::vector<double> temp_sol_vec(sol_.size_global(), 0.0);
  std::vector<int> temp_dof_ids;
  std::vector<double> temp_sol_values;
  sol_.GetAllDofsAndValues(temp_dof_ids, temp_sol_values);
  for (int i = 0; i < temp_sol_values.size(); ++i) {
    temp_sol_vec.at(temp_dof_ids[i]) = temp_sol_values.at(i);
    // temp_sol_values.at(i) corresponds to displacement values.
    // Note: temp_sol_vec does not contain the body's deformed state
    // (the deformed state would correspond to orig values + displacement values)
  }
  
  // Create vector that contains the displacement which is set onto the mesh/geometry
  // in order to yield the deformed state:
  std::vector<double> displacement_vector(3 * mesh_->num_entities(0));
  
  std::vector<int> local_dof_id;
  
  AttributePtr sub_domain_attr;
  // if attribute _sub_domain_ exists, there are several subdomains, and hence ghost cells, too.
  const bool has_ghost_cells = mesh_->has_attribute("_sub_domain_", mesh_->tdim());
  if (has_ghost_cells) {
      sub_domain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
  }
  
  // loop over all mesh cells
  for (EntityIterator cell = mesh_->begin(mesh_->tdim()), end_c = mesh_->end(mesh_->tdim());
       cell != end_c; ++cell) {
    
    int vertex_num = 0;
    
    if (has_ghost_cells && sub_domain_attr->get_int_value(cell->index()) != rank_) {
	continue; // in order not to write values twice/thrice/... for parallel processes
    }
    
    // loop over all vertices (incident_dim = 0) in current mesh cell
    for (IncidentEntityIterator vertex_it = cell->begin_incident(0); 
	 vertex_it != cell->end_incident(0); ++vertex_it) {
      
      // loop over DIMENSION
      for (int var = 0; var < 3; ++var) {
	
	// get local_dof_id w.r.t. vertex_num and var
	space_.dof().get_dofs_on_subentity(var, cell->index(), 0, vertex_num, local_dof_id);
	displacement_vector.at(3 * vertex_it->index() + var) = temp_sol_vec.at(local_dof_id[0]);
	// this for-loop writes the values from "visu_vec" (which contains the solution values (u1,u2,u3))
	// into the "displacement_vector", which then (below) is given to the mesh->move_vertices().
      }
      
      ++vertex_num;
    } // end of loop over all vertices of current mesh cell.
    
  } // end of loop over all mesh cells.
  
//   // prepare stuff for initialization of new VectorSpace object:
//   std::vector< int > temp_degrees(DIMENSION);
//   for (int c = 0; c < DIMENSION; ++c) {
//         temp_degrees.at(c) = u_deg;
//   }
//   // Initialize the VectorSpace object
//   VectorSpace<double> temp_space_;
//   temp_space_.Init(temp_degrees, *master_mesh_copy);
// //   std::cout << "ControlOutput: size of master_mesh_copy = " << temp_space_.dof().ndofs_global() << std::endl;
//   
//   // check that ( size_of_master_mesh_copy == size_of_temp_sol_vec ) is satisfied:
//   assert(sol_.size_global() == temp_space_.dof().ndofs_global() );
//   
//   // finally, move vertices of master_mesh_copy:
// //   master_mesh_copy->move_vertices( temp_sol_vec );
  
  // finally, move vertices of master_mesh_copy:
//   mesh_->move_vertices( displacement_vector ); // use master_mesh_copy instead of mesh_ ?!
  master_mesh_copy->move_vertices( displacement_vector );
  
  // ------------------------------------------------------------------------------------
  // Set up (time-dependent) "Contact BCs":
  
  // setup ContactBC-Assembler for contactBCs.
  
	// set up contactDistance_lookup_table as a map
	// containing (int)cell_indices and (double)current_nearest_distances_to_opposite_bdys :
	std::map< int, double > contactDistance_lookup_table;
	
	// ------------------------------------------------------------------------------------
	// Initialize GridGeometricSearch:
	//  in order to for a given point find the cell and its localCellIndex and its globalCellID.
	GridGeometricSearch geom_search(master_mesh_copy);
	
	// extract boundary of mesh
	MeshPtr boundary_mesh_pointer = geom_search.get_boundary_mesh();

	// prepare communication:
	
	// NOTE: algorithms not yet published // TODO.
	// code available only on request.
	// please contact: nicolai.schoch@iwr.uni-heidelberg.de
	
  // ------------------------------------------------------------------------------------
  // Re-transform the mesh, by means of subtracting the displacement_vector again:
  // the reason for the need of doing this: disp_t+1  = sol_(delta_t_) and NOT disp_t+1 - disp_t = sol_(delta_t_).
  for (int i = 0; i < displacement_vector.size(); ++i) { // displacement_vector *= -1.0; // componentwise...
        displacement_vector[i] = -1.0 * displacement_vector[i];
  }
//   mesh_->move_vertices( displacement_vector ); // use master_mesh_copy instead of mesh_ ?!?!
  master_mesh_copy->move_vertices( displacement_vector );
  
  // ------------------------------------------------------------------------------------
  
//   // ------------------------------------------------------------------------------------
//   // Set up (time-dependent) "Contact BCs":
//   // Set up ContactBC-type/struct:
//   InstationaryElasticity_contactBC_3D elast_contact_eval(contact_bdy1, contact_bdy2, ts_, delta_t_);
//   
//   // Re-transformation of the mesh (see above) needed before calling the assembler:
//   // NOTE: algorithms not yet published // TODO.
//   // code available only on request.
//   // please contact: nicolai.schoch@iwr.uni-heidelberg.de
//   InstationaryElasticityContactAssembler<InstationaryElasticity_contactBC_3D>
//       local_contact_asm(&elast_contact_eval, contact_bdy1, contact_bdy2, ts_, delta_t_, 
// 			master_mesh_copy, comm_, ContactToleranceThreshold, contactDistance_lookup_table);
//   
//   // assemble ContactBC vector.
//   global_asm_.assemble_vector_boundary(space_, local_contact_asm, cbc_);
//   
//   // ------------------------------------------------------------------------------------
  
  // add/subtract ContactBC cBC_ vector from rhs_ vector in order to integrate it into the whole weak formulation.
  // rhs_.Axpy(cbc_, 1.0);      // This is done below in "assembly()".
  // rhs_.UpdateCouplings();    // This is done below in "assembly()".
  
//   std::cout << "   -> I am process "<< rank_ + 1 << "/" << num_partitions_ << " and I am DONE with find_potential_contact_elements()." << std::endl;
}

void Elasticity::assemble_system() {
    
  if (!solve_instationary_) { // ATTENTION: NeumannBCs deprecated, and no ContactBCs at all. partly DEPRECATED.
    TimingScope tscope("Assemble_system_stationary");
    
    StationaryElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_);
    // TASK: Implement this element-specifically!
    global_asm_.assemble_matrix(space_, local_asm, matrix_);
    
    // Produce test output for mmread() in order to check symmetry properties of matrix_:
    //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
    // Then check for symmetry properties of matrix_ outside HiFlow3:
    // Console commands using Octave and mmread():
    // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
    // console output: ans = x, where x is not 0  ==>  A is symmetric.
    
    global_asm_.assemble_vector(space_, local_asm, rhs_);
    
    // Implement new NeumannBC-struct for all nBC-materials:
    // get (first) NeumannBC-materialID.
    const int neum_bdy1 = params_["Boundary"]["NeumannMaterial1"].get<int>();
#ifdef HEART_MV_SCENARIO
    // get (first) NeumannBC-pressure in mmHg.
    n_bdy1_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
    // transform (first) NeumannBC-pressure from mmHg into Pascal (1mmHg = 133,322Pa).
    n_bdy1_force_in_Pascal_ = 133.332 * n_bdy1_force_in_mmHg_ * 0.01;
#endif
#ifdef NON_MMHG_SCENARIO
    n_bdy1_force_in_Pascal_  = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
#endif
    // set up (first) NeumannBC-type (here: RadialNeumannBC).
    StationaryElasticity_RadialNeumannBC_3D elast_neum_eval[3] = {
    		StationaryElasticity_RadialNeumannBC_3D(0, n_bdy1_force_in_Pascal_),
    		StationaryElasticity_RadialNeumannBC_3D(1, n_bdy1_force_in_Pascal_),
    		StationaryElasticity_RadialNeumannBC_3D(2, n_bdy1_force_in_Pascal_)};
    // start NeumannBC-Assembler for (first) NeumannBC.
    StationaryElasticityNeumannAssembler<StationaryElasticity_RadialNeumannBC_3D> 
	local_neum_asm(elast_neum_eval, neum_bdy1);
    // assemble (first) NeumannBC vector.
    global_asm_.assemble_vector_boundary(space_, local_neum_asm, nbc_);
    // add/subtract (first) NeumannBC nBC_ vector from rhs_ vector.
    rhs_.Axpy(nbc_, 1.0);
    rhs_.UpdateCouplings();
    
    if (!dirichlet_dofs_.empty()) {
      // Correct Dirichlet dofs.
      matrix_.diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), 1.0);
      rhs_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                      vec2ptr(dirichlet_values_));
      sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                      vec2ptr(dirichlet_values_));
    }
    
    rhs_.UpdateCouplings();
    sol_.UpdateCouplings();
    
    // Setup Operator for Preconditioner/Solver:
    // (needs to be done AFTER DirichletBC-Setup)
    if (solver_name_ == "CG" && matrix_precond_ != NOPRECOND) {
      preconditioner_.SetupOperator(matrix_);
    }
    else if (solver_name_ == "CG" && matrix_precond_ == NOPRECOND) {
      cg_.SetupOperator(matrix_);
    }
    else {
      // GMRES (and ILUPP) by default.
      if (use_ilupp_) {
#ifdef WITH_ILUPP
	ilupp_.SetupOperator(matrix_);
#endif
      }
    }
    
    // Produce test output for mmread() in order to check symmetry properties of matrix_:
    //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
    // Then check for symmetry properties of matrix_ outside HiFlow3:
    // Console commands using Octave and mmread():
    // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
    // console output: ans = x, where x is not 0  ==>  A is symmetric.
    
  }
  else if (solve_instationary_) {
    TimingScope tscope("Assemble_system_instationary");
    
    // prepare dirichlet BC
    prepare_bc_instationary();
    // Note: the InstationaryElasticity_DirichletBC_3D-struct assumes 
    // that the given DirichletBC-values are reached linearly continuously and within 1.0sec real time,
    // i.e.: ts_ * delta_t_ * dirichlet_values_ (as long as (ts_ * delta_t_) <= 1.0)
    //  and:  dirichlet_values_ (as soon as ts_ * delta_t_ >= 1.0).
    // (except for fixed boundaries with u_D = 0.0 all time long).
    // Note: if a scenario is not set up in this way, one has to check for loosening DirichletBCs again
    // by means of uncommenting the if (time_ == delta_t_) case around the assemble_matrix() 20 lines below
    // and adaptations in the process_BCdataInputfile_into_BCdataStructure() function.
    
    if ( !corotFormValue ) { // Linear Elasticity Formulation.
      
      // Use LinearElasticity-Assembler:
      InstationaryLinElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_, dampingFactor_, rayleighAlpha_, rayleighBeta_);
      // TASK: implement this element-specifically!
      
      // Initialize timestepping
      local_asm.set_timestep_parameters(theta_, delta_t_);
      local_asm.set_time(time_);
      local_asm.set_prev_solution(&sol_prev_);
      local_asm.set_dt_prev_solution(&dt_sol_prev_);
      local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
      
      // Assembly of the system matrix K_eff:
      // In Linear Elasticity, this is done once only, since the Matrix is NOT timestep-dependent (as opposed to the R_eff-vector).
      // However, it has to be done in every step in case of changing Dirichlet BCs (or their DoFs, respectively), such that "Displacement BCs" are released again.
      // In Corotational Elasticity, the stiffness matrix must be computed / updated in every time step.
      if ( corotFormValue || dispDirBCsReleaseValue || time_ == delta_t_ ) { // depends on boolean variable "dispDirBCsReleaseValue".
        // Note: instead of "time_ == delta_t_" write "almost_equal<double>( time_, delta_t_, 2 )" --> from "rotation.h".
        
	// Assemble system matrix:
	global_asm_.assemble_matrix(space_, local_asm, matrix_);
	
        // Produce test output for mmread() in order to check symmetry properties of matrix_:
        //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
        // Then check for symmetry properties of matrix_ outside HiFlow3:
        // Console commands using Octave and mmread():
        // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
        // console output: ans = x, where x is not 0  ==>  A is symmetric.
	
	// TASK: adapt this for performance reasons, if the scenario settings allow.
	// If RHS-Vector (time-independent, constant body force) AND 
	// NBC-Vector (in case of time-independent constant pressure) remain constant,
	// they will need to be computed once only in this loop:
	// R_eff_initial_ = rhs_initial_ + nbc_initial_;
	// By means of "global_asm_.should_reset_assembly_target(false)" the additional time-dependent parts 
	// can then be added in the subsequent timestepping: R_eff_new = R_eff_old + ...
	// TASK: end.
      }
      
      // Assembly of the system rhs vector R_eff:
      //rhs_.Zeros(); // Annotation: this is not needed, since assemble()-functions reset assembly_target by default.
      // global_asm_.should_reset_assembly_target(false); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
      global_asm_.assemble_vector(space_, local_asm, rhs_);
      // global_asm_.should_reset_assembly_target(true); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
      
      // Assembly of NeumannBCs vector nBC_:
	// global_asm_.should_reset_assembly_target(false);
	// TASK: for performance reasons, set assembly_target = rhs_ instead of a new vector nbc_.
	//       using "global_asm_.should_reset_assembly_target(false)" allows better performance than "Axpy()" etc.
	// global_asm_.should_reset_assembly_target(true);
	
	// TASK: Implement "flexible NeumannBC" struct to process flexibly many/multiple nBC-materials:
	// get (first/second) NeumannBC-materialID.
	const int neum_bdy1 = params_["Boundary"]["NeumannMaterial1"].get<int>();
	const int neum_bdy2 = params_["Boundary"]["NeumannMaterial2"].get<int>();
#ifdef HEART_MV_SCENARIO
	// get (first/second) NeumannBC-pressure in mmHg.
	n_bdy1_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
	n_bdy2_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
	// transform (first/second) NeumannBC-pressure from mmHg into Pascal (1mmHg = 133,322Pa).
	n_bdy1_force_in_Pascal_ = 133.332 * n_bdy1_force_in_mmHg_ * 0.01;
	n_bdy2_force_in_Pascal_ = 133.332 * n_bdy2_force_in_mmHg_ * 0.01;
#endif
#ifdef NON_MMHG_SCENARIO
	n_bdy1_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
	n_bdy2_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
#endif
	
	// Different types on Neumann BCs can be set up here:
	// e.g. Set up time-dependent "Sinusoidal Neumann BCs" that act orthogonally on surface facets:
	// set up NeumannBC-type.
	InstationaryElasticity_sinusoidalNeumannBC_3D elast_neum_eval(neum_bdy1, neum_bdy2, 
						n_bdy1_force_in_Pascal_, n_bdy2_force_in_Pascal_, ts_, delta_t_);
	
	// start NeumannBC-Assembler for NeumannBCs.
	InstationaryElasticityNeumannAssembler<InstationaryElasticity_sinusoidalNeumannBC_3D> 
	    local_neum_asm(&elast_neum_eval, neum_bdy1, neum_bdy2, ts_, delta_t_);
	
// 	// alternatively, to "Sinusoidal Neumann BCs": e.g. Set up Radial Neumann BCs:
// 	// set up (first) NeumannBC-type (here: RadialNeumannBC).
// 	InstationaryElasticity_RadialNeumannBC_3D elast_neum_eval[3] = {
//     		InstationaryElasticity_RadialNeumannBC_3D(0, n_bdy1_force_in_Pascal_, ts_, delta_t_),
//     		InstationaryElasticity_RadialNeumannBC_3D(1, n_bdy1_force_in_Pascal_, ts_, delta_t_),
//     		InstationaryElasticity_RadialNeumannBC_3D(2, n_bdy1_force_in_Pascal_, ts_, delta_t_)};
// 	// start NeumannBC-Assembler for (first) NeumannBC.
// 	InstationaryElasticityNeumannAssembler<InstationaryElasticity_RadialNeumannBC_3D> 
// 	    local_neum_asm(elast_neum_eval, neum_bdy1, ts_, delta_t_);
	
	// assemble NeumannBC vector.
	global_asm_.assemble_vector_boundary(space_, local_neum_asm, nbc_);
	
	// add/subtract NeumannBC nBC_ vector from rhs_ vector.
	rhs_.Axpy(nbc_, 1.0); // TASK: use "should_reset_assembly_target(false)" instead of "Axpy()".
	rhs_.UpdateCouplings();
	
	if (chanceOfContactValue) {
	// add/subtract ContactBC cBC_ vector from rhs_ vector.
	double cbc_contactPenaltyFactor = params_["Boundary"]["ContactPenaltyFactor"].get<double>();
	rhs_.Axpy(cbc_, cbc_contactPenaltyFactor); // this means force is x-fold, with x = cbc_contactPenaltyFactor.
	
	rhs_.UpdateCouplings();
	}
      
    } // end of Linear Formulation. /////////////////////////////////////////////////////
    else { // Corotational Formulation.
      
      // Use CorotationalElasticity-Assembler:
      InstationaryCorotElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_, dampingFactor_, rayleighAlpha_, rayleighBeta_); 
      // TASK: implement this element-specifically!
      
      // Initialize timestepping
      local_asm.set_timestep_parameters(theta_, delta_t_);
      local_asm.set_time(time_);
      local_asm.set_prev_solution(&sol_prev_);
      local_asm.set_dt_prev_solution(&dt_sol_prev_);
      local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
      
      // Assembly of the system matrix K_eff:
      // In Linear Elasticity, this is done once only, since the Matrix is NOT timestep-dependent (as opposed to the R_eff-vector).
      // However, it has to be done in every step in case of changing Dirichlet BCs (or their DoFs, respectively), such that "Displacement BCs" are released again.
      // In Corotational Elasticity, the stiffness matrix must be computed / updated in every time step.
      if ( corotFormValue || dispDirBCsReleaseValue || time_ == delta_t_ ) { // depends on boolean variable "dispDirBCsReleaseValue".
        // Note: instead of "time_ == delta_t_" write "almost_equal<double>( time_, delta_t_, 2 )" --> from "rotation.h".
        
	// Assemble system matrix:
	global_asm_.assemble_matrix(space_, local_asm, matrix_);
	
        // Produce test output for mmread() in order to check symmetry properties of matrix_:
        //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
        // Then check for symmetry properties of matrix_ outside HiFlow3:
        // Console commands using Octave and mmread():
        // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
        // console output: ans = x, where x is not 0  ==>  A is symmetric.
	
	// TASK: adapt this for performance reasons, if the scenario settings allow.
	// If RHS-Vector (time-independent, constant body force) AND 
	// NBC-Vector (in case of time-independent constant pressure) remain constant,
	// they will need to be computed once only in this loop:
	// R_eff_initial_ = rhs_initial_ + nbc_initial_;
	// By means of "global_asm_.should_reset_assembly_target(false)" the additional time-dependent parts 
	// can then be added in the subsequent timestepping: R_eff_new = R_eff_old + ...
	// TASK: end.
      }
      
      // Assembly of the system rhs vector R_eff:
      //rhs_.Zeros(); // Annotation: this is not needed, since assemble()-functions reset assembly_target by default.
      // global_asm_.should_reset_assembly_target(false); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
      global_asm_.assemble_vector(space_, local_asm, rhs_);
      // global_asm_.should_reset_assembly_target(true); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
      
      // Assembly of NeumannBCs vector nBC_:
	// global_asm_.should_reset_assembly_target(false);
	// TASK: for performance reasons, set assembly_target = rhs_ instead of a new vector nbc_.
	//       using "global_asm_.should_reset_assembly_target(false)" allows better performance than "Axpy()" etc.
	// global_asm_.should_reset_assembly_target(true);
	
	// TASK: Implement "flexible NeumannBC" struct to process flexibly many/multiple nBC-materials:
	// get (first/second) NeumannBC-materialID.
	const int neum_bdy1 = params_["Boundary"]["NeumannMaterial1"].get<int>();
	const int neum_bdy2 = params_["Boundary"]["NeumannMaterial2"].get<int>();
#ifdef HEART_MV_SCENARIO
	// get (first/second) NeumannBC-pressure in mmHg.
	n_bdy1_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
	n_bdy2_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
	// transform (first/second) NeumannBC-pressure from mmHg into Pascal (1mmHg = 133,322Pa).
	n_bdy1_force_in_Pascal_ = 133.332 * n_bdy1_force_in_mmHg_ * 0.01;
	n_bdy2_force_in_Pascal_ = 133.332 * n_bdy2_force_in_mmHg_ * 0.01;
#endif
#ifdef NON_MMHG_SCENARIO
	n_bdy1_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
	n_bdy2_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
#endif
	
	// Different types on Neumann BCs can be set up here:
	// e.g. Set up time-dependent "Sinusoidal Neumann BCs" that act orthogonally on surface facets:
	// set up NeumannBC-type.
	InstationaryElasticity_sinusoidalNeumannBC_3D elast_neum_eval(neum_bdy1, neum_bdy2, 
						n_bdy1_force_in_Pascal_, n_bdy2_force_in_Pascal_, ts_, delta_t_);
	
	// start NeumannBC-Assembler for NeumannBCs.
	InstationaryElasticityNeumannAssembler<InstationaryElasticity_sinusoidalNeumannBC_3D> 
	    local_neum_asm(&elast_neum_eval, neum_bdy1, neum_bdy2, ts_, delta_t_);
	
// 	// alternatively, to "Sinusoidal Neumann BCs": e.g. Set up Radial Neumann BCs:
// 	// set up (first) NeumannBC-type (here: RadialNeumannBC).
// 	InstationaryElasticity_RadialNeumannBC_3D elast_neum_eval[3] = {
//     		InstationaryElasticity_RadialNeumannBC_3D(0, n_bdy1_force_in_Pascal_, ts_, delta_t_),
//     		InstationaryElasticity_RadialNeumannBC_3D(1, n_bdy1_force_in_Pascal_, ts_, delta_t_),
//     		InstationaryElasticity_RadialNeumannBC_3D(2, n_bdy1_force_in_Pascal_, ts_, delta_t_)};
// 	// start NeumannBC-Assembler for (first) NeumannBC.
// 	InstationaryElasticityNeumannAssembler<InstationaryElasticity_RadialNeumannBC_3D> 
// 	    local_neum_asm(elast_neum_eval, neum_bdy1, ts_, delta_t_);
	
	// assemble NeumannBC vector.
	global_asm_.assemble_vector_boundary(space_, local_neum_asm, nbc_);
	
	// add/subtract NeumannBC nBC_ vector from rhs_ vector.
	rhs_.Axpy(nbc_, 1.0); // TASK: use "should_reset_assembly_target(false)" instead of "Axpy()".
	rhs_.UpdateCouplings();
	
	if (chanceOfContactValue) {
	// add/subtract ContactBC cBC_ vector from rhs_ vector.
	double cbc_contactPenaltyFactor = params_["Boundary"]["ContactPenaltyFactor"].get<double>();
	rhs_.Axpy(cbc_, cbc_contactPenaltyFactor); // this means force is x-fold, with x = cbc_contactPenaltyFactor.
	
	rhs_.UpdateCouplings();
	}
      
    } // end of Corotational Formulation.
    
//     InstationaryElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_, dampingFactor_, rayleighAlpha_, rayleighBeta_); // DEPRECATED.
//     // TASK: implement this element-specifically!
    
//     // Initialize timestepping
//     local_asm.set_timestep_parameters(theta_, delta_t_);
//     local_asm.set_time(time_);
//     local_asm.set_prev_solution(&sol_prev_);
//     local_asm.set_dt_prev_solution(&dt_sol_prev_);
//     local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
//     
//     // Assembly of the system matrix K_eff:
//     // In Linear Elasticity, this is done once only, since the Matrix is NOT timestep-dependent (as opposed to the R_eff-vector).
//     // However, it has to be done in every step in case of changing Dirichlet BCs (or their DoFs, respectively), such that "Displacement BCs" are released again.
//     // In Corotational Elasticity, the stiffness matrix must be computed / updated in every time step.
//     if ( corotFormValue || dispDirBCsReleaseValue || time_ == delta_t_ ) { // depends on boolean variable "dispDirBCsReleaseValue".
//         // Note: instead of "time_ == delta_t_" write "almost_equal<double>( time_, delta_t_, 2 )" --> from "rotation.h".
//         
// 	// Assemble system matrix:
// 	global_asm_.assemble_matrix(space_, local_asm, matrix_);
// 	
//         // Produce test output for mmread() in order to check symmetry properties of matrix_:
//         //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
//         // Then check for symmetry properties of matrix_ outside HiFlow3:
//         // Console commands using Octave and mmread():
//         // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
//         // console output: ans = x, where x is not 0  ==>  A is symmetric.
// 	
// 	// TASK: adapt this for performance reasons, if the scenario settings allow.
// 	// If RHS-Vector (time-independent, constant body force) AND 
// 	// NBC-Vector (in case of time-independent constant pressure) remain constant,
// 	// they will need to be computed once only in this loop:
// 	// R_eff_initial_ = rhs_initial_ + nbc_initial_;
// 	// By means of "global_asm_.should_reset_assembly_target(false)" the additional time-dependent parts 
// 	// can then be added in the subsequent timestepping: R_eff_new = R_eff_old + ...
// 	// TASK: end.
//     }
//     
//     // Assembly of the system rhs vector R_eff:
//     //rhs_.Zeros(); // Annotation: this is not needed, since assemble()-functions reset assembly_target by default.
//     // global_asm_.should_reset_assembly_target(false); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
//     global_asm_.assemble_vector(space_, local_asm, rhs_);
//     // global_asm_.should_reset_assembly_target(true); // this may be needed if rhs_ and nBC_ are assembled into the same assembly_target.
//     
//     // Assembly of NeumannBCs vector nBC_:
// 	// global_asm_.should_reset_assembly_target(false);
// 	// TASK: for performance reasons, set assembly_target = rhs_ instead of a new vector nbc_.
// 	//       using "global_asm_.should_reset_assembly_target(false)" allows better performance than "Axpy()" etc.
// 	// global_asm_.should_reset_assembly_target(true);
// 	
// 	// TASK: Implement "flexible NeumannBC" struct to process flexibly many/multiple nBC-materials:
// 	// get (first/second) NeumannBC-materialID.
// 	const int neum_bdy1 = params_["Boundary"]["NeumannMaterial1"].get<int>();
// 	const int neum_bdy2 = params_["Boundary"]["NeumannMaterial2"].get<int>();
// #ifdef HEART_MV_SCENARIO
// 	// get (first/second) NeumannBC-pressure in mmHg.
// 	n_bdy1_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
// 	n_bdy2_force_in_mmHg_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
// 	// transform (first/second) NeumannBC-pressure from mmHg into Pascal (1mmHg = 133,322Pa).
// 	n_bdy1_force_in_Pascal_ = 133.332 * n_bdy1_force_in_mmHg_ * 0.01;
// 	n_bdy2_force_in_Pascal_ = 133.332 * n_bdy2_force_in_mmHg_ * 0.01;
// #endif
// #ifdef NON_MMHG_SCENARIO
// 	n_bdy1_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial1Pressure"].get<double>();
// 	n_bdy2_force_in_Pascal_ = params_["Boundary"]["NeumannMaterial2Pressure"].get<double>();
// #endif
// 	
// 	// Different types on Neumann BCs can be set up here:
// 	// e.g. Set up time-dependent "Sinusoidal Neumann BCs" that act orthogonally on surface facets:
// 	// set up NeumannBC-type.
// 	InstationaryElasticity_sinusoidalNeumannBC_3D elast_neum_eval(neum_bdy1, neum_bdy2, 
// 						n_bdy1_force_in_Pascal_, n_bdy2_force_in_Pascal_, ts_, delta_t_);
// 	
// 	// start NeumannBC-Assembler for NeumannBCs.
// 	InstationaryElasticityNeumannAssembler<InstationaryElasticity_sinusoidalNeumannBC_3D> 
// 	    local_neum_asm(&elast_neum_eval, neum_bdy1, neum_bdy2, ts_, delta_t_);
// 	
// // 	// alternatively, to "Sinusoidal Neumann BCs": e.g. Set up Radial Neumann BCs:
// // 	// set up (first) NeumannBC-type (here: RadialNeumannBC).
// // 	InstationaryElasticity_RadialNeumannBC_3D elast_neum_eval[3] = {
// //     		InstationaryElasticity_RadialNeumannBC_3D(0, n_bdy1_force_in_Pascal_, ts_, delta_t_),
// //     		InstationaryElasticity_RadialNeumannBC_3D(1, n_bdy1_force_in_Pascal_, ts_, delta_t_),
// //     		InstationaryElasticity_RadialNeumannBC_3D(2, n_bdy1_force_in_Pascal_, ts_, delta_t_)};
// // 	// start NeumannBC-Assembler for (first) NeumannBC.
// // 	InstationaryElasticityNeumannAssembler<InstationaryElasticity_RadialNeumannBC_3D> 
// // 	    local_neum_asm(elast_neum_eval, neum_bdy1, ts_, delta_t_);
// 	
// 	// assemble NeumannBC vector.
// 	global_asm_.assemble_vector_boundary(space_, local_neum_asm, nbc_);
// 	
// 	// add/subtract NeumannBC nBC_ vector from rhs_ vector.
// 	rhs_.Axpy(nbc_, 1.0); // TASK: use "should_reset_assembly_target(false)" instead of "Axpy()".
// 	rhs_.UpdateCouplings();
// 	
// 	// add/subtract ContactBC cBC_ vector from rhs_ vector.
// 	double cbc_contactPenaltyFactor = params_["Boundary"]["ContactPenaltyFactor"].get<double>();
// 	rhs_.Axpy(cbc_, cbc_contactPenaltyFactor); // this means force is x-fold, with x = cbc_contactPenaltyFactor.
// 	
// 	rhs_.UpdateCouplings();
	
    ////////////////////////////////
    
    // Accounting for DirichletBCs (stronger than NeumannBCs, therefore implemented after the NeumannBCs):
    if (!dirichlet_dofs_.empty()) {
      // Correct Dirichlet dofs.
      matrix_.diagonalize_rows(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(), 1.0); // Note: see note below.
      // NB: this would only have to be done once in the above condition "if (time_ == delta_t_) {...}".
      rhs_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                      vec2ptr(dirichlet_values_));
      sol_.SetValues(vec2ptr(dirichlet_dofs_), dirichlet_dofs_.size(),
                      vec2ptr(dirichlet_values_));
    }
    
    rhs_.UpdateCouplings();
    sol_.UpdateCouplings();
    
    // Setup Operator for Preconditioner/Solver.
    // NOTE: Theoretically, "SetupOperator(matrix_)" for Preconditioner, is needed once only, after the Dirichlet-BCs setup/computation.
    //       Then, since matrix_ columns are set to "1.0" at the very first timestep already, it should not be needed anymore afterwards.
    if (solver_name_ == "CG" && matrix_precond_ != NOPRECOND) {
      preconditioner_.SetupOperator(matrix_);
    }
    else if (solver_name_ == "CG" && matrix_precond_ == NOPRECOND) {
      cg_.SetupOperator(matrix_);
    }
    else {
      // GMRES (and ILUPP) by default.
      if (use_ilupp_) {
#ifdef WITH_ILUPP
	ilupp_.SetupOperator(matrix_);
#endif
      }
    }
    // Produce test output for mmread() in order to check symmetry properties of matrix_:
    //matrix_.diagonal().WriteFile("mmreadOutput_StationarySimulation_beforeDirBCs_prematrix.mat");
    // Then check for symmetry properties of matrix_ outside HiFlow3:
    // Console commands using Octave and mmread():
    // --> octave --> A = mmread("prematrix.mat"); --> issymmetric(A, 1.e-9)
    // console output: ans = x, where x is not 0  ==>  A is symmetric.
  }
  else {
    std::cout << "Error in assemble_system(). -> Boolean solve_instationary_ seems not to be known." << std::endl;
    assert(0);
  }
}

void Elasticity::solve_system() {
  
  // Solver setup: done in prepare_system() and assemble_system().
  // NOTE on implementation: this could alternatively be done in the Elasticity-class:
  // - via a solver factory, or
  // - via allocateing a linear-solver-object, which is then filled by CG, GMRES, etc. (as given in the xml-input-file),
  // instead of setting up a CG-solver AND a GMRES-solver <LAD> in the Elasticity-class.
  
  if (!solve_instationary_) {
    TimingScope tscope("Solve_system_stationary");
    
    // Solve linear system.
    if (solver_name_ == "CG") {
      cg_.Solve(rhs_, &sol_);
    } else /*if (solver_name_ == "GMRES")*/ {
      gmres_.Solve(rhs_, &sol_);
    }
    
    sol_.UpdateCouplings();
  }
  else if (solve_instationary_) {
    TimingScope tscope("Solve_system_instationary");
    
    if ( !corotFormValue ) { // Linear Elasticity Formulation.
      
      // Re-initialize "local_asm":
      InstationaryLinElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_, dampingFactor_, rayleighAlpha_, rayleighBeta_);
      // TASK: implement this element-specifically!
      
      // Initialize timestepping
      local_asm.set_timestep_parameters(theta_, delta_t_);
      local_asm.set_time(time_);
      local_asm.set_prev_solution(&sol_prev_);
      local_asm.set_dt_prev_solution(&dt_sol_prev_);
      local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
      
      // Compute Newmark Constants.
      double Nalpha0 = local_asm.get_Newmark_TimeIntegration_Constants(0);
      double Nalpha2 = local_asm.get_Newmark_TimeIntegration_Constants(2);
      double Nalpha3 = local_asm.get_Newmark_TimeIntegration_Constants(3);
      double Nalpha6 = local_asm.get_Newmark_TimeIntegration_Constants(6);
      double Nalpha7 = local_asm.get_Newmark_TimeIntegration_Constants(7);
      
      // Solve linear system.
      if (solver_name_ == "CG") {
        cg_.Solve(rhs_, &sol_);
      } else /*if (solver_name_ == "GMRES")*/ {
        gmres_.Solve(rhs_, &sol_);
      }
      // The gmres_.Solve(...) does not need be updated with the matrix_ in every timestep, since matrix_ is constant in Newmark.
      // In the Newmark Algorithm the system matrix K_eff does not change during the timesteps (as opposed to R_eff which does).
      
      sol_.UpdateCouplings();
      
      // Compute accelerations and velocities for current timestep (dt_sol_ and dt2_sol_) based on results from previous timestep:
      // (Updates from Newmark Algorithm):
      compute_dt2_sol(Nalpha0, Nalpha2, Nalpha3, sol_, sol_prev_, dt_sol_prev_, dt2_sol_prev_, &dt2_sol_);
      dt2_sol_.UpdateCouplings();
      compute_dt_sol(Nalpha6, Nalpha7, dt_sol_prev_, dt2_sol_prev_, dt2_sol_, &dt_sol_);
      dt_sol_.UpdateCouplings();
      
      sol_prev_.CloneFrom(sol_);
      dt_sol_prev_.CloneFrom(dt_sol_);
      dt2_sol_prev_.CloneFrom(dt2_sol_);
      
      sol_prev_.UpdateCouplings();
      dt_sol_prev_.UpdateCouplings();
      dt2_sol_prev_.UpdateCouplings();
      
    } // end of Linear Formulation.
    else { // Corotational Formulation.
      
      // Re-initialize "local_asm":
      InstationaryCorotElasticityAssembler local_asm(lambda_, mu_, rho_, gravity_, dampingFactor_, rayleighAlpha_, rayleighBeta_); 
      // TASK: implement this element-specifically!
      
      // Initialize timestepping
      local_asm.set_timestep_parameters(theta_, delta_t_);
      local_asm.set_time(time_);
      local_asm.set_prev_solution(&sol_prev_);
      local_asm.set_dt_prev_solution(&dt_sol_prev_);
      local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
      
      // Compute Newmark Constants.
      double Nalpha0 = local_asm.get_Newmark_TimeIntegration_Constants(0);
      double Nalpha2 = local_asm.get_Newmark_TimeIntegration_Constants(2);
      double Nalpha3 = local_asm.get_Newmark_TimeIntegration_Constants(3);
      double Nalpha6 = local_asm.get_Newmark_TimeIntegration_Constants(6);
      double Nalpha7 = local_asm.get_Newmark_TimeIntegration_Constants(7);
      
      // Solve linear system.
      if (solver_name_ == "CG") {
        cg_.Solve(rhs_, &sol_);
      } else /*if (solver_name_ == "GMRES")*/ {
        gmres_.Solve(rhs_, &sol_);
      }
      // The gmres_.Solve(...) does not need be updated with the matrix_ in every timestep, since matrix_ is constant in Newmark.
      // In the Newmark Algorithm the system matrix K_eff does not change during the timesteps (as opposed to R_eff which does).
      
      sol_.UpdateCouplings();
      
      // Compute accelerations and velocities for current timestep (dt_sol_ and dt2_sol_) based on results from previous timestep:
      // (Updates from Newmark Algorithm):
      compute_dt2_sol(Nalpha0, Nalpha2, Nalpha3, sol_, sol_prev_, dt_sol_prev_, dt2_sol_prev_, &dt2_sol_);
      dt2_sol_.UpdateCouplings();
      compute_dt_sol(Nalpha6, Nalpha7, dt_sol_prev_, dt2_sol_prev_, dt2_sol_, &dt_sol_);
      dt_sol_.UpdateCouplings();
      
      sol_prev_.CloneFrom(sol_);
      dt_sol_prev_.CloneFrom(dt_sol_);
      dt2_sol_prev_.CloneFrom(dt2_sol_);
      
      sol_prev_.UpdateCouplings();
      dt_sol_prev_.UpdateCouplings();
      dt2_sol_prev_.UpdateCouplings();
      
    } // end of Corotational Formulation.
    
//      // Initialize timestepping
//      local_asm.set_timestep_parameters(theta_, delta_t_);
//      local_asm.set_time(time_);
//      local_asm.set_prev_solution(&sol_prev_);
//      local_asm.set_dt_prev_solution(&dt_sol_prev_);
//      local_asm.set_dt2_prev_solution(&dt2_sol_prev_);
//      
//      // Compute Newmark Constants.
//      double Nalpha0 = local_asm.get_Newmark_TimeIntegration_Constants(0);
//      double Nalpha2 = local_asm.get_Newmark_TimeIntegration_Constants(2);
//      double Nalpha3 = local_asm.get_Newmark_TimeIntegration_Constants(3);
//      double Nalpha6 = local_asm.get_Newmark_TimeIntegration_Constants(6);
//      double Nalpha7 = local_asm.get_Newmark_TimeIntegration_Constants(7);
//      
//      // Solve linear system.
//      if (solver_name_ == "CG") {
//         cg_.Solve(rhs_, &sol_);
//      } else /*if (solver_name_ == "GMRES")*/ {
//         gmres_.Solve(rhs_, &sol_);
//      }
//      // The gmres_.Solve(...) does not need be updated with the matrix_ in every timestep, since matrix_ is constant in Newmark.
//      // In the Newmark Algorithm the system matrix K_eff does not change during the timesteps (as opposed to R_eff which does).
//      
//      sol_.UpdateCouplings();
//      
//      // Compute accelerations and velocities for current timestep (dt_sol_ and dt2_sol_) based on results from previous timestep:
//      // (Updates from Newmark Algorithm):
//      compute_dt2_sol(Nalpha0, Nalpha2, Nalpha3, sol_, sol_prev_, dt_sol_prev_, dt2_sol_prev_, &dt2_sol_);
//      dt2_sol_.UpdateCouplings();
//      compute_dt_sol(Nalpha6, Nalpha7, dt_sol_prev_, dt2_sol_prev_, dt2_sol_, &dt_sol_);
//      dt_sol_.UpdateCouplings();
//      
//      sol_prev_.CloneFrom(sol_);
//      dt_sol_prev_.CloneFrom(dt_sol_);
//      dt2_sol_prev_.CloneFrom(dt2_sol_);
//      
//      sol_prev_.UpdateCouplings();
//      dt_sol_prev_.UpdateCouplings();
//      dt2_sol_prev_.UpdateCouplings();
//      
//     // sol_.Zeros(); // not needed, since "Solver" computes sol_ newly in every solution step.
//     // dt_sol_.Zeros(); // not needed in case it is done in the compute_dt_sol()-function.
//     // dt2_sol_.Zeros(); // not needed in case it is done in the compute_dt2_sol()-function.
  }
  
  if (solver_name_ == "CG") {
    if (matrix_precond_ == 0) {
      CONSOLE_OUTPUT(3, "Linear solver (CG) not using preconditioner." );
    }
    else {
      if ( matrix_precond_ == 3 ) {
	CONSOLE_OUTPUT(3, "Linear solver (CG) using the SymmetricGaussSeidel preconditioner." );
      } else {
        CONSOLE_OUTPUT(3, "Linear solver (CG) using the following preconditioner: " << matrix_precond_ << "." );
        CONSOLE_OUTPUT(3, "Note: Jacobi = 1, GaussSeidel = 2, SymmetricGaussSeidel = 3, SOR = 4, SSOR = 5, ILU = 6, ILU2 = ILU_pp = 7, ILU_P = 8." );
      }
    }
    CONSOLE_OUTPUT(3, "Linear solver (CG) computed solution in " << cg_.iter() << " iterations."); //<< cg_.num_iter() << " iterations.");
    CONSOLE_OUTPUT(3, "Residual norm for solution = " << cg_.res());
  } else {
    if (matrix_precond_ == 0) {
      CONSOLE_OUTPUT(3, "Linear solver (GMRES) not using preconditioner." );
    }
    else {
      CONSOLE_OUTPUT(3, "Linear solver (GMRES) using the following preconditioner: " << matrix_precond_ << "." );
      CONSOLE_OUTPUT(3, "Note: Jacobi = 1, GaussSeidel = 2, SymmetricGaussSeidel = 3, SOR = 4, SSOR = 5, ILU = 6, ILU2 = ILU_pp = 7, ILU_P = 8." );
    }
    CONSOLE_OUTPUT(3, "Linear solver (GMRES) computed solution in " << gmres_.iter() << " iterations.");
    CONSOLE_OUTPUT(3, "Residual norm for solution = " << gmres_.res());
  }
}

void Elasticity::compute_dt2_sol(const double Nalpha0, const double Nalpha2, const double Nalpha3, 
				 const CVector& sol_, const CVector& sol_prev_, const CVector& dt_sol_prev_, 
				 const CVector& dt2_sol_prev_, CVector* dt2_sol_) {
   
   double negNalpha0 = -1 * Nalpha0;
   double negNalpha2 = -1 * Nalpha2;
   double negNalpha3 = -1 * Nalpha3;
   
   // Initialize call-by-reference "return" vector:
   dt2_sol_->Zeros();
   
   // Compute dt2_sol_ according to Newmark Algorithm Step B.3.1:
   //dt2_sol_ = Nalpha0 * (sol_ - sol_prev_) - Nalpha2 * dt_sol_prev_ - Nalpha3 * dt2_sol_prev_;
   //dt2_sol_ = Nalpha0 * sol_ - Nalpha0 * sol_prev_ - Nalpha2 * dt_sol_prev_ - Nalpha3 * dt2_sol_prev_;
   dt2_sol_->Axpy(sol_, Nalpha0);
   dt2_sol_->Axpy(sol_prev_, negNalpha0);
   dt2_sol_->Axpy(dt_sol_prev_, negNalpha2);
   dt2_sol_->Axpy(dt2_sol_prev_, negNalpha3);
   
   dt2_sol_->UpdateCouplings(); // TASK: update dampingFactor_ if dampingFactor_ != 1.0;
}

void Elasticity::compute_dt_sol(const double Nalpha6, const double Nalpha7, 
				const CVector& dt_sol_prev_, const CVector& dt2_sol_prev_, 
				const CVector& dt2_sol_, CVector* dt_sol_) {
   
   // Initialize call-by-reference "return" vector:
   dt_sol_->Zeros();
   
   // Compute dt_sol_ according to Newmark Algorithm Step B.3.2:
   // dt_sol_ = dt_sol_prev_ + Nalpha6_ * dt2_sol_prev_ + Nalpha7_ * dt2_sol_;
   dt_sol_->Axpy(dt_sol_prev_, 1);
   dt_sol_->Axpy(dt2_sol_prev_, Nalpha6);
   dt_sol_->Axpy(dt2_sol_, Nalpha7);
   
   dt_sol_->UpdateCouplings(); // TASK: update dampingFactor_ if dampingFactor_ != 1.0;
}

void Elasticity::calc_volume () {
  TimingScope tscope("initial volume calculation");
  
  mesh::AttributePtr subdomain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
  
  std::vector<Coordinate> vertices;
  
  // Counters
  double total_local_volume = 0.;
  double volume_curr_entity = 0.;
  int c; // dof adjuster variable.
  
  EntityIterator entity = mesh_->begin(DIMENSION);
  
  if (entity->num_vertices() == 4) { // if (mesh elements == tetrahedra)
    
    double triple[3][3];
    vertices.reserve(4*DIMENSION); // 12 = 4 (nodes) times 3 (coords/DIMENSION). // needed?!
    
    // loop over all elements (that are not ghost cells):
    for (EntityIterator tetrahedron = mesh_->begin(DIMENSION), tetrahedron_end = mesh_->end(DIMENSION);
	 tetrahedron != tetrahedron_end; ++tetrahedron) {
      if (subdomain_attr->get_int_value(tetrahedron->index()) == rank_) {
	
	tetrahedron->get_coordinates(vertices);
	
        for (int i = 0; i < 3; ++i) {
          c = 0;
          for (int j = 0; j < 3; ++j) {
            c += 3; 
            triple[i][j] = vertices[c+i] - vertices[i];
          }
        }
        
        volume_curr_entity = ( triple[0][0]*triple[1][1]*triple[2][2] 
                         + triple[0][1]*triple[1][2]*triple[2][0] 
                         + triple[0][2]*triple[1][0]*triple[2][1] 
                         - triple[0][2]*triple[1][1]*triple[2][0] 
                         - triple[0][1]*triple[1][0]*triple[2][2] 
                         - triple[0][0]*triple[1][2]*triple[2][1] )/6.0;
        
	volume_curr_entity = fabs(volume_curr_entity);
	
        // compute total local-mesh volume on current process:
        total_local_volume += volume_curr_entity;
      }
    }
  
  } // end if (entity->num_vertices() == 4).
  else { // if (entity->num_vertices() == 8). // if (mesh elements == cuboids)
    
    double quadruple[3][4];
    vertices.reserve (8*DIMENSION); // 24 = 8 (nodes) times 3 (coords/DIMENSION). // needed?!
    
    // loop over all elements (that are not ghost cells):
    for (EntityIterator cuboid = mesh_->begin(DIMENSION), cuboid_end = mesh_->end(DIMENSION);
	 cuboid != cuboid_end; ++cuboid) {
      if (subdomain_attr->get_int_value(cuboid->index()) == rank_) {
	
        cuboid->get_coordinates(vertices);
       
        for (int i = 0; i < 3; ++i) {
          c = 0;
          for (int j = 0; j < 4; ++j) {
            c += 3;
            quadruple[i][j] = vertices[c+i] - vertices[i];
          }
        }
       
        volume_curr_entity = quadruple[0][0]*quadruple[1][1]*quadruple[2][3]
                           + quadruple[0][1]*quadruple[1][3]*quadruple[2][0]
                           + quadruple[0][3]*quadruple[1][0]*quadruple[2][1]
                           - quadruple[0][3]*quadruple[1][1]*quadruple[2][0]
                           - quadruple[0][1]*quadruple[1][0]*quadruple[2][3]
                           - quadruple[0][0]*quadruple[1][3]*quadruple[2][1];
                           
        volume_curr_entity = fabs(volume_curr_entity);
        
        if (volume_curr_entity < 1.0e-15) { // i.e. if ( first three points are linearly dependent, thus, lie in one plane)
          
          volume_curr_entity = quadruple[0][0]*quadruple[1][1]*quadruple[2][2]
                             + quadruple[0][1]*quadruple[1][2]*quadruple[2][0]
                             + quadruple[0][2]*quadruple[1][0]*quadruple[2][1]
                             - quadruple[0][2]*quadruple[1][1]*quadruple[2][0]
                             - quadruple[0][1]*quadruple[1][0]*quadruple[2][2]
                             - quadruple[0][0]*quadruple[1][2]*quadruple[2][1]; 
                           
          volume_curr_entity = fabs(volume_curr_entity);
        }
        
        // compute total local-mesh volume on current process:
        total_local_volume += volume_curr_entity;
      }
    }
    
    std::cout << " not the case of cuboids, on process " << rank_ << std::endl;
  }
  
  // Reduce all of the local volumes into the global volume:
  // MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  MPI_Reduce(&total_local_volume, &initial_global_volume, 1, MPI_DOUBLE, MPI_SUM, 0, comm_ );
  
}

void Elasticity::calc_def_volume() {
  TimingScope tscope("timestep volume calculation");
  // NOTE: works for tet4-elements only.
  
  // ------------------------------------------------------------------------------------
  // Set up new mesh-pointer w.r.t. the current timestep's deformed state of mesh:
  
  // therefore, copy the last timestep's solution-vector 
  // (i.e. the last timestep's displacement w.r.t. the initial state):
  std::vector<double> temp_sol_vec(sol_.size_global(), 0.0);
  std::vector<int> temp_dof_ids;
  std::vector<double> temp_sol_values;
  sol_.GetAllDofsAndValues(temp_dof_ids, temp_sol_values);
  for (int i = 0; i < temp_sol_values.size(); ++i) {
    temp_sol_vec.at(temp_dof_ids[i]) = temp_sol_values.at(i);
    // temp_sol_values.at(i) corresponds to displacement values.
    // Note: temp_sol_vec does not contain the body's deformed state
    // (the deformed state would correspond to orig values + displacement values)
  }
  
  // Create vector that contains the displacement which is set onto the mesh/geometry
  // in order to yield the deformed state:
  std::vector<double> displacement_vector(DIMENSION * mesh_->num_entities(0));
  
  std::vector<int> local_dof_id;
  
  AttributePtr sub_domain_attr;
  // if attribute _sub_domain_ exists, there are several subdomains, and hence ghost cells, too.
  const bool has_ghost_cells = mesh_->has_attribute("_sub_domain_", mesh_->tdim());
  if (has_ghost_cells) {
      sub_domain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
  }
  
  // loop over all mesh cells
  for (EntityIterator cell = mesh_->begin(mesh_->tdim()), end_c = mesh_->end(mesh_->tdim());
       cell != end_c; ++cell) {
    
    int vertex_num = 0;
    
    if (has_ghost_cells && sub_domain_attr->get_int_value(cell->index()) != rank_) {
	continue; // in order not to write values twice/thrice/... for parallel processes
    }
    
    // loop over all vertices (incident_dim = 0) in current mesh cell
    for (IncidentEntityIterator vertex_it = cell->begin_incident(0); 
	 vertex_it != cell->end_incident(0); ++vertex_it) {
      
      // loop over DIMENSION
      for (int var = 0; var < DIMENSION; ++var) {
	
	// get local_dof_id w.r.t. vertex_num and var
	space_.dof().get_dofs_on_subentity(var, cell->index(), 0, vertex_num, local_dof_id);
	displacement_vector.at(3 * vertex_it->index() + var) = temp_sol_vec.at(local_dof_id[0]);
	// this for-loop writes the values from "visu_vec" (which contains the solution values (u1,u2,u3))
	// into the "displacement_vector", which then (below) is given to mesh_->move_vertices().
      }
      
      ++vertex_num;
    } // end of loop over all vertices of current mesh cell.
    
  } // end of loop over all mesh cells.
  
  // finally, move vertices of master_mesh_copy:
  mesh_->move_vertices( displacement_vector );
  
  // -----------------------------------------------------------------------------------
  // Now, the mesh is updated to the current timestep's deformed state, 
  // thus, the computation of the mesh's volume can begin here (just as it was done initially, see above):
  
  mesh::AttributePtr subdomain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
  
  std::vector<Coordinate> vertices;
  
  // Counters
  double total_local_volume = 0.;
  double volume_curr_entity = 0.;
  int c; // dof adjuster variable.
  
  EntityIterator entity = mesh_->begin(DIMENSION);
  
  if (entity->num_vertices() == 4) { // if (mesh elements == tetrahedra)
    
    double triple[3][3];
    vertices.reserve(4*DIMENSION); // 12 = 4 (nodes) times 3 (coords/DIMENSION). // needed?!
    
    // loop over all elements (that are not ghost cells):
    for (EntityIterator tetrahedron = mesh_->begin(DIMENSION), tetrahedron_end = mesh_->end(DIMENSION);
	 tetrahedron != tetrahedron_end; ++tetrahedron) {
      if (subdomain_attr->get_int_value(tetrahedron->index()) == rank_) {
	
	tetrahedron->get_coordinates(vertices);
	
        for (int i = 0; i < 3; ++i) {
          c = 0;
          for (int j = 0; j < 3; ++j) {
            c += 3; 
            triple[i][j] = vertices[c+i] - vertices[i];
          }
        }
        
        volume_curr_entity = ( triple[0][0]*triple[1][1]*triple[2][2] 
                         + triple[0][1]*triple[1][2]*triple[2][0] 
                         + triple[0][2]*triple[1][0]*triple[2][1] 
                         - triple[0][2]*triple[1][1]*triple[2][0] 
                         - triple[0][1]*triple[1][0]*triple[2][2] 
                         - triple[0][0]*triple[1][2]*triple[2][1] )/6.0;
        
	volume_curr_entity = fabs(volume_curr_entity);
	
        // compute total local-mesh volume on current process:
        total_local_volume += volume_curr_entity;
      }
    }
  
  } // end if (entity->num_vertices() == 4).
  else { // if (entity->num_vertices() == 8). // if (mesh elements == cuboids)
    
    double quadruple[3][4];
    vertices.reserve (8*DIMENSION); // 24 = 8 (nodes) times 3 (coords/DIMENSION). // needed?!
    
    // loop over all elements (that are not ghost cells):
    for (EntityIterator cuboid = mesh_->begin(DIMENSION), cuboid_end = mesh_->end(DIMENSION);
	 cuboid != cuboid_end; ++cuboid) {
      if (subdomain_attr->get_int_value(cuboid->index()) == rank_) {
	
        cuboid->get_coordinates(vertices);
       
        for (int i = 0; i < 3; ++i) {
          c = 0;
          for (int j = 0; j < 4; ++j) {
            c += 3;
            quadruple[i][j] = vertices[c+i] - vertices[i];
          }
        }
       
        volume_curr_entity = quadruple[0][0]*quadruple[1][1]*quadruple[2][3]
                           + quadruple[0][1]*quadruple[1][3]*quadruple[2][0]
                           + quadruple[0][3]*quadruple[1][0]*quadruple[2][1]
                           - quadruple[0][3]*quadruple[1][1]*quadruple[2][0]
                           - quadruple[0][1]*quadruple[1][0]*quadruple[2][3]
                           - quadruple[0][0]*quadruple[1][3]*quadruple[2][1];
                           
        volume_curr_entity = fabs(volume_curr_entity);
        
        if (volume_curr_entity < 1.0e-15) { // i.e. if ( first three points are linearly dependent, thus, lie in one plane)
          
          volume_curr_entity = quadruple[0][0]*quadruple[1][1]*quadruple[2][2]
                             + quadruple[0][1]*quadruple[1][2]*quadruple[2][0]
                             + quadruple[0][2]*quadruple[1][0]*quadruple[2][1]
                             - quadruple[0][2]*quadruple[1][1]*quadruple[2][0]
                             - quadruple[0][1]*quadruple[1][0]*quadruple[2][2]
                             - quadruple[0][0]*quadruple[1][2]*quadruple[2][1]; 
                           
          volume_curr_entity = fabs(volume_curr_entity);
        }
        
        // compute total local-mesh volume on current process:
        total_local_volume += volume_curr_entity;
      }
    }
    
  } // end elseif (entity->num_vertices() == 8).
  
  // Reduce all of the local volumes into the global volume:
  // MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
  MPI_Reduce(&total_local_volume, &current_timesteps_global_volume, 1, MPI_DOUBLE, MPI_SUM, 0, comm_ );
  
  // ------------------------------------------------------------------------------------
  // Re-transform the mesh, by means of subtracting the displacement_vector again:
  // the reason for the need of doing this: disp_t+1  = sol_(delta_t_) and NOT disp_t+1 - disp_t = sol_(delta_t_).
  for (int i = 0; i < displacement_vector.size(); ++i) { // temp_sol_vec *= -1.0; // componentwise...
        displacement_vector[i] = -1.0 * displacement_vector[i];
  }
  mesh_->move_vertices( displacement_vector );
  
  // ------------------------------------------------------------------------------------
}

void Elasticity::visualize(/* int ts_*/) {
    TimingScope tscope("Visualization");

//     Visualization visu; // Old Visualization (for old code working on HiFlow3-Releases <= 1.3).
//     visu.set_mesh(mesh_.get());
//     visu.set_dof(&space_.dof());
//     visu.set_fe_manager(&space_.fe_manager());
//     visu.set_var_names(visu_names_);
//     visu.set_visualize_all_attributes(true);
    
    // Setup visualization object.
    int num_intervals = u_deg; // num_intervals for CellVisualization is reasonably less or equal the FE degree.
    ParallelCellVisualization<double> visu(space_, num_intervals, comm_, MASTER_RANK);
    
  if( extVis ) { //////////////////////////////////////////////////////// in case of extended visualization:
    // Generate filename for mesh which includes solution-array.
    std::stringstream input;
    input << simul_name_ << "_solution";
    
    const int xml_init_ref_lvl = params_["Mesh"]["InitialRefLevel"].get<int>();
    
    input << "_np" << num_partitions() << "_RefLvl" << xml_init_ref_lvl << "_Tstep";
    
    if (solve_instationary_) {
        if (ts_ < 10)
            input << ".000" << ts_;
        else if (ts_ < 100)
            input << ".00" << ts_;
        else if  (ts_ < 1000)
            input << ".0" << ts_;
        else
            input << "." << ts_;
    } else {
        input << "_stationary";
    }
    
    if (num_partitions() > 1)
        input << ".pvtu";
    else
        input << ".vtu";
    
    std::vector<double> remote_index(mesh_->num_entities(mesh_->tdim()), 0);
    std::vector<double> sub_domain(mesh_->num_entities(mesh_->tdim()), 0);
    std::vector<double> material_number(mesh_->num_entities(mesh_->tdim()), 0);
    
    for (mesh::EntityIterator it = mesh_->begin(mesh_->tdim()); it != mesh_->end(mesh_->tdim()); ++it)
    {
    	int temp1, temp2;
	double temp00, temp01, temp02;
    	mesh_->get_attribute_value ( "_remote_index_", mesh_->tdim(),
    			it->index(),
    			&temp1);
    	mesh_->get_attribute_value ( "_sub_domain_", mesh_->tdim(),
    			it->index(),
    			&temp2);
    	remote_index.at(it->index()) = temp1;
    	sub_domain.at(it->index()) = temp2;
    	material_number.at(it->index()) = mesh_->get_material_number(mesh_->tdim(), it->index());
    }
    
    sol_.UpdateCouplings();
    
    visu.visualize(EvalFeFunction<LAD>(space_, sol_, 0), "u0");
    visu.visualize(EvalFeFunction<LAD>(space_, sol_, 1), "u1");
    visu.visualize(EvalFeFunction<LAD>(space_, sol_, 2), "u2");
    
    visu.visualize_cell_data(material_number, "Material Id");
    visu.visualize_cell_data(remote_index, "_remote_index_");
    visu.visualize_cell_data(sub_domain, "_sub_domain_");
    visu.write(input.str());

    // Create visualization helper vector from solution vector sol_.
    std::vector<LAD::DataType> visu_vec(sol_.size_global(), 1.e20);
    std::vector<int> dof_ids;
    std::vector<double> values;
    sol_.GetAllDofsAndValues(dof_ids, values); // solution values (u1,u2,u3) from "sol_" are given into the "visu" object.
    for (int i = 0; i < values.size(); ++i) {
        visu_vec.at(dof_ids[i]) = values.at(i); // values.at(i) correspond to displacement values (as in (pp_)sol_).
	// Note: visu_vec does not contain the body's deformed state
	// (the deformed state would correspond to orig values + displacement values)
    }

    // Visualize the Original mesh AND the solution (displacement) vector (u1, u2, u3) AND all other attributes (...).
//     visu.Visualize(visu_vec);

    // ---------------------------------------------------------
    // START THE ADDITIONAL DEFORMATION VISUALIZATION PART HERE:
    // Compute "deformed state"
    // as sum of initial coords (= mesh input) and displacement values (= solution vector u).
    
    std::vector<double> deformation_vector(3 * mesh_->num_entities(0)); // This vector contains the mesh/geometry in deformed state.
    
    std::vector<int> local_dof_id;
    
    AttributePtr sub_domain_attr;
    // if attribute _sub_domain_ exists, there are several subdomains, and hence ghost cells, too.
    const bool has_ghost_cells = mesh_->has_attribute("_sub_domain_", mesh_->tdim());
    if (has_ghost_cells) {
      sub_domain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
    }
    
    for (EntityIterator cell = mesh_->begin(mesh_->tdim()), end_c = mesh_->end(mesh_->tdim()); cell != end_c; ++cell) {
      // optimized writing for: 
      // for (EntityIterator cell = mesh_->begin(mesh_->tdim()); cell != mesh_->end(mesh_->tdim()); ++cell).
      int vertex_num = 0;
      
      if (has_ghost_cells && sub_domain_attr->get_int_value(cell->index()) != rank_) {
	continue; // in order not to write values twice/thrice/... for parallel processes
      }
      
      for (IncidentEntityIterator vertex_it = cell->begin_incident(0); vertex_it != cell->end_incident(0); ++vertex_it) {
      //for (int vertex = 0; vertex < cell->num_vertices(); ++vertex) {
	for (int var = 0; var < 3; ++var) {
	  space_.dof().get_dofs_on_subentity(var, cell->index(), 0, vertex_num, local_dof_id);
	  deformation_vector.at(3 * vertex_it->index() + var) = visu_vec.at(local_dof_id[0]);
	  // this for-loop writes the values from "visu_vec" (which contains the solution values (u1,u2,u3))
	  // into the "deformation_vector", which then (below) is given into the (p)vtk-writer,
	  // which in turn "sets the deformation" into the original mesh by means of "vertex[c] += deformation_->at(...c...)";
	  // (as adapted in the hiflow3-src visualization() and writer()).
	}
	++vertex_num;
      }
    }
    
    // ---------------------------------------------------------------------------------
    // Based on the "deformation_vector", compute the Mises Stress here in the following. // possible ToDo.
    // Alternatively, don't complicatedly compute it here, but let ParaView compute it instead!
    // ---------------------------------------------------------------------------------
    
    // Generate output filename and write output mesh/geometry of deformed state.
    std::stringstream input2;
    input2 << simul_name_ << "_deformedSolution";
    input2 << "_np" << num_partitions() << "_RefLvl" << xml_init_ref_lvl << "_Tstep"; // was before: "refinement_level_"
    
    if (solve_instationary_) {
        if (ts_ < 10)
            input2 << ".000" << ts_;
        else if (ts_ < 100)
            input2 << ".00" << ts_;
        else if  (ts_ < 1000)
            input2 << ".0" << ts_;
        else
            input2 << "." << ts_;
    } else {
        input2 << "_stationary";
    }
    
    if (num_partitions() == 1) {
      input2 << ".pvtu"; // should be ".vtu", yet this does not work. so, we always (even for np=1) produce a .pvtu AND a .vtu file.
      PVtkWriter deformation_writer(comm_);
      deformation_writer.set_deformation(&deformation_vector);
      deformation_writer.add_all_attributes(*mesh_, true);
      deformation_writer.write(input2.str().c_str(), *mesh_); // input2.str().c_str() instead of "filename.vtu"
      
    } else if (num_partitions() > 1) {
      // compare following code part to "HiFlow/scr/space/visualize.cc"
      mesh::MeshPtr mesh_ptr;
      mesh::TDim tdim = mesh_->tdim();
      
      if (mesh_->has_attribute("_sub_domain_", tdim)) {
	// create new mesh without ghost layers
	mesh::MeshDbViewBuilder builder((static_cast<mesh::MeshDbView*>(mesh_.get()))->get_db());
	
	// cell index map
	std::vector<int> index_map;
	
	// get __remote_indices__ attribute
	mesh::AttributePtr subdomain_attr = mesh_->get_attribute("_sub_domain_", tdim);
	
	// loop over cells that are not ghosts
	for(mesh::EntityIterator cell_it = mesh_->begin(tdim); cell_it != mesh_->end(tdim); ++cell_it) {
	  if (subdomain_attr->get_int_value(cell_it->index()) == rank_) {
	    // loop over vertices of cell
	    std::vector<mesh::MeshBuilder::VertexHandle> vertex_handle;
	    for(mesh::IncidentEntityIterator inc_vert_it = cell_it->begin_incident(0); inc_vert_it != cell_it->end_incident(0); ++inc_vert_it) {
	      //vertex_handle.push_back(builder.add_vertex(inc_vert_it->get_coordinates())); // Note: "get_coordinates()" deleted during streamlining process.
	      std::vector<double> coord(3,0.0);
	      inc_vert_it->get_coordinates(coord);
	      vertex_handle.push_back(builder.add_vertex(coord));
	    }
	    builder.add_entity(tdim, vertex_handle);
	    index_map.push_back(cell_it->index());
	  }
	}
	
	mesh_ptr = builder.build();
	mesh::AttributePtr index_map_attr(new mesh::IntAttribute(index_map));
	mesh_ptr->add_attribute("__index_map__", mesh_ptr->tdim(), index_map_attr);
	
	// transfer cell attributes
	std::vector< std::string > cell_attr_names = mesh_->get_attribute_names(mesh_ptr->tdim());
	for (std::vector< std::string >::const_iterator it = cell_attr_names.begin(), 
	     end_it = cell_attr_names.end(); it != end_it; ++it) {
	  mesh::AttributePtr mapped_attr(
	    new mesh::InheritedAttribute(mesh_->get_attribute(*it, mesh_->tdim()), index_map_attr));
	  mesh_ptr->add_attribute(*it, mesh_->tdim(), mapped_attr);
	}
      } else {
	std::cout << "Error in Visualization Routine! Missing subdomain attribute!\n";
      }
      
      // Since deformation_vector still contains some vertices twice/thrice/... (according to num processes, and ghost cells),
      // every vertex is now (by means of an EntityIterator) set/mapped (once only) into a mapped_deformation_vector.
      // Thus, finally, the mapped_deformation_vector has the size 3*mesh_ptr->num_entities(0).
      std::vector<double> mapped_deformation_vector(3 * mesh_ptr->num_entities(0)); // dim = number of nodes in mesh * dimension
      for (EntityIterator v_it = mesh_ptr->begin(0), end_v = mesh_ptr->end(0); v_it != end_v; ++v_it) {
	    mesh::Id v_id = v_it->id();
	    mesh::EntityNumber v_index = -1;
	    bool found = mesh_->find_entity(0, v_id, &v_index);
	    if (!found) {
	      std::cout << "Error in Visualization Routine! Missing vertex!\n";
	    }
	    for (int c = 0; c < 3; ++c) {
	      mapped_deformation_vector[3 * v_it->index() + c] = deformation_vector[3 * v_index + c];
	    }
      }
      
      input2 << ".pvtu";
      PVtkWriter p_deformation_writer(comm_);
      p_deformation_writer.set_deformation(&mapped_deformation_vector);
      p_deformation_writer.add_all_attributes(*mesh_ptr, true);
      p_deformation_writer.write(input2.str().c_str(), *mesh_ptr); // input2.str().c_str() instead of "filename.vtu"
      
    } else {
        std::cout << "Error in Visualization Routine! Wrong counting and managing the number of partitions: num_partitions_ <= 0 is wrong." << std::endl;
    }
    
  } //////////////////////////////////////////////////////// end of in case of extended visualization:
  else { ///// in case of non-extended visualization:
    const int xml_init_ref_lvl = params_["Mesh"]["InitialRefLevel"].get<int>();
    
    std::vector<double> remote_index(mesh_->num_entities(mesh_->tdim()), 0);
    std::vector<double> sub_domain(mesh_->num_entities(mesh_->tdim()), 0);
    std::vector<double> material_number(mesh_->num_entities(mesh_->tdim()), 0);
    
    for (mesh::EntityIterator it = mesh_->begin(mesh_->tdim()); it != mesh_->end(mesh_->tdim()); ++it)
    {
    	int temp1, temp2;
	double temp00, temp01, temp02;
    	mesh_->get_attribute_value ( "_remote_index_", mesh_->tdim(),
    			it->index(),
    			&temp1);
    	mesh_->get_attribute_value ( "_sub_domain_", mesh_->tdim(),
    			it->index(),
    			&temp2);
    	remote_index.at(it->index()) = temp1;
    	sub_domain.at(it->index()) = temp2;
    	material_number.at(it->index()) = mesh_->get_material_number(mesh_->tdim(), it->index());
    }
    
    sol_.UpdateCouplings();
    
    // Create visualization helper vector from solution vector sol_.
    std::vector<LAD::DataType> visu_vec(sol_.size_global(), 1.e20);
    std::vector<int> dof_ids;
    std::vector<double> values;
    sol_.GetAllDofsAndValues(dof_ids, values); // solution values (u1,u2,u3) from "sol_" are given into the "visu" object.
    for (int i = 0; i < values.size(); ++i) {
        visu_vec.at(dof_ids[i]) = values.at(i); // values.at(i) correspond to displacement values (as in (pp_)sol_).
	// Note: visu_vec does not contain the body's deformed state
	// (the deformed state would correspond to orig values + displacement values)
    }
    
    
    // ---------------------------------------------------------
    // START THE ADDITIONAL DEFORMATION VISUALIZATION PART HERE:
    // Compute "deformed state"
    // as sum of initial coords (= mesh input) and displacement values (= solution vector u).
    
    std::vector<double> deformation_vector(3 * mesh_->num_entities(0)); // This vector contains the mesh/geometry in deformed state.
    
    std::vector<int> local_dof_id;
    
    AttributePtr sub_domain_attr;
    // if attribute _sub_domain_ exists, there are several subdomains, and hence ghost cells, too.
    const bool has_ghost_cells = mesh_->has_attribute("_sub_domain_", mesh_->tdim());
    if (has_ghost_cells) {
      sub_domain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
    }
    
    for (EntityIterator cell = mesh_->begin(mesh_->tdim()), end_c = mesh_->end(mesh_->tdim()); cell != end_c; ++cell) {
      // optimized writing for: 
      // for (EntityIterator cell = mesh_->begin(mesh_->tdim()); cell != mesh_->end(mesh_->tdim()); ++cell).
      int vertex_num = 0;
      
      if (has_ghost_cells && sub_domain_attr->get_int_value(cell->index()) != rank_) {
	continue; // in order not to write values twice/thrice/... for parallel processes
      }
      
      for (IncidentEntityIterator vertex_it = cell->begin_incident(0); vertex_it != cell->end_incident(0); ++vertex_it) {
      //for (int vertex = 0; vertex < cell->num_vertices(); ++vertex) {
	for (int var = 0; var < 3; ++var) {
	  space_.dof().get_dofs_on_subentity(var, cell->index(), 0, vertex_num, local_dof_id);
	  deformation_vector.at(3 * vertex_it->index() + var) = visu_vec.at(local_dof_id[0]);
	  // this for-loop writes the values from "visu_vec" (which contains the solution values (u1,u2,u3))
	  // into the "deformation_vector", which then (below) is given into the (p)vtk-writer,
	  // which in turn "sets the deformation" into the original mesh by means of "vertex[c] += deformation_->at(...c...)";
	  // (as adapted in the hiflow3-src visualization() and writer()).
	}
	++vertex_num;
      }
    }
    
    // ---------------------------------------------------------------------------------
    // Based on the "deformation_vector", compute the Mises Stress here in the following. // possible ToDo.
    // Alternatively, don't complicatedly compute it here, but let ParaView compute it instead!
    // ---------------------------------------------------------------------------------
    
    // Generate output filename and write output mesh/geometry of deformed state.
    std::stringstream input2;
    input2 << simul_name_ << "_deformedSolution";
    input2 << "_np" << num_partitions() << "_RefLvl" << xml_init_ref_lvl << "_Tstep"; // was before: "refinement_level_"
    
    if (solve_instationary_) {
        if (ts_ < 10)
            input2 << ".000" << ts_;
        else if (ts_ < 100)
            input2 << ".00" << ts_;
        else if  (ts_ < 1000)
            input2 << ".0" << ts_;
        else
            input2 << "." << ts_;
    } else {
        input2 << "_stationary";
    }
    
    if (num_partitions() == 1) {
      input2 << ".pvtu"; // should be ".vtu", yet this does not work. so, we always (even for np=1) produce a .pvtu AND a .vtu file.
      PVtkWriter deformation_writer(comm_);
      deformation_writer.set_deformation(&deformation_vector);
      deformation_writer.add_all_attributes(*mesh_, true);
      deformation_writer.write(input2.str().c_str(), *mesh_); // input2.str().c_str() instead of "filename.vtu"
      
    } else if (num_partitions() > 1) {
      // compare following code part to "HiFlow/scr/space/visualize.cc"
      mesh::MeshPtr mesh_ptr;
      mesh::TDim tdim = mesh_->tdim();
      
      if (mesh_->has_attribute("_sub_domain_", tdim)) {
	// create new mesh without ghost layers
	mesh::MeshDbViewBuilder builder((static_cast<mesh::MeshDbView*>(mesh_.get()))->get_db());
	
	// cell index map
	std::vector<int> index_map;
	
	// get __remote_indices__ attribute
	mesh::AttributePtr subdomain_attr = mesh_->get_attribute("_sub_domain_", tdim);
	
	// loop over cells that are not ghosts
	for(mesh::EntityIterator cell_it = mesh_->begin(tdim); cell_it != mesh_->end(tdim); ++cell_it) {
	  if (subdomain_attr->get_int_value(cell_it->index()) == rank_) {
	    // loop over vertices of cell
	    std::vector<mesh::MeshBuilder::VertexHandle> vertex_handle;
	    for(mesh::IncidentEntityIterator inc_vert_it = cell_it->begin_incident(0); inc_vert_it != cell_it->end_incident(0); ++inc_vert_it) {
	      //vertex_handle.push_back(builder.add_vertex(inc_vert_it->get_coordinates())); // Note: "get_coordinates()" deleted during streamlining process.
	      std::vector<double> coord(3,0.0);
	      inc_vert_it->get_coordinates(coord);
	      vertex_handle.push_back(builder.add_vertex(coord));
	    }
	    builder.add_entity(tdim, vertex_handle);
	    index_map.push_back(cell_it->index());
	  }
	}
	
	mesh_ptr = builder.build();
	mesh::AttributePtr index_map_attr(new mesh::IntAttribute(index_map));
	mesh_ptr->add_attribute("__index_map__", mesh_ptr->tdim(), index_map_attr);
	
	// transfer cell attributes
	std::vector< std::string > cell_attr_names = mesh_->get_attribute_names(mesh_ptr->tdim());
	for (std::vector< std::string >::const_iterator it = cell_attr_names.begin(), 
	     end_it = cell_attr_names.end(); it != end_it; ++it) {
	  mesh::AttributePtr mapped_attr(
	    new mesh::InheritedAttribute(mesh_->get_attribute(*it, mesh_->tdim()), index_map_attr));
	  mesh_ptr->add_attribute(*it, mesh_->tdim(), mapped_attr);
	}
      } else {
	std::cout << "Error in Visualization Routine! Missing subdomain attribute!\n";
      }
      
      // Since deformation_vector still contains some vertices twice/thrice/... (according to num processes, and ghost cells),
      // every vertex is now (by means of an EntityIterator) set/mapped (once only) into a mapped_deformation_vector.
      // Thus, finally, the mapped_deformation_vector has the size 3*mesh_ptr->num_entities(0).
      std::vector<double> mapped_deformation_vector(3 * mesh_ptr->num_entities(0)); // dim = number of nodes in mesh * dimension
      for (EntityIterator v_it = mesh_ptr->begin(0), end_v = mesh_ptr->end(0); v_it != end_v; ++v_it) {
	    mesh::Id v_id = v_it->id();
	    mesh::EntityNumber v_index = -1;
	    bool found = mesh_->find_entity(0, v_id, &v_index);
	    if (!found) {
	      std::cout << "Error in Visualization Routine! Missing vertex!\n";
	    }
	    for (int c = 0; c < 3; ++c) {
	      mapped_deformation_vector[3 * v_it->index() + c] = deformation_vector[3 * v_index + c];
	    }
      }
      
      input2 << ".pvtu";
      PVtkWriter p_deformation_writer(comm_);
      p_deformation_writer.set_deformation(&mapped_deformation_vector);
      p_deformation_writer.add_all_attributes(*mesh_ptr, true);
      p_deformation_writer.write(input2.str().c_str(), *mesh_ptr); // input2.str().c_str() instead of "filename.vtu"
      
    } else {
        std::cout << "Error in Visualization Routine! Wrong counting and managing the number of partitions: num_partitions_ <= 0 is wrong." << std::endl;
    }
    
  } ////// end of non-extended visualization.
    
/*    
    // ---------------------------------------------------------
    // START THE ADDITIONAL DEFORMATION VISUALIZATION PART HERE:
    // Compute "deformed state"
    // as sum of initial coords (= mesh input) and displacement values (= solution vector u).
    
    std::vector<double> deformation_vector(3 * mesh_->num_entities(0)); // This vector contains the mesh/geometry in deformed state.
    
    std::vector<int> local_dof_id;
    
    AttributePtr sub_domain_attr;
    // if attribute _sub_domain_ exists, there are several subdomains, and hence ghost cells, too.
    const bool has_ghost_cells = mesh_->has_attribute("_sub_domain_", mesh_->tdim());
    if (has_ghost_cells) {
      sub_domain_attr = mesh_->get_attribute("_sub_domain_", mesh_->tdim());
    }
    
    for (EntityIterator cell = mesh_->begin(mesh_->tdim()), end_c = mesh_->end(mesh_->tdim()); cell != end_c; ++cell) {
      // optimized writing for: 
      // for (EntityIterator cell = mesh_->begin(mesh_->tdim()); cell != mesh_->end(mesh_->tdim()); ++cell).
      int vertex_num = 0;
      
      if (has_ghost_cells && sub_domain_attr->get_int_value(cell->index()) != rank_) {
	continue; // in order not to write values twice/thrice/... for parallel processes
      }
      
      for (IncidentEntityIterator vertex_it = cell->begin_incident(0); vertex_it != cell->end_incident(0); ++vertex_it) {
      //for (int vertex = 0; vertex < cell->num_vertices(); ++vertex) {
	for (int var = 0; var < 3; ++var) {
	  space_.dof().get_dofs_on_subentity(var, cell->index(), 0, vertex_num, local_dof_id);
	  deformation_vector.at(3 * vertex_it->index() + var) = visu_vec.at(local_dof_id[0]);
	  // this for-loop writes the values from "visu_vec" (which contains the solution values (u1,u2,u3))
	  // into the "deformation_vector", which then (below) is given into the (p)vtk-writer,
	  // which in turn "sets the deformation" into the original mesh by means of "vertex[c] += deformation_->at(...c...)";
	  // (as adapted in the hiflow3-src visualization() and writer()).
	}
	++vertex_num;
      }
    }
    
    // ---------------------------------------------------------------------------------
    // Based on the "deformation_vector", compute the Mises Stress here in the following. // possible ToDo.
    // Alternatively, don't complicatedly compute it here, but let ParaView compute it instead!
    // ---------------------------------------------------------------------------------
    
    // Generate output filename and write output mesh/geometry of deformed state.
    std::stringstream input2;
    input2 << simul_name_ << "_deformedSolution";
    input2 << "_np" << num_partitions() << "_RefLvl" << xml_init_ref_lvl << "_Tstep"; // was before: "refinement_level_"
    
    if (solve_instationary_) {
        if (ts_ < 10)
            input2 << ".000" << ts_;
        else if (ts_ < 100)
            input2 << ".00" << ts_;
        else if  (ts_ < 1000)
            input2 << ".0" << ts_;
        else
            input2 << "." << ts_;
    } else {
        input2 << "_stationary";
    }
    
    if (num_partitions() == 1) {
      input2 << ".pvtu"; // should be ".vtu", yet this does not work. so, we always (even for np=1) produce a .pvtu AND a .vtu file.
      PVtkWriter deformation_writer(comm_);
      deformation_writer.set_deformation(&deformation_vector);
      deformation_writer.add_all_attributes(*mesh_, true);
      deformation_writer.write(input2.str().c_str(), *mesh_); // input2.str().c_str() instead of "filename.vtu"
      
    } else if (num_partitions() > 1) {
      // compare following code part to "HiFlow/scr/space/visualize.cc"
      mesh::MeshPtr mesh_ptr;
      mesh::TDim tdim = mesh_->tdim();
      
      if (mesh_->has_attribute("_sub_domain_", tdim)) {
	// create new mesh without ghost layers
	mesh::MeshDbViewBuilder builder((static_cast<mesh::MeshDbView*>(mesh_.get()))->get_db());
	
	// cell index map
	std::vector<int> index_map;
	
	// get __remote_indices__ attribute
	mesh::AttributePtr subdomain_attr = mesh_->get_attribute("_sub_domain_", tdim);
	
	// loop over cells that are not ghosts
	for(mesh::EntityIterator cell_it = mesh_->begin(tdim); cell_it != mesh_->end(tdim); ++cell_it) {
	  if (subdomain_attr->get_int_value(cell_it->index()) == rank_) {
	    // loop over vertices of cell
	    std::vector<mesh::MeshBuilder::VertexHandle> vertex_handle;
	    for(mesh::IncidentEntityIterator inc_vert_it = cell_it->begin_incident(0); inc_vert_it != cell_it->end_incident(0); ++inc_vert_it) {
	      //vertex_handle.push_back(builder.add_vertex(inc_vert_it->get_coordinates())); // Note: "get_coordinates()" deleted during streamlining process.
	      std::vector<double> coord(3,0.0);
	      inc_vert_it->get_coordinates(coord);
	      vertex_handle.push_back(builder.add_vertex(coord));
	    }
	    builder.add_entity(tdim, vertex_handle);
	    index_map.push_back(cell_it->index());
	  }
	}
	
	mesh_ptr = builder.build();
	mesh::AttributePtr index_map_attr(new mesh::IntAttribute(index_map));
	mesh_ptr->add_attribute("__index_map__", mesh_ptr->tdim(), index_map_attr);
	
	// transfer cell attributes
	std::vector< std::string > cell_attr_names = mesh_->get_attribute_names(mesh_ptr->tdim());
	for (std::vector< std::string >::const_iterator it = cell_attr_names.begin(), 
	     end_it = cell_attr_names.end(); it != end_it; ++it) {
	  mesh::AttributePtr mapped_attr(
	    new mesh::InheritedAttribute(mesh_->get_attribute(*it, mesh_->tdim()), index_map_attr));
	  mesh_ptr->add_attribute(*it, mesh_->tdim(), mapped_attr);
	}
      } else {
	std::cout << "Error in Visualization Routine! Missing subdomain attribute!\n";
      }
      
      // Since deformation_vector still contains some vertices twice/thrice/... (according to num processes, and ghost cells),
      // every vertex is now (by means of an EntityIterator) set/mapped (once only) into a mapped_deformation_vector.
      // Thus, finally, the mapped_deformation_vector has the size 3*mesh_ptr->num_entities(0).
      std::vector<double> mapped_deformation_vector(3 * mesh_ptr->num_entities(0)); // dim = number of nodes in mesh * dimension
      for (EntityIterator v_it = mesh_ptr->begin(0), end_v = mesh_ptr->end(0); v_it != end_v; ++v_it) {
	    mesh::Id v_id = v_it->id();
	    mesh::EntityNumber v_index = -1;
	    bool found = mesh_->find_entity(0, v_id, &v_index);
	    if (!found) {
	      std::cout << "Error in Visualization Routine! Missing vertex!\n";
	    }
	    for (int c = 0; c < 3; ++c) {
	      mapped_deformation_vector[3 * v_it->index() + c] = deformation_vector[3 * v_index + c];
	    }
      }
      
      input2 << ".pvtu";
      PVtkWriter p_deformation_writer(comm_);
      p_deformation_writer.set_deformation(&mapped_deformation_vector);
      p_deformation_writer.add_all_attributes(*mesh_ptr, true);
      p_deformation_writer.write(input2.str().c_str(), *mesh_ptr); // input2.str().c_str() instead of "filename.vtu"
      
    } else {
        std::cout << "Error in Visualization Routine! Wrong counting and managing the number of partitions: num_partitions_ <= 0 is wrong." << std::endl;
    }
    */
    
    
}

void Elasticity::setup_linear_algebra() {
    
    TimingScope tscope("setup_linear_algebra");
    
    const std::string platform_str = params_["LinearAlgebra"]["Platform"].get<std::string>();
    if (platform_str == "CPU") {
        la_sys_.Platform = CPU;
    } else if (platform_str == "GPU") {
        la_sys_.Platform = GPU;
    } else {
        throw UnexpectedParameterValue("LinearAlgebra.Platform", platform_str);
    }
    init_platform(la_sys_);

    const std::string impl_str = params_["LinearAlgebra"]["Implementation"].get<std::string>();
    if (impl_str == "Naive") {
        la_impl_ = NAIVE;
    } else if (impl_str == "BLAS") {
        la_impl_ = BLAS;
    } else if (impl_str == "MKL") {
        la_impl_ = MKL;
    } else if (impl_str == "OPENMP") {
        la_impl_ = OPENMP;
    } else if (impl_str == "SCALAR") {
        la_impl_ = SCALAR;
    } else if (impl_str == "SCALAR_TEX") {
        la_impl_ = SCALAR_TEX;
    } else {
        throw UnexpectedParameterValue("LinearAlgebra.Implementation", impl_str);
    }

    const std::string matrix_str = params_["LinearAlgebra"]["MatrixFormat"].get<std::string>();
    if (matrix_str == "CSR") {
        la_matrix_format_ = CSR;
    } else if (matrix_str == "COO") {
        la_matrix_format_ = COO;
    } else {
        throw UnexpectedParameterValue("LinearAlgebra.MatrixFormat", impl_str);
    }
    
    // the following part is needed to initialize class member "matrix_precond_".
    const std::string precond_str = params_["LinearSolver"]["PreconditionerName"].get<std::string>();
    if (precond_str == "NOPRECOND") {
      matrix_precond_ = NOPRECOND;
    } else if (precond_str == "JACOBI") {
      matrix_precond_ = JACOBI;
    } else if (precond_str == "GAUSS_SEIDEL") {
      matrix_precond_ = GAUSS_SEIDEL;
    } else if (precond_str == "SGAUSS_SEIDEL") {
      matrix_precond_ = SGAUSS_SEIDEL;
    } else if (precond_str == "SOR") {
      matrix_precond_ = SOR;
    } else if (precond_str == "SSOR") {
      matrix_precond_ = SSOR;
    } else if (precond_str == "ILU") {
      matrix_precond_ = ILU;
    } else if (precond_str == "ILU2") {
      matrix_precond_ = ILU2; // ILU2 = ILUpp;
    }/* else if (precond_str == "ILU_p") { // TODO TODO TODO ILU_p not defined ?!?!?
      matrix_precond_ = ILU_p;
    }*/ else {
      throw UnexpectedParameterValue("LinearSolver.PreconditionerName", precond_str);
    }
}
