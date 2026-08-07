#include <vector>
#include <iostream>
#include <fstream>

#include <Epetra_RowMatrixTransposer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "parameters/all_parameters.h"

#include "dg/dg.h"
#include "mesh_error_estimate.h"
#include "functional/functional.h"
#include "physics/physics.h"
#include "linear_solver/linear_solver.h"
#include "post_processor/physics_post_processor.h"
#include "physics/physics_factory.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: ~MeshErrorEstimateBase(){}

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: MeshErrorEstimateBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
    {}

template <int dim, typename real, typename MeshType>
ResidualErrorEstimate<dim, real, MeshType> :: ResidualErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
    {}

template <int dim, typename real, typename MeshType>
dealii::Vector<real> ResidualErrorEstimate<dim, real, MeshType> :: compute_cellwise_errors()
{
    std::vector<dealii::types::global_dof_index> dofs_indices;
    dealii::Vector<real> cellwise_errors (this->dg->high_order_grid->triangulation->n_active_cells());
    this->dg->assemble_residual();

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;

        const int i_fele = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
        const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();
        dofs_indices.resize(n_dofs_cell);
        cell->get_dof_indices (dofs_indices);
        real max_residual = 0;
        for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) 
        {
            const unsigned int index = dofs_indices[idof];
            const real res = std::abs(this->dg->right_hand_side[index]);
            if (res > max_residual) 
                max_residual = res;
        }
        cellwise_errors[cell->active_cell_index()] = max_residual;
    }

    return cellwise_errors;
}

template <int dim, int nstate, typename real, typename MeshType>
LESErrorEstimate<dim, nstate, real, MeshType> :: LESErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input, const Parameters::MeshAdaptationParam *const mesh_adaptation_param_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
    , solution_coarse(this->dg->solution)
    , solution_refinement_state(SolutionRefinementStateEnum::coarse)
    , mesh_adaptation_param(mesh_adaptation_param_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    {}


template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> LESErrorEstimate<dim, nstate, real, MeshType> :: compute_cellwise_errors()
{
    // Declare physics pointer --> should probably initialize this in class constructor like dg pointer then pass it through
    std::shared_ptr<PHiLiP::Physics::NavierStokes<dim,nstate, real> > navier_stokes_physics;

    // Initialize (copied from periodic_turbulence line 49)
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = *(this->dg->all_parameters);
    parameters_navier_stokes.pde_type = PDE_enum::navier_stokes;
    navier_stokes_physics = std::dynamic_pointer_cast<PHiLiP::Physics::NavierStokes<dim, nstate, real>>(
                PHiLiP::Physics::PhysicsFactory<dim,nstate,real>::create_Physics(&parameters_navier_stokes));
    reinit();
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);
    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    
    std::cout << "n_active_cells at adjoint_residual construction: " 
          << this->dg->triangulation->n_active_cells() << std::endl;
    dealii::Vector<real> adjoint_residual(this->dg->triangulation->n_active_cells());
    this->dg->assemble_residual(); //assemble residual of projected mesh

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators())
    {
        if(!cell->is_locally_owned()) continue;
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        const int poly_degree = cell->active_fe_index();
        //const int n_dofs = cell->get_fe().n_dofs_per_cell();
    
       const unsigned int n_quad_pts  = this->dg->volume_quadrature_collection[poly_degree].size();

        const unsigned int n_shape_fns = n_dofs_curr_cell / nstate;


        std::array<std::vector<real>,nstate> soln_coeff;
        std::array<std::vector<real>,nstate> rhs_coeff;
        const unsigned int init_grid_degree = this->dg->high_order_grid->fe_system.tensor_degree();
        dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),poly_degree);
        OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, init_grid_degree); 
        OPERATOR::vol_projection_operator<dim,2*dim> soln_basis_projection_oper(1, poly_degree, this->dg->max_grid_degree);
        soln_basis.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], vol_quad_equidistant_1D);
        soln_basis.build_1D_gradient_operator(this->dg->oneD_fe_collection_1state[poly_degree], vol_quad_equidistant_1D); 
        soln_basis_projection_oper.build_1D_volume_operator(this->dg->oneD_fe_collection_1state[poly_degree], vol_quad_equidistant_1D);

        for (unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof) {
            const unsigned int istate = (cell->get_fe().system_to_component_index(idof)).first;
            const unsigned int ishape = (cell->get_fe().system_to_component_index(idof)).second;
            // allocate
            if(ishape == 0){
                soln_coeff[istate].resize(n_shape_fns);
                rhs_coeff[istate].resize(n_shape_fns);
            }
            // solve
            soln_coeff[istate][ishape] = this->dg->solution(current_dofs_indices[idof]);
            rhs_coeff[istate][ishape] = this->dg->right_hand_side(current_dofs_indices[idof]);

            //project onto quadrature points
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);

                // Interpolate soln coeff to volume cubature nodes.
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                                soln_basis.oneD_vol_operator);
            }
        
        std::vector<std::array<real,nstate>> entropy_var_at_q(n_quad_pts);
      
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad)
        {
            
            //std::array<real, nstate> entropy_variables;
            std::array<real, nstate> solution;
            for (unsigned int istate=0; istate<nstate; ++istate)
            {
                solution[istate] = soln_at_q[istate][iquad];
            }
            entropy_var_at_q[iquad] = navier_stokes_physics->compute_entropy_variables(solution);

            
        }
        std::array<std::vector<real>,nstate> entropy_var_coeff;
        std::array<std::vector<real>,nstate> entropy_var_at_q_bystate;
        for (unsigned int istate=0; istate<nstate; ++istate) {
            entropy_var_at_q_bystate[istate].resize(n_quad_pts);
            entropy_var_coeff[istate].resize(n_shape_fns);
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                entropy_var_at_q_bystate[istate][iquad] = entropy_var_at_q[iquad][istate];
            }
            soln_basis_projection_oper.matrix_vector_mult_1D(entropy_var_at_q_bystate[istate],
                                                            entropy_var_coeff[istate],
                                                            soln_basis_projection_oper.oneD_vol_operator);
        }
        auto print_array = [](const auto &arr) {
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < arr.size(); ++i) {
            os << arr[i] << (i + 1 < arr.size() ? ", " : "");
        }
        os << "]";
        return os.str();
        };
        real adjoint_residual_sum = 0.0;
        for (unsigned int ishape=0; ishape<n_shape_fns; ++ishape)
        {
     
            std::vector<real> product_at_shape_fns(n_shape_fns);
            std::array<real,nstate> entropy_var_at_shape, rhs_at_shape;

            for (unsigned int istate=0; istate<nstate; ++istate) {
                entropy_var_at_shape[istate] = entropy_var_coeff[istate][ishape];
                rhs_at_shape[istate] = rhs_coeff[istate][ishape];
            }
            
            product_at_shape_fns[ishape] = std::inner_product(entropy_var_at_shape.begin(), entropy_var_at_shape.end(), rhs_at_shape.begin(), 0.0); 
            adjoint_residual_sum += product_at_shape_fns[ishape];

            std::cout << "adjoint_residual.size()=" << adjoint_residual.size()
          << " active_cell_index=" << cell->active_cell_index()
          << " triangulation n_active_cells=" << this->dg->triangulation->n_active_cells()
          << " adjoint_residual=" << adjoint_residual[cell->active_cell_index()]
          << " rhs_at_shape=" << print_array(rhs_at_shape)
          << " entropy_var_at_shape=" << print_array(entropy_var_at_shape)
          << std::endl;
        }
        adjoint_residual[cell->active_cell_index()] = std::abs(adjoint_residual_sum);
        
    }
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse); 
    pcout<<"computed adjoint residual"<<std::endl;
    return adjoint_residual;
    // once we have dg->solution at a node, we can mimic line 642 pm periodic_turbulence.cpp
    
    /*
    //the below code computes the unsteady residual of value epsilon=(Res(P_{p+1}[Q_p])-P_{p+1}[Res(Q_p)])
    //auto Q_p = this->dg->solution; //save original solution
    //compute residual at p+1
    reinit();
    this->dg->assemble_residual();

    //required variables to calculate P_{p+1}[Res(Q_p)]
    //std::vector<std::vector<real>> p_order_residual(this->dg->triangulation->n_active_cells());
    std::vector<std::vector<real>> projected_residual(this->dg->triangulation->n_active_cells());

    //record average solution per state in each cell for normalization
    std::vector<real> sum_per_state(nstate, 0.0);
    int dofs_per_state = (this->dg->dof_handler.locally_owned_dofs().size() / nstate);

    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);

    // cell loop to project the residual to p+1 and obtain P_{p+1}[Res(Q_p)]
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;
        

        if ((fe_index_curr_cell+1) == this->dg->all_parameters->flow_solver_param.max_poly_degree_for_adaptation) continue;
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);
        std::vector<real> p_order_residual(n_dofs_curr_cell);
       
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            p_order_residual[idof] = this->dg->right_hand_side[current_dofs_indices[idof]];
            sum_per_state[(cell->get_fe().system_to_component_index(idof)).first] += std::abs(this->dg->solution[current_dofs_indices[idof]]); //calculate time average solution for normalization
        }
         
        //gather inputs for project_function(), and then project the rhs of active cell to p+1
        const int poly_degree = cell->active_fe_index();
        //out <<"current cell: "<<cell->active_cell_index()<< "poly_degree: " << static_cast<unsigned int>(poly_degree) << std::endl;
        const dealii::FESystem<dim,dim> &fe_input = this->dg->fe_collection[poly_degree];
        const dealii::FESystem<dim,dim> &fe_output = this->dg->fe_collection[poly_degree + 1];  
        const dealii::QGauss <dim> projection_quadrature(fe_index_curr_cell + 2); //notation is +2 to account for Gauss 2n-1 rule
        //std::vector<real> p_order_residual_per_cell = p_order_residual[cell->active_cell_index()];

        projected_residual[cell->active_cell_index()] = project_function(p_order_residual, fe_input, fe_output, projection_quadrature); 
        
    }    
    // add sum_per_state across all MPI ranks
    MPI_Allreduce(MPI_IN_PLACE, sum_per_state.data(), nstate, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //Project mesh to p+1 to compute Res(P_{p+1}[Q_p])
    //reinit(); //do we need this?? maybe for residual vector. remove
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);
    this->dg->assemble_residual(); //assemble residual of projected mesh
    dealii::Vector<real> unsteady_residual(this->dg->triangulation->n_active_cells());

    
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        if ((fe_index_curr_cell+1) == this->dg->all_parameters->flow_solver_param.max_poly_degree_for_adaptation)
        {   unsteady_residual[cell->active_cell_index()]= 0.0;
            continue; }
        

        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        // check sizes are consistent
        Assert(projected_residual[cell->active_cell_index()].size() == n_dofs_curr_cell,
        dealii::ExcMessage("projected_residual size mismatch after mesh refinement"));

        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);
        // compute epsilon=(Res(P_{p+1}[Q_p])-P_{p+1}[Res(Q_p)])
        std::vector<real> residual_per_state_per_cell(nstate, 0.0);

        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            const real rhs_cell = this->dg->right_hand_side[current_dofs_indices[idof]] - projected_residual[cell->active_cell_index()][idof];
            std::pair<unsigned int, unsigned int> state_and_node = cell->get_fe().system_to_component_index(idof);
            //<pcout<"current state: "<<(state_and_node.first)<<"; current residual_per_state_per_cell: "<<residual_per_state_per_cell[state_and_node.first]<<std::endl;
            residual_per_state_per_cell[state_and_node.first] += std::abs(rhs_cell);
        }
        
        //normalize the solution at each state
        real total_residual_momentum = 0.0;
        real dofs_momentum = 0;
        real sum_momentum = 0.0;
      
        for (unsigned int state = 1; state < (nstate-1); ++state)
        {
            sum_momentum += sum_per_state[state];
        }

        //combine residual of all momentum states
        for (unsigned int state = 1; state < (nstate-1); ++state)
        {
            total_residual_momentum += residual_per_state_per_cell[state];
            dofs_momentum += dofs_per_state;
        } 
        real residual_mass = residual_per_state_per_cell[0]*dofs_per_state/sum_per_state[0];
        real residual_energy = residual_per_state_per_cell[(nstate - 1)]*dofs_per_state/sum_per_state[(nstate - 1)];
    
        //unsteady_residual[cell->active_cell_index()] = residual_mass + (total_residual_momentum*dofs_momentum/sum_momentum) + residual_energy;
        const auto state = this->mesh_adaptation_param->indicator_state;
        if (nstate > 1)
        {   
            if (state == Parameters::MeshAdaptationParam::x_momentum)
                {unsteady_residual[cell->active_cell_index()] = residual_per_state_per_cell[1]*dofs_per_state/sum_per_state[1];}

            else if (state == Parameters::MeshAdaptationParam::all)
                {unsteady_residual[cell->active_cell_index()] = residual_mass + (total_residual_momentum*dofs_momentum/sum_momentum) + residual_energy;}
            else if (state == Parameters::MeshAdaptationParam::energy)
                {unsteady_residual[cell->active_cell_index()] = residual_energy;}
            else if (state == Parameters::MeshAdaptationParam::mass)
                {unsteady_residual[cell->active_cell_index()] = residual_mass;}
            else if (state == Parameters::MeshAdaptationParam::y_momentum)
                {if (nstate > 2)
                    {unsteady_residual[cell->active_cell_index()] = residual_per_state_per_cell[2]*dofs_per_state/sum_per_state[2];}}
            else if (state == Parameters::MeshAdaptationParam::x_y_momentum)
                {if (nstate > 2)
                    {unsteady_residual[cell->active_cell_index()] = residual_per_state_per_cell[1]*dofs_per_state/sum_per_state[1] + residual_per_state_per_cell[2]*dofs_per_state/sum_per_state[2];}}
            else if (state == Parameters::MeshAdaptationParam::z_momentum)
                {if (nstate > 3)
                    {unsteady_residual[cell->active_cell_index()] = residual_per_state_per_cell[3]*dofs_per_state/sum_per_state[3];}}
            else if (state == Parameters::MeshAdaptationParam::momentum)
                {unsteady_residual[cell->active_cell_index()] = total_residual_momentum*dofs_momentum/sum_momentum;}
        }
        else {
            unsteady_residual[cell->active_cell_index()] = residual_per_state_per_cell[0]*dofs_per_state/sum_per_state[0];
        }
 
        pcout<<"residual mass: "<<residual_mass<<"; residual momentum: "<<(total_residual_momentum*dofs_momentum/sum_momentum)<<"; residual energy: "<<residual_energy<<std::endl;
        //pcout<<"UNSTEADY RESIDUAL: "<<unsteady_residual[cell->active_cell_index()]<<std::endl;
    }
    
    
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);  // restore mesh TURNED OFF FOR TROUBLESHOOTING

    //this->dg->solution = Q_p; //restore solution vector 
    return unsteady_residual;
    */
}

template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::reinit()
{
    // reinitilizing all variables after triangulation in the constructor
    solution_coarse = this->dg->solution;
    solution_refinement_state = SolutionRefinementStateEnum::coarse;

    // storing the original FE degree distribution
    coarse_fe_index.reinit(this->dg->triangulation->n_active_cells());
    
    // looping over the cells
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(cell->is_locally_owned())
        {
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum required_refinement_state)
{   
    // checks if conversion is needed
    //pcout<<"SolutionRefinementStateEnum: "<<solution_refinement_state<<std::endl;
    //pcout<<"Required refinement state: "<<required_refinement_state<<std::endl;
    if(solution_refinement_state == required_refinement_state)
    {
        return;
    }
    // calls corresponding function for state conversions
    else if(solution_refinement_state == SolutionRefinementStateEnum::coarse && required_refinement_state == SolutionRefinementStateEnum::fine)
    {
        coarse_to_fine();
    }
    
    else if(solution_refinement_state == SolutionRefinementStateEnum::fine && required_refinement_state == SolutionRefinementStateEnum::coarse)
    {
        fine_to_coarse();
    }
    else
    {
        pcout<<"Invalid state. Aborting.."<<std::endl;
        std::abort();
    }
}


template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::coarse_to_fine()
{
    if (this->dg->get_max_fe_degree() >= this->dg->max_degree) 
    {
        pcout<<"Polynomial degree of DG will exceed the maximum allowable after refinement. Update max_degree in dg"<<std::endl;
        std::abort();
    } 
    
    [[maybe_unused]] unsigned int no_of_cells_before_changing_p = this->dg->triangulation->n_active_cells(); // used in debug mode (in assert).  

    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs =  this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);
    //pcout<<"locally_owned_dofs.size()="<<locally_owned_dofs.size()<<std::endl;
   // pcout<<"locally_relevant_dofs.size()="<<locally_relevant_dofs.size()<<std::endl;
    solution_coarse.update_ghost_values();
    
    // Solution Transfer to fine grid
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(cell->active_fe_index()+1);
        }
    }

    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    this->dg->allocate_system(false, false, false);
    this->dg->solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, 
                                 decltype(solution_transfer)>) {
        solution_transfer.interpolate(solution_coarse, this->dg->solution);
    } else {
        solution_transfer.interpolate(this->dg->solution);
    }
    
    this->dg->solution.update_ghost_values();
    
    [[maybe_unused]] unsigned int no_of_cells_after_changing_p = this->dg->triangulation->n_active_cells(); // It's used when compiled in debug mode (in assert). 

    AssertDimension(no_of_cells_before_changing_p, no_of_cells_after_changing_p);

    solution_refinement_state = SolutionRefinementStateEnum::fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::fine_to_coarse()
{
    [[maybe_unused]] unsigned int no_of_cells_before_changing_p = this->dg->triangulation->n_active_cells(); // Used in assert (i.e remains unused in Release mode).
    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(coarse_fe_index[cell->active_cell_index()]);
        }
    }
    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();
    this->dg->allocate_system(false, false, false);
    this->dg->solution.zero_out_ghosts();

    this->dg->solution = solution_coarse;
    
    [[maybe_unused]] unsigned int no_of_cells_after_changing_p = this->dg->triangulation->n_active_cells(); // Used when compiled in debug mode (in assert).

    AssertDimension(no_of_cells_before_changing_p, no_of_cells_after_changing_p);

    solution_refinement_state = SolutionRefinementStateEnum::coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle, const dealii::Vector <real> &cellwise_errors)
{
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(this->dg->dof_handler);

    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(this->dg->all_parameters);
    data_out.add_data_vector(this->dg->solution, *post_processor);

    dealii::Vector<float> subdomain(this->dg->triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) 
    {
        subdomain(i) = this->dg->triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    //get parameter compute_cellwise_errors
    
    //output error estimate
    //dealii::Vector<real> error_estimate = compute_cellwise_errors();
    //cellwise_errors = meshadaptation->cellwise_errors;
    data_out.add_data_vector(cellwise_errors, "error_estimate", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    this->dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(this->dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // set names of data to be output in the vtu file.
    std::vector<std::string> derivative_functional_wrt_solution_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "derivative_functional_wrt_solution" + dealii::Utilities::int_to_string(s,1);
        derivative_functional_wrt_solution_names.push_back(varname);
    }

        //process and finalize the data
    const dealii::Mapping<dim> &mapping = (*(this->dg->high_order_grid->mapping_fe_field));
    const int n_subdivisions = this->dg->get_max_fe_degree();
    data_out.build_patches(mapping, n_subdivisions, 
        dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells);
   

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::string filename_prefix = "LESErrorEstimate";

    // vtu filename
    std::string filename = this->dg->all_parameters->solution_vtk_files_directory_name 
                        + "/" + filename_prefix + "-" 
                        + dealii::Utilities::int_to_string(dim, 1) + "D-";
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            // must match vtu filename exactly
            std::string fn = filename_prefix + "-" 
                        + dealii::Utilities::int_to_string(dim, 1) + "D-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        // pvtu master file
        std::string master_fn = this->dg->all_parameters->solution_vtk_files_directory_name 
                            + "/" + filename_prefix + "-" 
                            + dealii::Utilities::int_to_string(dim, 1) + "D-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }
    

}


template <int dim, typename real, typename MeshType>
ExplicitErrorEstimate<dim, real, MeshType> :: ExplicitErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
    {}


template <int dim, typename real, typename MeshType>
dealii::Vector<real> ExplicitErrorEstimate<dim, real, MeshType> :: compute_cellwise_errors()
{
    std::vector<dealii::types::global_dof_index> dofs_indices;
    dealii::Vector<real> cellwise_errors (this->dg->high_order_grid->triangulation->n_active_cells());
    this->dg->assemble_residual();

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;

        if(cell->active_cell_index()%2 == 0){
            cellwise_errors[cell->active_cell_index()] = 1.0;
        }else{
            cellwise_errors[cell->active_cell_index()] = 0;
        }
    }

    return cellwise_errors;
}

// constructor
template <int dim, int nstate, typename real, typename MeshType>
DualWeightedResidualError<dim, nstate, real, MeshType>::DualWeightedResidualError(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim,real,MeshType> (dg_input) 
    , solution_coarse(this->dg->solution)
    , solution_refinement_state(SolutionRefinementStateEnum::coarse)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    Assert(this->dg->triangulation->get_mesh_smoothing() == typename dealii::Triangulation<dim>::MeshSmoothing(dealii::Triangulation<dim>::none), 
           dealii::ExcMessage("Mesh smoothing might h-refine cells while computing the dual weighted residual."));
    // storing the original FE degree distribution
    coarse_fe_index.reinit(this->dg->triangulation->n_active_cells());

    // create functional
    functional = FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(this->dg->all_parameters->functional_param, this->dg);

    // looping over the cells
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(cell->is_locally_owned())
        {
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
real DualWeightedResidualError<dim, nstate, real, MeshType>::total_dual_weighted_residual_error()
{
    dealii::Vector<real> cellwise_errors = compute_cellwise_errors();
    real error_sum = cellwise_errors.l1_norm();
    return dealii::Utilities::MPI::sum(error_sum, mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::compute_cellwise_errors()
{
    dealii::Vector<real> cellwise_errors(this->dg->triangulation->n_active_cells());
    reinit();
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);
    pcout<<"Computing fine grid adjoint..."<<std::endl;
    fine_grid_adjoint();
    pcout<<"Computing dual weighted residual..."<<std::endl;
    cellwise_errors = dual_weighted_residual();
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);

    pcout<<"Done computing the goal oriented error indicator."<<std::endl;
    return cellwise_errors;
}


template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::reinit()
{
    // reinitilizing all variables after triangulation in the constructor
    solution_coarse = this->dg->solution;
    solution_refinement_state = SolutionRefinementStateEnum::coarse;

    // storing the original FE degree distribution
    coarse_fe_index.reinit(this->dg->triangulation->n_active_cells());
    
    // looping over the cells
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(cell->is_locally_owned())
        {
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
        }
    }

    // for remaining, clear the values
    derivative_functional_wrt_solution_fine      = dealii::LinearAlgebra::distributed::Vector<real>();
    derivative_functional_wrt_solution_coarse    = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_fine   = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_coarse = dealii::LinearAlgebra::distributed::Vector<real>();

    dual_weighted_residual_fine = dealii::Vector<real>();
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum required_refinement_state)
{   
    // checks if conversion is needed
    if(solution_refinement_state == required_refinement_state)
    {
        return;
    }
    // calls corresponding function for state conversions
    else if(solution_refinement_state == SolutionRefinementStateEnum::coarse && required_refinement_state == SolutionRefinementStateEnum::fine)
    {
        coarse_to_fine();
    }
    
    else if(solution_refinement_state == SolutionRefinementStateEnum::fine && required_refinement_state == SolutionRefinementStateEnum::coarse)
    {
        fine_to_coarse();
    }
    else
    {
        pcout<<"Invalid state. Aborting.."<<std::endl;
        std::abort();
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::coarse_to_fine()
{
    if (this->dg->get_max_fe_degree() >= this->dg->max_degree) 
    {
        pcout<<"Polynomial degree of DG will exceed the maximum allowable after refinement. Update max_degree in dg"<<std::endl;
        std::abort();
    }
    
    [[maybe_unused]] unsigned int no_of_cells_before_changing_p = this->dg->triangulation->n_active_cells(); // used in debug mode (in assert).  

    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs =  this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);

    solution_coarse.update_ghost_values();
    
    // Solution Transfer to fine grid
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(cell->active_fe_index()+1);
        }
    }

    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, 
                                 decltype(solution_transfer)>) {
        solution_transfer.interpolate(solution_coarse, this->dg->solution);
    } else {
        solution_transfer.interpolate(this->dg->solution);
    }
    
    this->dg->solution.update_ghost_values();
    
    [[maybe_unused]] unsigned int no_of_cells_after_changing_p = this->dg->triangulation->n_active_cells(); // It's used when compiled in debug mode (in assert). 

    AssertDimension(no_of_cells_before_changing_p, no_of_cells_after_changing_p);

    solution_refinement_state = SolutionRefinementStateEnum::fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::fine_to_coarse()
{
    [[maybe_unused]] unsigned int no_of_cells_before_changing_p = this->dg->triangulation->n_active_cells(); // Used in assert (i.e remains unused in Release mode).
    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(coarse_fe_index[cell->active_cell_index()]);
        }
    }

    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    this->dg->solution = solution_coarse;
    
    [[maybe_unused]] unsigned int no_of_cells_after_changing_p = this->dg->triangulation->n_active_cells(); // Used when compiled in debug mode (in assert).

    AssertDimension(no_of_cells_before_changing_p, no_of_cells_after_changing_p);

    solution_refinement_state = SolutionRefinementStateEnum::coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::fine_grid_adjoint()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);

    adjoint_fine = compute_adjoint(derivative_functional_wrt_solution_fine, adjoint_fine);

    return adjoint_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::coarse_grid_adjoint()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);

    adjoint_coarse = compute_adjoint(derivative_functional_wrt_solution_coarse, adjoint_coarse);

    return adjoint_coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>
::compute_adjoint(dealii::LinearAlgebra::distributed::Vector<real> &derivative_functional_wrt_solution, 
                  dealii::LinearAlgebra::distributed::Vector<real> &adjoint_variable)
{
    derivative_functional_wrt_solution.reinit(this->dg->solution);
    adjoint_variable.reinit(this->dg->solution);
    
    const bool compute_derivative_functional_wrt_solution = true, compute_derivative_functional_wrt_grid_dofs = false;
    const real functional_value = functional->evaluate_functional(compute_derivative_functional_wrt_solution, compute_derivative_functional_wrt_grid_dofs);
    (void) functional_value;
    derivative_functional_wrt_solution = functional->dIdw;
    derivative_functional_wrt_solution.update_ghost_values();


    this->dg->assemble_residual(true);
    
    AssertDimension(derivative_functional_wrt_solution.size(), adjoint_variable.size());
    AssertDimension(this->dg->system_matrix_transpose.n(), adjoint_variable.size());
   
    solve_linear(this->dg->system_matrix_transpose, derivative_functional_wrt_solution, adjoint_variable, this->dg->all_parameters->linear_solver_param);
    adjoint_variable *= -1.0;
    
    adjoint_variable.compress(dealii::VectorOperation::add);
    adjoint_variable.update_ghost_values();

    return adjoint_variable;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::dual_weighted_residual()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);

    // allocate 
    dual_weighted_residual_fine.reinit(this->dg->triangulation->n_active_cells());

    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);

    // compute the error indicator cell-wise by taking the dot product over the DOFs with the residual vector
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;
        
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        real dwr_cell = 0;
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            dwr_cell += this->dg->right_hand_side[current_dofs_indices[idof]]*adjoint_fine[current_dofs_indices[idof]];
        }

        dual_weighted_residual_fine[cell->active_cell_index()] = std::abs(dwr_cell);
    }

    return dual_weighted_residual_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle)
{
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(this->dg->dof_handler);

    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(this->dg->all_parameters);
    data_out.add_data_vector(this->dg->solution, *post_processor);

    dealii::Vector<float> subdomain(this->dg->triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) 
    {
        subdomain(i) = this->dg->triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    this->dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(this->dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // set names of data to be output in the vtu file.
    std::vector<std::string> derivative_functional_wrt_solution_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "derivative_functional_wrt_solution" + dealii::Utilities::int_to_string(s,1);
        derivative_functional_wrt_solution_names.push_back(varname);
    }

    std::vector<std::string> adjoint_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1);
        adjoint_names.push_back(varname);
    }

    // add the data structures specific to this class, check if currently fine or coarse
    if(solution_refinement_state == SolutionRefinementStateEnum::fine) {
        data_out.add_data_vector(derivative_functional_wrt_solution_fine, derivative_functional_wrt_solution_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_fine, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

        data_out.add_data_vector(dual_weighted_residual_fine, "DWR", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    } else if(solution_refinement_state == SolutionRefinementStateEnum::coarse) {
        data_out.add_data_vector(derivative_functional_wrt_solution_coarse, derivative_functional_wrt_solution_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_coarse, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    }

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    //data_out.build_patches (mapping_collection[mapping_collection.size()-1]);
    data_out.build_patches();
    // data_out.build_patches(*(this->dg->high_order_grid.mapping_fe_field), this->dg->max_degree, dealii::DataOut<dim, dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells);
    //data_out.build_patches(*(high_order_grid.mapping_fe_field), fe_collection.size(), dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
    std::string filename = "adjoint-" ;
    if(solution_refinement_state == SolutionRefinementStateEnum::fine)
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) 
    {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) 
        {
            std::string fn = "adjoint-";
            if(solution_refinement_state == SolutionRefinementStateEnum::fine)
                fn += "fine-";
            else if(solution_refinement_state == SolutionRefinementStateEnum::coarse)
                fn += "coarse-";
            fn += dealii::Utilities::int_to_string(dim, 1) + "D-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "adjoint-";
        if(solution_refinement_state == SolutionRefinementStateEnum::fine)
            master_fn += "fine-";
        else if(solution_refinement_state == SolutionRefinementStateEnum::coarse)
            master_fn += "coarse-";
        master_fn += dealii::Utilities::int_to_string(dim, 1) +"D-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }
}

template<int dim, typename real> // To be replaced with operators->projection_operator
std::vector< real > project_function(
    const std::vector< real > &function_coeff,
    const dealii::FESystem<dim,dim> &fe_input,
    const dealii::FESystem<dim,dim> &fe_output,
    const dealii::QGauss<dim> &projection_quadrature
    //,mpi_communicator(MPI_COMM_WORLD),
    
    )
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    const unsigned int nstate = fe_input.n_components();
    const unsigned int n_vector_dofs_in = fe_input.dofs_per_cell;
    const unsigned int n_vector_dofs_out = fe_output.dofs_per_cell;
    const unsigned int n_dofs_in = n_vector_dofs_in / nstate;
    const unsigned int n_dofs_out = n_vector_dofs_out / nstate;

    assert(n_vector_dofs_in == function_coeff.size());
    assert(nstate == fe_output.n_components());

    const unsigned int n_quad_pts = projection_quadrature.size();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = projection_quadrature.get_points();

    std::vector< real > function_coeff_out(n_vector_dofs_out); // output function coefficients.
    for (unsigned istate = 0; istate < nstate; ++istate) {

        std::vector< real > function_at_quad(n_quad_pts);

        // Output interpolation_operator is V^T in the notes.
        dealii::FullMatrix<double> interpolation_operator(n_dofs_out,n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            function_at_quad[iquad] = 0.0;
            for (unsigned int idof=0; idof<n_dofs_in; ++idof) {
                const unsigned int idof_vector = fe_input.component_to_system_index(istate,idof);
                function_at_quad[iquad] += function_coeff[idof_vector] * fe_input.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }
            function_at_quad[iquad] *= projection_quadrature.weight(iquad);

            for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
                const unsigned int idof_vector = fe_output.component_to_system_index(istate,idof);
                interpolation_operator[idof][iquad] = fe_output.shape_value_component(idof_vector,unit_quad_pts[iquad],istate);
            }
        }
        std::vector< real > rhs(n_dofs_out);
        for (unsigned int idof=0; idof<n_dofs_out; ++idof) {
            rhs[idof] = 0.0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs[idof] += interpolation_operator[idof][iquad] * function_at_quad[iquad];
            }
        }
        
        dealii::FullMatrix<double> mass(n_dofs_out, n_dofs_out);
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                mass[row][col] = 0;
            }
        }
        for(unsigned int row=0; row<n_dofs_out; ++row) {
            for(unsigned int col=0; col<n_dofs_out; ++col) {
                for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                    mass[row][col] += interpolation_operator[row][iquad] * interpolation_operator[col][iquad] * projection_quadrature.weight(iquad);
                    //std::cout<<"mass matrix: "<<mass[row][col]<<std::endl; //PRINT STATEMENT TO BE REMOVED code is fine here
                }
            }
        }
    
        dealii::FullMatrix<double> inverse_mass(n_dofs_out, n_dofs_out);
        //pcout << "n_dofs_out = " << n_dofs_out << std::endl;
        //pcout << "mass diagonal check: ";
        //for (unsigned int i = 0; i < n_dofs_out; ++i)
           // pcout << mass[i][i] << " ";
        //pcout << std::endl;
        inverse_mass.invert(mass);

                for(unsigned int row=0; row<n_dofs_out; ++row) {
                    const unsigned int idof_vector = fe_output.component_to_system_index(istate,row);
                    function_coeff_out[idof_vector] = 0.0;
                    for(unsigned int col=0; col<n_dofs_out; ++col) {
                        function_coeff_out[idof_vector] += inverse_mass[row][col] * rhs[col];
                        //std::cout<<"function_coeff_out: "<<function_coeff_out[idof_vector]<<std::endl; //PRINT STATEMENT TO BE REMOVED; these numbers are blown up

                    }
                }
            }

    return function_coeff_out;

}
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


template class ExplicitErrorEstimate<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ExplicitErrorEstimate<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ExplicitErrorEstimate<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

template class LESErrorEstimate <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class LESErrorEstimate <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class LESErrorEstimate <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class LESErrorEstimate <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
