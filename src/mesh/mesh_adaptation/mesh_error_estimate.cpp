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
LESErrorEstimate<dim, nstate, real, MeshType> :: LESErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
    , solution_coarse(this->dg->solution)
    , solution_refinement_state(SolutionRefinementStateEnum::coarse)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    {}


template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> LESErrorEstimate<dim, nstate, real, MeshType> :: compute_cellwise_errors()
{
    auto Q_p = this->dg->solution; //save original solution
    //compute residual at p+1
    this->dg->assemble_residual();
    pcout<<"Residual is assembled..."<<std::endl;

    //calculate Projection(R{Q_p})
    std::vector<std::vector<real>> p_order_residual(this->dg->triangulation->n_active_cells());
    std::vector<std::vector<real>> projected_residual(this->dg->triangulation->n_active_cells());

    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    pcout<<"About to go through cell loop..."<<std::endl;

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;
        pcout<<"clue 1"<<std::endl;
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);
        pcout<<"clue 2"<<std::endl;
        p_order_residual[cell->active_cell_index()].resize(n_dofs_curr_cell);   //resize vector for DOFs of current cell
         //gather inputs for project_function(), and then project the rhs to p+1
        const int poly_degree = cell->active_fe_index();
        pcout << "poly_degree: " << static_cast<unsigned int>(poly_degree) << std::endl;
        if (static_cast<unsigned int>(cell->active_fe_index() + 1) >= this->dg->fe_collection.size()) {
           pcout << "ERROR: cell " << cell->active_cell_index()
                 << " has fe_index " << cell->active_fe_index()
                 << " but fe_collection only has " << this->dg->fe_collection.size()
                 << " entries!" << std::endl;
        }

        pcout<<"clue 3 and poly_degree: "<<poly_degree<<std::endl;
        const dealii::FESystem<dim,dim> &fe_input = this->dg->fe_collection[poly_degree];
        pcout<<"clue 3.5"<<std::endl;
        const dealii::FESystem<dim,dim> &fe_output = this->dg->fe_collection[poly_degree + 1];  
        pcout<<"clue 4"<<std::endl;
        const dealii::QGauss <dim> projection_quadrature(fe_index_curr_cell +2);
        pcout<<"clue 5"<<std::endl;

        std::vector<real> p_order_residual_per_cell = p_order_residual[cell->active_cell_index()];
        pcout<<"About to call project_function..."<<std::endl;

        projected_residual[cell->active_cell_index()] = project_function(p_order_residual_per_cell, fe_input, fe_output, projection_quadrature); 
        pcout<<"called project_function..."<<std::endl;

        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            if(cell->active_cell_index() == 0){
                pcout<<"p+1_order_residual_per_cell =  "<<projected_residual[cell->active_cell_index()][idof]<<std::endl;
                }
            }
    }    
    //project mesh to p+1
    this->reinit(); //do we need this?? maybe for residual vector. remove
    this->convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);
    pcout<<"Projected mesh to p+1..."<<std::endl;
    this->dg->assemble_residual(); //assemble residual of projected mesh
    pcout<<"Assembled residual of projected mesh..."<<std::endl;

    unsteady_residual.reinit(this->dg->triangulation->n_active_cells());
    // compute the error indicator cell-wise by taking the dot product over the DOFs with the residual vector
    pcout<<"About to go through cell loop for error indicator..."<<std::endl;
    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
        if(!cell->is_locally_owned())  continue;
        
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        real rhs_cell = 0;
        real rhs_cell_sum = 0;
        //pcout<<"cell"<<cell->active_cell_index()<<":"<<std::endl;
        //pcout<<"Size of projected_residual[cell->active_cell_index()]: "<<projected_residual[cell->active_cell_index()].size()<<std::endl;
        //pcout<<"Number of DOFs in current cell: "<<n_dofs_curr_cell<<std::endl;
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            //pcout<<"Solution for this DOF: "<<this->dg->solution[idof]<<std::endl;
            rhs_cell = std::abs((this->dg->right_hand_side[current_dofs_indices[idof]] - projected_residual[cell->active_cell_index()][idof])); //subtract here
            if (this->dg->solution[idof] > 1.0) {
                rhs_cell = rhs_cell/(this->dg->solution[idof]);
            }
            if(cell->active_cell_index() == 0){
                pcout<<"Solution for this DOF (AFTER Refinement): "<<this->dg->solution[idof]<<std::endl;
                pcout<<"Projected residual for this DOF(AFTER): " <<projected_residual[cell->active_cell_index()][idof]<<std::endl;
            }

            rhs_cell_sum += rhs_cell;
        }  
        //pcout<<"rhs_cell_sum="<<rhs_cell_sum<<std::endl;
        unsteady_residual[cell->active_cell_index()] = std::abs(rhs_cell_sum/(nstate*n_dofs_curr_cell));
    }
    pcout<<"end of error estimation cell loop..."<<std::endl;
    
    this->convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);  // restore mesh 
    pcout<<"Restored mesh to coarse..."<<std::endl;
    this->dg->solution = Q_p; //restore solution vector
    pcout<<"Restored solution vector..."<<std::endl;
    //adapt the p-order
    //std::vector<dealii::types::global_dof_index> dofs_indices;
    //dealii::Vector<real> cellwise_errors (this->dg->high_order_grid->triangulation->n_active_cells());
    pcout<<"end of error estimation..."<<std::endl;
    return unsteady_residual;
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
    pcout<<"locally_owned_dofs.size()="<<locally_owned_dofs.size()<<std::endl;
    solution_coarse.update_ghost_values();
    
    // Solution Transfer to fine grid
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    pcout<<"Prepared for coarsening and refinement..."<<std::endl;
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
    pcout<<"Allocated system for p+1"<<std::endl;
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

    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    this->dg->solution = solution_coarse;
    
    [[maybe_unused]] unsigned int no_of_cells_after_changing_p = this->dg->triangulation->n_active_cells(); // Used when compiled in debug mode (in assert).

    AssertDimension(no_of_cells_before_changing_p, no_of_cells_after_changing_p);

    solution_refinement_state = SolutionRefinementStateEnum::coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
void LESErrorEstimate<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle)
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

    //output error estimate
    dealii::Vector<real> error_estimate = compute_cellwise_errors();
    data_out.add_data_vector(error_estimate, "error_estimate", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

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
    //pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    )
{
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
