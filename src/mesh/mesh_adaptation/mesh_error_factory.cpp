#include "mesh_error_factory.h"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
std::unique_ptr <MeshErrorEstimateBase <dim, nstate, real, MeshType>> MeshErrorFactory<dim, nstate, real, MeshType>::create_mesh_error(std::shared_ptr< DGBase<dim,real,MeshType>> dg, const Parameters::MeshAdaptationParam *const mesh_adaptation_param)
{
    if (!(dg->all_parameters->mesh_adaptation_param.use_goal_oriented_mesh_adaptation) && dg->all_parameters->mesh_adaptation_param.use_LES_mesh_adaptation)
    {
        return std::make_unique<LESErrorEstimate<dim, nstate, real, MeshType>>(dg, mesh_adaptation_param);
    }
    else if (!(dg->all_parameters->mesh_adaptation_param.use_goal_oriented_mesh_adaptation) && dg->all_parameters->mesh_adaptation_param.use_entropy_gen_mesh_adaptation)
    {
        return std::make_unique<EntropyGenErrorEstimate<dim, nstate, real, MeshType>>(dg, mesh_adaptation_param);
    }
    else if (!(dg->all_parameters->mesh_adaptation_param.use_goal_oriented_mesh_adaptation) && dg->all_parameters->mesh_adaptation_param.use_fidkowski_mesh_adaptation)
    {
        return std::make_unique<FidkowskiErrorEstimate<dim, nstate, real, MeshType>>(dg, mesh_adaptation_param);
    }
    else if (!(dg->all_parameters->mesh_adaptation_param.use_goal_oriented_mesh_adaptation))
    {
        return std::make_unique<ResidualErrorEstimate<dim, nstate, real, MeshType>>(dg, mesh_adaptation_param);
    }
    

    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == dg->all_parameters->dimension)
    {
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected dual-weighted residual type with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == dg->all_parameters->nstate) 
        {
            return std::make_unique<DualWeightedResidualError<dim, nstate , real, MeshType>>(dg, mesh_adaptation_param);
        }
        else if constexpr (nstate > 1)
            return MeshErrorFactory<dim, nstate, real, MeshType>::create_mesh_error(dg, mesh_adaptation_param);
        else
            return nullptr;
    }
    else
    {
        std::cout<<"Cannot create MeshErrorEstimate. Invalid input"<<std::endl;
        return nullptr;
    }
}

// Instantiations for standard dealii::Triangulation
template class MeshErrorFactory<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 6, double, dealii::Triangulation<PHILIP_DIM>>;

// Instantiations for dealii::parallel::shared::Triangulation
template class MeshErrorFactory<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 6, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1

// Instantiations for dealii::parallel::distributed::Triangulation
template class MeshErrorFactory<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MeshErrorFactory<PHILIP_DIM, 6, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // namespace PHiLiP
