#include "parameters/parameters_mesh_adaptation.h"

namespace PHiLiP {
namespace Parameters {

MeshAdaptationParam::MeshAdaptationParam() {}

void MeshAdaptationParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        prm.declare_entry("total_mesh_adaptation_cycles","0",
                           dealii::Patterns::Integer(),
                          "Maximum adaptation steps for a problem.");
        prm.declare_entry("time_steps_between_adaptation_cycles","1000",
                           dealii::Patterns::Integer(),
                          "Number of time steps between each adaptation cycle.");
                  
        prm.declare_entry("mesh_adaptation_type", "h_adaptation",
                          dealii::Patterns::Selection(
                          " h_adaptation | "
                          " p_adaptation | "
                          " hp_adaptation | "
                          " anisotropic_adaptation "
                          ),
                          "Mesh adaptation type that we want to use. "
                          "Choices are "
                          " <h_adaptation | "
                          "  p_adaptation | "
                          "  hp_adaptation  "
                          "  anisotropic_adaptation>.");
        
        prm.declare_entry("use_goal_oriented_mesh_adaptation","false",
                          dealii::Patterns::Bool(),
                          "Flag to use goal oriented mesh adaptation. False by default.");

        prm.declare_entry("use_LES_mesh_adaptation","false",
                          dealii::Patterns::Bool(),
                          "Flag to use goal oriented mesh adaptation. False by default.");
        prm.declare_entry("mesh_adaptation_start_time","0.0",
                              dealii::Patterns::Double(0.0,1e5),
                              "Time at which to begin mesh adaptation.");
        prm.declare_entry("mesh_adaptation_end_time","0.0",
                              dealii::Patterns::Double(0.0,1e5),
                              "Time at which to end mesh adaptation.");

        prm.declare_entry("indicator_state", "all",
                            dealii::Patterns::Selection(
                            " energy | "
                            " x_momentum | "
                            " y_momentum | "
                            " x_y_momentum | "
                            " z_momentum | "
                            " momentum | "
                            " all | "
                            " mass "),
                            "State used for error indicator."
                            "Choices are "
                            " energy | "
                            " x_momentum | "
                            " y_momentum | "
                            " x_y_momentum | "
                            " z_momentum | "
                            " momentum | "
                            " all | "
                            " mass>.");

        prm.enter_subsection("fixed-fraction");
        {
            prm.declare_entry("refine_fraction","0.0",
                              dealii::Patterns::Double(0.0,1.0),
                              "Fraction of cells to be h-refined.");

            prm.declare_entry("h_coarsen_fraction","0.0",
                              dealii::Patterns::Double(0.0,1.0),
                              "Fraction of cells to be h-coarsened.");

            prm.declare_entry("hp_smoothness_tolerance","1.0e-6",
                              dealii::Patterns::Double(0.0,1.0e5),
                              "Tolerance to decide between h- or p-refinement.");
        }
        prm.leave_subsection();// "fixed-fraction"

        prm.enter_subsection("threshold");
        {
            prm.declare_entry("refine_threshold_p","0.0",
                              dealii::Patterns::Double(0.0,1.0e5),
                              "Threshold cells to be p-refined.");

            prm.declare_entry("coarsen_threshold_p","0.0",
                              dealii::Patterns::Double(0.0,1.0e3),
                              "Threshold cells to be p-coarsened.");
        }
        prm.leave_subsection();// "fixed-fraction"

        prm.enter_subsection("anisotropic");
        { 
            prm.declare_entry("mesh_complexity_anisotropic_adaptation","50.0",
                              dealii::Patterns::Double(0.0,1.0e5),
                              "Continuous equivalent of number of vertices/elements.");
            
            prm.declare_entry("norm_Lp_anisotropic_adaptation","2.0",
                              dealii::Patterns::Double(0.0,1.0e5),
                              "Lp norm w.r.t. which the optimization is performed in the continuous mesh framework.");
        }
        prm.leave_subsection(); // "anisotropic"
    }
    prm.leave_subsection(); // "mesh adaptation"

}

void MeshAdaptationParam::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("mesh adaptation");
    {
        total_mesh_adaptation_cycles = prm.get_integer("total_mesh_adaptation_cycles");
        time_steps_between_adaptation_cycles = prm.get_integer("time_steps_between_adaptation_cycles");
        const std::string mesh_adaptation_string = prm.get("mesh_adaptation_type");
        if(mesh_adaptation_string == "h_adaptation")                {mesh_adaptation_type = MeshAdaptationType::h_adaptation;}
        else if(mesh_adaptation_string == "p_adaptation")           {mesh_adaptation_type = MeshAdaptationType::p_adaptation;}
        else if(mesh_adaptation_string == "hp_adaptation")          {mesh_adaptation_type = MeshAdaptationType::hp_adaptation;}
        else if(mesh_adaptation_string == "anisotropic_adaptation") {mesh_adaptation_type = MeshAdaptationType::anisotropic_adaptation;}
        
        use_goal_oriented_mesh_adaptation = prm.get_bool("use_goal_oriented_mesh_adaptation");

        use_LES_mesh_adaptation = prm.get_bool("use_LES_mesh_adaptation");
        mesh_adaptation_start_time = prm.get_double("mesh_adaptation_start_time");
        mesh_adaptation_end_time = prm.get_double("mesh_adaptation_end_time");

        const std::string indicator_state_string = prm.get("indicator_state");
        if(indicator_state_string == "energy")                {indicator_state = IndicatorState::energy;}
        else if(indicator_state_string == "x_momentum")           {indicator_state = IndicatorState::x_momentum;}
        else if(indicator_state_string == "y_momentum")          {indicator_state = IndicatorState::y_momentum;}
        else if(indicator_state_string == "x_y_momentum") {indicator_state = IndicatorState::x_y_momentum;}
        else if(indicator_state_string == "z_momentum")           {indicator_state = IndicatorState::z_momentum;}
        else if(indicator_state_string == "momentum")          {indicator_state = IndicatorState::momentum;}
        else if(indicator_state_string == "all") {indicator_state = IndicatorState::all;}
        else if(indicator_state_string == "mass") {indicator_state = IndicatorState::mass;}

        prm.enter_subsection("fixed-fraction");
        {
            refine_fraction = prm.get_double("refine_fraction");
            h_coarsen_fraction = prm.get_double("h_coarsen_fraction");
            hp_smoothness_tolerance = prm.get_double("hp_smoothness_tolerance");
        }
        prm.leave_subsection();// "fixed-fraction"

        prm.enter_subsection("threshold");
        {
            refine_threshold_p = prm.get_double("refine_threshold_p");
            coarsen_threshold_p = prm.get_double("coarsen_threshold_p");
        }
        prm.leave_subsection();// "threshold"
        
        prm.enter_subsection("anisotropic");
        {
            mesh_complexity_anisotropic_adaptation = prm.get_double("mesh_complexity_anisotropic_adaptation");
            norm_Lp_anisotropic_adaptation = prm.get_double("norm_Lp_anisotropic_adaptation");
        }
        prm.leave_subsection(); // "anisotropic"
    }
    prm.leave_subsection(); // "mesh adaptation"
}

} // namespace Parameters
} // namespace PHiLiP

