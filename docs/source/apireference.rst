.. currentmodule:: pvpumpingsystem

#############
API reference
#############


Classes
=======

The different classes of *pvpumpingsystem*.

.. autosummary::
   :toctree: generated/

   pvgeneration.PVGeneration
   mppt.MPPT
   pump.Pump
   pipenetwork.PipeNetwork
   reservoir.Reservoir
   consumption.Consumption
   pvpumpsystem.PVPumpSystem


Functions and methods
=====================


Components modeling
-------------------

.. autosummary::
   :toctree: generated/

   reservoir.Reservoir.change_water_volume

   consumption.adapt_to_flow_pumped

   pipenetwork.PipeNetwork.dynamichead

   pvgeneration.PVGeneration.run_model

   pump.Pump.iv_curve_data
   pump.Pump.functIforVH
   pump.Pump.functIforVH_Arab
   pump.Pump.functIforVH_Kou
   pump.Pump.functIforVH_theoretical
   pump.Pump.functQforVH
   pump.Pump.functQforPH
   pump.Pump.functQforPH_Hamidat
   pump.Pump.functQforPH_Arab
   pump.Pump.functQforPH_Kou
   pump.Pump.functQforPH_theoretical
   pump.get_data_pump
   pump.specs_completeness
   pump._curves_coeffs_Arab06
   pump._curves_coeffs_Kou98
   pump._curves_coeffs_Hamidat08
   pump._curves_coeffs_theoretical
   pump._curves_coeffs_theoretical_variable_efficiency
   pump._curves_coeffs_theoretical_constant_efficiency
   pump._curves_coeffs_theoretical_basic
   pump._domain_V_H
   pump._domain_P_H
   pump._extrapolate_pow_eff_with_cst_efficiency
   pump.plot_Q_vs_P_H_3d
   pump.plot_I_vs_V_H_3d
   pump.plot_Q_vs_V_H_2d




Global modeling
-------------------

.. autosummary::
   :toctree: generated/

   pvpumpsystem.PVPumpSystem.define_motorpump_model
   pvpumpsystem.PVPumpSystem.operating_point_noiteration
   pvpumpsystem.PVPumpSystem.calc_flow
   pvpumpsystem.PVPumpSystem.calc_efficiency
   pvpumpsystem.PVPumpSystem.calc_reservoir
   pvpumpsystem.PVPumpSystem.run_model
   pvpumpsystem.function_i_from_v
   pvpumpsystem.operating_point_noiteration
   pvpumpsystem.calc_flow_directly_coupled
   pvpumpsystem.calc_flow_mppt_coupled
   pvpumpsystem.calc_efficiency

Sizing tools
------------

.. autosummary::
   :toctree: generated/

   sizing.shrink_weather_representative
   sizing.shrink_weather_worst_month
   sizing.subset_respecting_llp_direct
   sizing.size_nb_pv_direct
   sizing.subset_respecting_llp_mppt
   sizing.size_nb_pv_mppt
   sizing.sizing_minimize_npv



Ancillary functions
-------------------

.. autosummary::
   :toctree: generated/

   function_models.correlation_stats
   function_models.compound_polynomial_1_2
   function_models.compound_polynomial_1_3
   function_models.compound_polynomial_2_2
   function_models.compound_polynomial_2_3
   function_models.compound_polynomial_3_3
   function_models.polynomial_multivar_3_3_4
   function_models.polynomial_multivar_3_3_1
   function_models.polynomial_multivar_2_2_1
   function_models.polynomial_multivar_2_2_0
   function_models.polynomial_multivar_1_1_0
   function_models.polynomial_multivar_0_1_0
   function_models.polynomial_5
   function_models.polynomial_4
   function_models.polynomial_3
   function_models.polynomial_2
   function_models.polynomial_1
   function_models.polynomial_divided_2_1

   waterproperties.water_prop

   finance.initial_investment
   finance.net_present_value


