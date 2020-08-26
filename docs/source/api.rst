.. currentmodule:: pvpumpingsystem

API reference
=============

Modules
-------

.. autosummary::
   :toctree: modules

   pvgeneration
   mppt
   pump
   pipenetwork
   reservoir
   consumption
   pvpumpsystem
   

Classes
-------

The different classes of *pvpumpingsystem*.

.. toctree::

   generated/pvpumpingsystem.pvgeneration.PVGeneration
   generated/pvpumpingsystem.mppt.MPPT
   generated/pvpumpingsystem.pump.Pump
   generated/pvpumpingsystem.pipenetwork.PipeNetwork
   generated/pvpumpingsystem.reservoir.Reservoir
   generated/pvpumpingsystem.consumption.Consumption
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem


Functions and methods
---------------------

Components modeling
^^^^^^^^^^^^^^^^^^^

.. toctree::

   generated/pvpumpingsystem.reservoir.Reservoir.change_water_volume

   generated/pvpumpingsystem.consumption.adapt_to_flow_pumped

   generated/pvpumpingsystem.pipenetwork.PipeNetwork.dynamichead

   generated/pvpumpingsystem.pvgeneration.PVGeneration.run_model

   generated/pvpumpingsystem.pump.Pump.iv_curve_data
   generated/pvpumpingsystem.pump.Pump.functIforVH
   generated/pvpumpingsystem.pump.Pump.functIforVH_Arab
   generated/pvpumpingsystem.pump.Pump.functIforVH_Kou
   generated/pvpumpingsystem.pump.Pump.functIforVH_theoretical
   generated/pvpumpingsystem.pump.Pump.functQforVH
   generated/pvpumpingsystem.pump.Pump.functQforPH
   generated/pvpumpingsystem.pump.Pump.functQforPH_Hamidat
   generated/pvpumpingsystem.pump.Pump.functQforPH_Arab
   generated/pvpumpingsystem.pump.Pump.functQforPH_Kou
   generated/pvpumpingsystem.pump.Pump.functQforPH_theoretical
   generated/pvpumpingsystem.pump.get_data_pump
   generated/pvpumpingsystem.pump.specs_completeness
   generated/pvpumpingsystem.pump._curves_coeffs_Arab06
   generated/pvpumpingsystem.pump._curves_coeffs_Kou98
   generated/pvpumpingsystem.pump._curves_coeffs_Hamidat08
   generated/pvpumpingsystem.pump._curves_coeffs_theoretical
   generated/pvpumpingsystem.pump._curves_coeffs_theoretical_variable_efficiency
   generated/pvpumpingsystem.pump._curves_coeffs_theoretical_constant_efficiency
   generated/pvpumpingsystem.pump._curves_coeffs_theoretical_basic
   generated/pvpumpingsystem.pump._domain_V_H
   generated/pvpumpingsystem.pump._domain_P_H
   generated/pvpumpingsystem.pump._extrapolate_pow_eff_with_cst_efficiency
   generated/pvpumpingsystem.pump.plot_Q_vs_P_H_3d
   generated/pvpumpingsystem.pump.plot_I_vs_V_H_3d
   generated/pvpumpingsystem.pump.plot_Q_vs_V_H_2d


Global modeling
^^^^^^^^^^^^^^^

.. toctree::

   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.define_motorpump_model
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.operating_point_noiteration
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.calc_flow
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.calc_efficiency
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.calc_reservoir
   generated/pvpumpingsystem.pvpumpsystem.PVPumpSystem.run_model
   generated/pvpumpingsystem.pvpumpsystem.function_i_from_v
   generated/pvpumpingsystem.pvpumpsystem.operating_point_noiteration
   generated/pvpumpingsystem.pvpumpsystem.calc_flow_directly_coupled
   generated/pvpumpingsystem.pvpumpsystem.calc_flow_mppt_coupled
   generated/pvpumpingsystem.pvpumpsystem.calc_efficiency


Sizing tools
^^^^^^^^^^^^

.. toctree::

   generated/pvpumpingsystem.sizing.shrink_weather_representative
   generated/pvpumpingsystem.sizing.shrink_weather_worst_month
   generated/pvpumpingsystem.sizing.subset_respecting_llp_direct
   generated/pvpumpingsystem.sizing.size_nb_pv_direct
   generated/pvpumpingsystem.sizing.subset_respecting_llp_mppt
   generated/pvpumpingsystem.sizing.size_nb_pv_mppt
   generated/pvpumpingsystem.sizing.sizing_minimize_npv


Ancillary functions
^^^^^^^^^^^^^^^^^^^

.. toctree::

   generated/pvpumpingsystem.function_models.correlation_stats
   generated/pvpumpingsystem.function_models.compound_polynomial_1_2
   generated/pvpumpingsystem.function_models.compound_polynomial_1_3
   generated/pvpumpingsystem.function_models.compound_polynomial_2_2
   generated/pvpumpingsystem.function_models.compound_polynomial_2_3
   generated/pvpumpingsystem.function_models.compound_polynomial_3_3
   generated/pvpumpingsystem.function_models.polynomial_multivar_3_3_4
   generated/pvpumpingsystem.function_models.polynomial_multivar_3_3_1
   generated/pvpumpingsystem.function_models.polynomial_multivar_2_2_1
   generated/pvpumpingsystem.function_models.polynomial_multivar_2_2_0
   generated/pvpumpingsystem.function_models.polynomial_multivar_1_1_0
   generated/pvpumpingsystem.function_models.polynomial_multivar_0_1_0
   generated/pvpumpingsystem.function_models.polynomial_5
   generated/pvpumpingsystem.function_models.polynomial_4
   generated/pvpumpingsystem.function_models.polynomial_3
   generated/pvpumpingsystem.function_models.polynomial_2
   generated/pvpumpingsystem.function_models.polynomial_1
   generated/pvpumpingsystem.function_models.polynomial_divided_2_1

   generated/pvpumpingsystem.waterproperties.water_prop

   generated/pvpumpingsystem.finance.initial_investment
   generated/pvpumpingsystem.finance.net_present_value