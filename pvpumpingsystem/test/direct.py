# import pytest
# import numpy as np
# import os
# import inspect

# import pvpumpingsystem.pump as pp
# import pvpumpingsystem.mppt as mppt
# import pvpumpingsystem.pipenetwork as pn
# import pvpumpingsystem.reservoir as rv
# import pvpumpingsystem.consumption as cs
# import pvpumpingsystem.pvpumpsystem as pvps
# import pvpumpingsystem.pvgeneration as pvgen


# test_dir = os.path.dirname(
#     os.path.abspath(inspect.getfile(inspect.currentframe())))



# pvgen1 = pvgen.PVGeneration(
#         # Weather data
#         weather_data_and_metadata=(
#                 os.path.join(test_dir,
#                                 '../data/weather_files/CAN_PQ_Montreal'
#                                 '.Intl.AP.716270_CWEC_truncated.epw')),

#         # PV array parameters
#         pv_module_name='kyocera solar KU270 6MCA',
#         price_per_watt=1,  # in US dollars
#         surface_tilt=45,  # 0 = horizontal, 90 = vertical
#         surface_azimuth=180,  # 180 = South, 90 = East
#         albedo=0,  # between 0 and 1
#         modules_per_string=2,
#         strings_in_parallel=2,
#         # PV module glazing parameters (not always given in specs)
#         glass_params={'K': 4,  # extinction coefficient [1/m]
#                         'L': 0.002,  # thickness [m]
#                         'n': 1.526},  # refractive index
#         racking_model='open_rack',  # or'close_mount' or 'insulated_back'

#         # Models used (check pvlib.modelchain for all available models)
#         orientation_strategy=None,  # or 'flat' or 'south_at_latitude_tilt'
#         clearsky_model='ineichen',
#         transposition_model='haydavies',
#         solar_position_method='nrel_numpy',
#         airmass_model='kastenyoung1989',
#         dc_model='desoto',  # 'desoto' or 'cec' only
#         ac_model='pvwatts',
#         aoi_model='physical',
#         spectral_model='no_loss',
#         temperature_model='sapm',
#         losses_model='pvwatts'
#         )
# pvgen1.run_model()

# mppt1 = mppt.MPPT(efficiency=1,
#                     price=200)

# pump_testfile = os.path.join(test_dir,
#                                 '../data/pump_files/SCB_10_150_120_BL.txt')
# pump1 = pp.Pump(path=pump_testfile,
#                 modeling_method='arab')

# pipes1 = pn.PipeNetwork(h_stat=10, l_tot=100, diam=0.08,
#                         material='plastic', optimism=True)

# reserv1 = rv.Reservoir()

# consum1 = cs.Consumption(constant_flow=1)

# pvps1 = pvps.PVPumpSystem(pvgen1,
#                             pump1,
#                             coupling='direct',
#                             mppt=mppt1,
#                             pipes=pipes1,
#                             consumption=consum1,
#                             reservoir=reserv1)
# pvps1.run_model()
