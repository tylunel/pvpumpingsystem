# -*- coding: utf-8 -*-
"""
Example of a sizing with pvpumpingsystem package.

@author: Tanguy Lunel
"""

import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.pvgeneration as pvgen
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.mppt as mppt
from pvpumpingsystem import sizing

# The sizing presented here is a sizing that chooses the best combination of
# motor-pump and PV module to reduce the life cycle cost of a system.
# It first computes the number of PV modules required for the combination
# to pump enough water in order to respect the maximum water shortage
# probability (named llp) accepted. Then it selects the combination with
# the lowest net present value.
# --------- ELEMENTS TO SIZE----------------------------------------------

# ------------- Pumps -----------
# Three pumps are available here. The user wants to find the one which fits
# best for the application. First the 3 pumps must be imported.
pump_1 = pp.Pump(
    path="../../pvpumpingsystem/data/pump_files/SCB_10_150_120_BL.txt")

pump_2 = pp.Pump(
    path="../../pvpumpingsystem/data/pump_files/SCB_10_150_180_BL.txt")

# Note that this pump does not have complete information given in the file,
# so some details are added here.
pump_3 = pp.Pump(
    path="../../pvpumpingsystem/data/pump_files/Shurflo_9325.txt",
    idname='Shurflo_9325',
    price=700,
    motor_electrical_architecture='permanent_magnet')

# The database must be provided under the form of a list for the sizing:
pump_database = [pump_1,
                 pump_2,
                 pump_3]

# ------------- PV Modules -----------
# PV array database:
pv_database = ['Kyocera solar KU270 6MCA',
               'Canadian Solar CS5C 80M']


# --------- MAIN INPUTS --------------------------------------------------

# Weather input
weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
    '../../pvpumpingsystem/data/weather_files/TUN_Tunis.607150_IWEC.epw',
    coerce_year=2005)

# shorten the weather data by keeping the worst month only (based on GHI)
# in order to compute faster.
weather_data = sizing.shrink_weather_worst_month(weather_data)

# Consumption input
consumption_data = cs.Consumption(constant_flow=5)  # in L/min

# Pipes set-up
pipes = pn.PipeNetwork(h_stat=20,  # vertical static head [m]
                       l_tot=100,  # length of pipes [m]
                       diam=0.05,  # diameter of pipes [m]
                       material='plastic')

# Reservoir
reservoir1 = rv.Reservoir(size=5000,  # [L]
                          water_volume=0,  # [L] at beginning
                          price=(1010+210))  # 210 is pipes price

# MPPT
mppt1 = mppt.MPPT(efficiency=0.96,
                  price=1000)

# PV generator parameters
pvgen1 = pvgen.PVGeneration(
            # Weather data
            weather_data_and_metadata={
                    'weather_data': weather_data,
                    'weather_metadata': weather_metadata},  # to adapt:

            # PV array parameters
            pv_module_name=pv_database[0],
            price_per_watt=2.5,  # in US dollars
            surface_tilt=45,  # 0 = horizontal, 90 = vertical
            surface_azimuth=180,  # 180 = South, 90 = East
            albedo=0.3,  # between 0 and 1
            racking_model='open_rack',  # or'close_mount' or 'insulated_back'

            # Models used (check pvlib.modelchain for all available models)
            orientation_strategy=None,  # or 'flat' or 'south_at_latitude_tilt'
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto',  # 'desoto' or 'cec'.
            ac_model='pvwatts',
            aoi_model='physical',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
            )

# Definition of the system. PVGeneration object must be given even
# if it will be changed afterward by the sizing function. Pump attribute can
# be kept as None.
pvps_fixture = pvps.PVPumpSystem(pvgen1,
                                 None,
                                 motorpump_model='arab',
                                 coupling='mppt',
                                 mppt=mppt1,
                                 reservoir=reservoir1,
                                 pipes=pipes,
                                 consumption=consumption_data)


# --------- RUN SIZING ---------------------------------------------------

selection, total = sizing.sizing_minimize_npv(pv_database,
                                              pump_database,
                                              weather_data,
                                              weather_metadata,
                                              pvps_fixture,
                                              llp_accepted=0.05,
                                              M_s_guess=5)

print('configurations for llp of 0.05:\n', selection)
