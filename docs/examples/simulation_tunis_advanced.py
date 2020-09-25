# -*- coding: utf-8 -*-
"""
Example of a more advanced simulation with pvpumpingsystem package.
The code is explained in details in the Jupyter Notebook file of the same
name in the same folder.

@author: Tanguy Lunel
"""

import matplotlib.pyplot as plt
import pvlib

import pvpumpingsystem.pvgeneration as pvgen
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps

# allows pandas to convert timestamp for matplotlib
# import pandas as pd
# pd.plotting.register_matplotlib_converters()

# ------------ LOCATION & WEATHER FILE IMPORT ---------------

# Pvlib tools can be used to import the weather file wanted. All options can
# be found at: https://pvlib-python.readthedocs.io/en/stable/api.html#io-tools

# Import a weather file built by PVGIS according to latitude and longitude
data, _, _, metadata = pvlib.iotools.get_pvgis_tmy(
    lat=36,  # latitude of selected location
    lon=10,  # longitude of selected location
    outputformat='epw')  # output format of file

# coerce the weather file to one single year to avoid bug with consumption file
data.year = 2005
data.index = data.index.map(lambda t: t.replace(year=2005))

# ------------ PV MODELING ----------------------

pvgen1 = pvgen.PVGeneration(
    # Weather data path
    weather_data_and_metadata={
        'weather_data': data,
        'weather_metadata': metadata},

    # PV array parameters
    pv_module_name='Canadian Solar CS5C 80M',
    price_per_watt=2.5,  # in US dollars,
    # The price must be given in dollar per watt
    # (if a 200W module costs 400USD, the price is 400/200 = 2 USD/W)
    surface_tilt=50,  # 0 = horizontal, 90 = vertical
    # More tilted to increase the yield in winter (at the expense of summer)
    surface_azimuth=180,  # 180 = South, 90 = East
    albedo=0.3,  # in [0, 1]. Albedo of soil, 0.3 is typical of dry soils.
    modules_per_string=5,  # 4 for mppt, 5 for direct (llp smaller, cheaper)
    strings_in_parallel=1,
    # PV module glazing parameters (not always given in specs)
    glass_params={'K': 4,  # extinction coefficient [1/m]
                  'L': 0.002,  # thickness [m]
                  'n': 1.526},  # refractive index
    racking_model='open_rack',  # or'close_mount' or 'insulated_back'

    # Models used (check pvlib.modelchain for all available models)
    orientation_strategy=None,  # or 'flat', or 'south_at_latitude_tilt'
    clearsky_model='ineichen',
    transposition_model='isotropic',
    solar_position_method='nrel_numpy',
    airmass_model='kastenyoung1989',
    dc_model='desoto',  # 'desoto' or 'cec', must be a single diode model.
    ac_model='pvwatts',
    aoi_model='physical',
    spectral_model='no_loss',  # recommended when not sure if the file is
    # detailed enough (ex: precipitable water column not filled can cause pb)
    temperature_model='sapm',
    losses_model='pvwatts'
    )

# Runs the PV generation model separately.
# Allows to check some outputs on power generation without running the whole
# simulation.
pvgen1.run_model()

# ------------ MPPT/DC-DC CONVERTER -------

mppt1 = mppt.MPPT(efficiency=0.96,
                  price=410,
                  idname='PCA-120-BLS-M2'
                  )

# ------------ PUMPS -----------------

# For entering new pump data:
# 1) go in: "../data/pump_files/0_template_for_pump_specs.txt"
# 2) write your specs (watch the units!),
# 3) save it under a new name (like "name_of_pump.txt"),
# 4) and close the file.
#
# To use it here then, download it with the path as follows:
pump_sunpump = pp.Pump(path="../../pvpumpingsystem/data/"
                       "pump_files/SCB_10_150_120_BL.txt",
                       modeling_method='kou')

pump_shurflo = pp.Pump(path="../../pvpumpingsystem/data/"
                       "../data/pump_files/Shurflo_9325.txt",
                       price=640,  # USD
                       motor_electrical_architecture='permanent_magnet',
                       modeling_method='arab')

# ------------ PIPES ------------------------

pipes1 = pn.PipeNetwork(h_stat=20,  # static head [m]
                        l_tot=100,  # length of pipes [m]
                        diam=0.05,  # diameter [m]
                        material='plastic')

# ------------ RESERVOIRS ------------------

reservoir1 = rv.Reservoir(size=5000,  # [L]
                          water_volume=0,  # [L] at beginning
                          price=(1010+210))  # 210 is pipes price

reservoir2 = rv.Reservoir(size=1000,  # [L]
                          water_volume=0,  # [L] at beginning
                          price=(500+210))  # 210 is pipes price


# ------------ CONSUMPTION PROFILES ------------------

# represents 2880L/day ~ continuous drip irrigation of 1000m² of garden in a
# dry climate
consumption_cst = cs.Consumption(constant_flow=2)  # output flow rate [L/min]

# represents 2880L/day ~ domestic water use of 40 people
consumption_daily = cs.Consumption(
    repeated_flow=[0, 0, 0, 0, 0, 0,
                   0, 0, 2, 1, 1, 3,
                   9, 7, 3, 3, 3, 5,
                   6, 3, 1, 1, 0, 0])


# ------------ PVPS DEFINITION -----------

# Here you gather all components of you PV pumping system previously defined:
pvps1 = pvps.PVPumpSystem(pvgen1,
                          pump_sunpump,
                          coupling='direct',  # to adapt: 'mppt' or 'direct',
                          mppt=mppt1,
                          pipes=pipes1,
                          consumption=consumption_daily,
                          reservoir=reservoir1)


# ------------ RUNNING MODEL -----------------

pvps1.run_model(iteration=False, starting_soc='morning',
                discount_rate=0.05, labour_price_coefficient=0.2, opex=200,
                lifespan_pv=30, lifespan_mppt=15, lifespan_pump=10)


# ------------ RESULTS ----------------------

print(pvps1)
print('Total water pumped in the year = ', pvps1.flow.Qlpm.sum()*60)
print('LLP = ', pvps1.llp)
print('Initial investment = {0} USD'.format(pvps1.initial_investment))
print('NPV = {0} USD'.format(pvps1.npv))

if pvps1.coupling == 'direct':
    pvps1.operating_point_noiteration(plot=True)


# ------------ GRAPHS -----------------------
# Part of data to plot
truncated_flow = pvps1.flow['2005-02-15' <= pvps1.flow.index]
truncated_flow = truncated_flow[truncated_flow.index <= '2005-02-20']

truncated_tank = pvps1.water_stored['2005-02-15' <= pvps1.water_stored.index]
truncated_tank = truncated_tank[truncated_tank.index <= '2005-02-20']

# Water volume in reservoir and output flow rate
fig, ax1 = plt.subplots()

ax1.set_xlabel('time')
ax1.set_ylabel('Water volume in tank [L]', color='r')
ax1.plot(truncated_tank.index, truncated_tank.volume, color='r',
         linewidth=1)
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Pump output flow-rate [L/min]', color='b')
ax2.plot(truncated_flow.index, truncated_flow.Qlpm, color='b',
         linewidth=1)
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
