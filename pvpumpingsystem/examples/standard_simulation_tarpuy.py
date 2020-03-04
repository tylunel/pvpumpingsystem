# -*- coding: utf-8 -*-
"""
Example of a simulation with pvpumpingsystem package.

@author: Tanguy Lunel
"""

import matplotlib.pyplot as plt

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pvgeneration as pvgen


# ------------ SCALE FACTOR ------------------------------
# As the case studied asks for a high output flow rate, and as this level
# of flow rate is rare for DC pump, the comparison will be made by
# assuming more than 1 pump is installed. The below parameter set this
# number of pump, and will subsequently decrease the consumption and
# the number of pv modules by the same factor.
scale_factor = 4
nb_pv_mod = 39/scale_factor  # 39 is the original number of PV module

# ------------ PV MODELING DEFINITION -----------------------

pvgen1 = pvgen.PVGeneration(
            # Weather data
            # check 'Hogar Escuela Tarpuy' for more precise location
            weather_data_and_metadata=('../data/weather_files/PER_Arequipa.847520_IWEC.epw'),
#            weather_data=('../data/weather_files/PER_Lima.846280_IWEC.epw'),

            # PV array parameters
            pv_module_name='Canadian solar 370P',
            price_per_watt=2.5,  # in US dollars
            surface_tilt=45,  # 0 = horizontal, 90 = vertical
            surface_azimuth=180,  # 180 = South, 90 = East
            albedo=0,  # between 0 and 1
            modules_per_string=nb_pv_mod,
            strings_in_parallel=1,
            # PV module glazing parameters (not always given in specs)
            glass_params={'K': 4,  # extinction coefficient [1/m]
                          'L': 0.002,  # thickness [m]
                          'n': 1.526},  # refractive index
            racking_model='open_rack',  # or'close_mount' or 'insulated_back'

            # Models used (check pvlib.modelchain for all available models)
            orientation_strategy='south_at_latitude_tilt',  # or 'flat'
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto',  # 'desoto' or 'cec'.
            ac_model='pvwatts',
            aoi_model='physical',
            spectral_model='first_solar',
            temperature_model='sapm',
            losses_model='no_loss'  # already considered via mppt object
            )

# Runs of the PV generation model
pvgen1.run_model()


# ------------ PUMP DEFINITION -----------------

# For entering new pump data:
# 1) go in: "../data/pump_files/0_template_for_pump_specs.txt"
# 2) write your specs (watch the units!),
# 3) save it under a new name (like "name_of_pump.txt"),
# 4) and close the file.
#
# To use it here then, download it with the path as follows:
pump_rosen = pp.Pump(path="../data/pump_files/rosen_SC33-158-D380-9200.txt",
                     idname='rosen_SC33-158',
                     motor_electrical_architecture='permanent_magnet',
                     price=4000,  # USD
                     modeling_method='theoretical')

pump_sunpump = pp.Pump(path="../data/pump_files/SCS_22_300_240_BL.txt",
                       motor_electrical_architecture='permanent_magnet',
                       modeling_method='arab')


# ------------ PVPS MODELING STEPS ------------------------

mppt1 = mppt.MPPT(efficiency=0.96,
                  price=1000)

pipes1 = pn.PipeNetwork(h_stat=80,  # static head [m]
                        l_tot=400,  # length of pipes [m]
                        diam=0.15,  # diameter [m]
                        material='plastic',
                        fittings=None,  # Not available yet
                        optimism=True)

reservoir1 = rv.Reservoir(size=150000,  # size [L]
                          price=500)

consumption_night = cs.Consumption(repeated_flow=[0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 0, 0, 0,
                                                  420, 420, 420, 420, 420, 420,
                                                  ])
                                   # output flow rate [L/min]
consumption_day = cs.Consumption(repeated_flow=[0, 0, 0, 0, 0, 0,
                                                0,   0,   0, 420, 420, 420,
                                                420, 420, 420, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0
                                                ])
                                 # output flow rate [L/min]
consumption_continuous = cs.Consumption(constant_flow=104)
                                        # output flow rate [L/min]

consumption_used = consumption_day
consumption_used.flow_rate = consumption_used.flow_rate/scale_factor

pvps1 = pvps.PVPumpSystem(pvgen1,
                          pump_sunpump,
                          coupling='mppt',  # to adapt: 'mppt' or 'direct',
                          mppt=mppt1,
                          pipes=pipes1,
                          consumption=consumption_used,
                          reservoir=reservoir1)


# ------------ RUNNING MODEL -----------------

pvps1.run_model()
print(pvps1)
print('LLP = ', pvps1.llp)
print('Initial investment = {0} USD'.format(
        pvps1.initial_investment * scale_factor))
print('NPV = {0} USD'.format(pvps1.npv * scale_factor))
if pvps1.coupling == 'direct':
    pvps1.functioning_point_noiteration(plot=True)


# ------------ GRAPHS -----------------------

# effective irradiance on PV array
plt.figure()
plt.plot(pvps1.efficiency.index,
         pvps1.pvgeneration.modelchain.effective_irradiance)
plt.title('Effective irradiance vs time')


# PV electric power
#plt.figure()
#plt.plot(pvps1.efficiency.index, pvps1.efficiency.electric_power)
#plt.title('Electric power in vs time')


# used for following data visualization in graphs
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


# Water volume in reservoir and output flow rate
fig, ax1 = plt.subplots()

ax1.set_xlabel('time')
ax1.set_ylabel('Water volume [L]', color='r')

ax1.plot(pvps1.water_stored.index, pvps1.water_stored.volume, color='r',
         linewidth=1, label='Reservoir level')

ax1.plot(pvps1.efficiency.index, pvps1.water_stored.extra_water,
         linewidth=1, label='Extra/lacking water')

ax1.tick_params(axis='y', labelcolor='r')


ax2 = ax1.twinx()  # instantiate a second axe that shares the same x-axis

ax2.set_ylabel('flow-rate [L/min]', color='b')
ax2.plot(pvps1.efficiency.index, pvps1.flow.Qlpm, color='b',
         linewidth=1, label='Pump output')
ax2.tick_params(axis='y', labelcolor='b')

align_yaxis(ax1, 0, ax2, 0)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()

