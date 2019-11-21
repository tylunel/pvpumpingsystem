# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:11:38 2019

@author: AP78430
"""

import matplotlib.pyplot as plt
import pvlib
import pandas as pd

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps


# %% def of fixture

# Do not use this data, because precipitable water data is incorrect, and
# will result in wrongly null output
#weather_denver = ('https://energyplus.net/weather-download/' +
#                  'north_and_central_america_wmo_region_4/USA/CO/' +
#                  'USA_CO_Denver.Intl.AP.725650_TMY3/' +
#                  'USA_CO_Denver.Intl.AP.725650_TMY3.epw')

weather_montreal = ('https://energyplus.net/weather-download/' +
                    'north_and_central_america_wmo_region_4/CAN/PQ/' +
                    'CAN_PQ_Montreal.Intl.AP.716270_CWEC/' +
                    'CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')

pump_sunpump = pp.Pump(path="../pumps_files/SCB_10_150_120_BL.txt",
                       model='SCB_10',
                       modeling_method='hamidat')
pump_shurflo = pp.Pump(lpm={12: [212, 204, 197, 189, 186, 178, 174, 166, 163,
                                 155, 136],
                            24: [443, 432, 413, 401, 390, 382, 375, 371, 352,
                                 345, 310]},
                       tdh={12: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
                                 54.9, 61.0, 70.1],
                            24: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
                                 54.9, 61.0, 70.1]},
                       current={12: [1.2, 1.5, 1.8, 2.0, 2.1, 2.4, 2.7, 3.0,
                                     3.3, 3.4, 3.9],
                                24: [1.5, 1.7, 2.1, 2.4, 2.6, 2.8, 3.1, 3.3,
                                     3.6, 3.8, 4.1]
                                },
                       model='Shurflo_9325',
                       motor_electrical_architecture='permanent_magnet',
                       modeling_method='arab')

M_s = 2
M_p = 2
weather_path = weather_montreal
pump1 = pump_sunpump
coupling_method = 'direct'

# %% modeling steps
CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
pvsys1 = pvlib.pvsystem.PVSystem(
            surface_tilt=50, surface_azimuth=180,
            albedo=0, surface_type=None,
            module=CECMOD.Kyocera_Solar_KU270_6MCA,
            module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
                               **glass_params},
            modules_per_string=M_s, strings_per_inverter=M_p,
            inverter=None, inverter_parameters={'pdc0': 700},
            racking_model='open_rack_cell_glassback',
            losses_parameters=None, name=None
            )

weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(
    weather_path, coerce_year=2005)

locat1 = pvlib.location.Location.from_epw(metadata1)

chain1 = pvlib.modelchain.ModelChain(
            system=pvsys1, location=locat1,
            orientation_strategy=None,
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto', ac_model='pvwatts', aoi_model='physical',
            spectral_model='first_solar', temperature_model='sapm',
            losses_model='pvwatts', name=None)

chain1.run_model(times=weatherdata1.index, weather=weatherdata1)

pipes1 = pn.PipeNetwork(40, 100, 0.08, material='glass', optimism=True)
consumption1 = cs.Consumption(constant_flow=1)
reservoir1 = rv.Reservoir(10000, 0)

pvps1 = pvps.PVPumpSystem(chain1, pump1, coupling=coupling_method,
                          pipes=pipes1,
                          consumption=consumption1,
                          reservoir=reservoir1)

# %% comparison mppt direct coupling
#res1 = pvps.calc_flow_directly_coupled(chain1, pump1, pipes1, atol=0.01,
#                                       stop=8760)
#res2 = pvps.calc_flow_mppt_coupled(chain1, pump1, pipes1, atol=0.01,
#                                   stop=8760)
#compare = pd.DataFrame({'direct1': res1.Qlpm,
#                        'mppt': res2.Qlpm})
#eff1 = pvps1.calc_efficiency()

flow = pvps1.calc_flow()

eff2 = pvps1.calc_efficiency()
# %% figures
#plt.figure()
#plt.plot(pvps1.efficiency.index, pvps1.efficiency.electric_power)
#plt.title('Electric power in vs time')
#
#plt.figure()
#plt.plot(pvps1.efficiency.index, pvps1.modelchain.effective_irradiance)
#plt.title('Effective irradiance vs time')
#
# %% water volume in tank and flow rate vs time
pvps1.calc_reservoir()

fig, ax1 = plt.subplots()

ax1.set_xlabel('time')
ax1.set_ylabel('Water volume in tank [L]', color='r')
ax1.plot(pvps1.water_stored.index, pvps1.water_stored.volume, color='r',
         linewidth=1)
ax1.tick_params(axis='y', labelcolor='r')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Pump output flow-rate [L/min]', color='b')
ax2.plot(pvps1.efficiency.index, pvps1.flow.Qlpm, color='b',
         linewidth=1)
ax2.tick_params(axis='y', labelcolor='b')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# %% find unused electricity
electric_power = pvps1.efficiency.electric_power
flowrate = pvps1.flow.Qlpm
# unused power only because power is too low, check pvps.flow.P_unused instead
unused_power = electric_power.loc[electric_power > 0].loc[flowrate == 0]
used_power = electric_power.loc[electric_power > 0].loc[flowrate != 0]
total_unused_power_Wh = sum(unused_power)
total_used_power_Wh = sum(used_power)

total_pumped_water_L = (flowrate*60).sum()

# potabilization of freshwater polluted from pathogen us bacteria can
# be obtained with MF-UF that requires low energy consumption: 1.2 kWh/m3
# or 1.2 Wh/L according to :
# 'Water Purification-Desalination with membrane technology supplied
# with renewable energy', Massimo Pizzichini, Claudio Russo
ratio_potabilized = ((total_unused_power_Wh / 1.2) /
                                 total_pumped_water_L)
# creuser avec ajout de cette machine sur installation:
# https://lacentrale-eco.com/fr/traitement-eau-fr/eau-domestique/traitement-uv-maison/platine-uv/platine-dom-de-traitement-uv-kit-complet-30w-ou-55w-jusqua-2-55-m-h.html

# with 4.2kJ/kg/K, water temperature can be increased of 50K with 58.5 Wh/L
ratio_heated_50K = ((total_unused_power_Wh / 58.5) /
                                   total_pumped_water_L)

print('ratio potabilized: ', ratio_potabilized,
      '\nratio heated +50C:', ratio_heated_50K)
