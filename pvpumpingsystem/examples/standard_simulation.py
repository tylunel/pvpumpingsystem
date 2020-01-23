# -*- coding: utf-8 -*-
"""
Example of a simulation with pvpumpingsystem package.

@author: Tanguy Lunel
"""

import matplotlib.pyplot as plt
import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps


# ------------ DEFINITION OF FIXTURE -----------------

weather_montreal = (
    '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC_truncated.epw')

pump_sunpump = pp.Pump(path="../data/pump_files/SCB_10_150_120_BL.txt",
                       model='SCB_10',
                       modeling_method='arab')
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
                       modeling_method='arab')  # to adapt:

CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

pv_module = CECMOD.Kyocera_Solar_KU270_6MCA  # to adapt: PV module
weather_selected = weather_montreal  # to adapt:
pump_selected = pump_sunpump  # to adapt:
coupling_method_selected = 'mppt'  # to adapt:

# ------------ PV MODELING STEPS -----------------------

# PV module glazing parameters
glass_params = {'K': 4,  # to adapt: extinction coefficient [1/m]
                'L': 0.002,  # to adapt: thickness [m]
                'n': 1.526}  # to adapt: refractive index

# Definition of PV generator
pvsys1 = pvlib.pvsystem.PVSystem(
            surface_tilt=45,  # to adapt: 0 = horizontal, 90 = vertical
            surface_azimuth=180,  # to adapt: 180 = South, 90 = East
            albedo=0,  # to adapt: between 0 and 1
            surface_type=None,
            module=pv_module,
            module_parameters={**dict(pv_module),
                               **glass_params},
            module_type='glass_polymer',  # to adapt
            modules_per_string=2,  # to adapt
            strings_per_inverter=2,  # to adapt
            inverter=None,
            inverter_parameters={'pdc0': 700},
            racking_model='open_rack',  # to adapt
            losses_parameters=None,
            name=None
            )

# Import of weather
weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(weather_selected,
                                                     coerce_year=2005)
locat1 = pvlib.location.Location.from_epw(metadata1)

# Choices of models to use
chain1 = pvlib.modelchain.ModelChain(
            system=pvsys1,
            location=locat1,
            orientation_strategy=None,  # to adapt: can be ...
            clearsky_model='ineichen',  # to choose
            transposition_model='isotropic',  # to choose
            solar_position_method='nrel_numpy',  # to choose
            airmass_model='kastenyoung1989',  # to choose
            dc_model='desoto',  # to choose between 'desoto' and 'cec'.
            # Others will yield an error in pvpumpingsystem.
            ac_model='pvwatts',  # to choose
            aoi_model='physical',  # to choose
            spectral_model='first_solar',  # to choose
            temperature_model='sapm',  # to choose
            losses_model='pvwatts',  # to choose
            name=None)

# Running of the PV generation model
chain1.run_model(weather=weatherdata1)


# ------------ PVPS MODELING STEPS ------------------------

pipes1 = pn.PipeNetwork(h_stat=10,  # to adapt: static head
                        l_tot=100,  # to adapt: length of pipes
                        diam=0.08,  # to adapt: diameter [m]
                        material='plastic',  # to adapt
                        fittings=None,  # Not available yet
                        optimism=True)

reservoir1 = rv.Reservoir(size=1000000,  # to adapt: size [L]
                          water_volume=0   # to adapt: initial water in it [L]
                          )

consumption1 = cs.Consumption(constant_flow=1,
                              length=len(weatherdata1))

pvps1 = pvps.PVPumpSystem(chain1,
                          pump_selected,
                          coupling=coupling_method_selected,
                          pipes=pipes1,
                          consumption=consumption1,
                          reservoir=reservoir1)


# ------------ COMPARISON MPPT VS DIRECT COUPLING -----------------

#res1 = pvps.calc_flow_directly_coupled(chain1, pump1, pipes1, atol=0.01,
#                                       stop=8760)
#res2 = pvps.calc_flow_mppt_coupled(chain1, pump1, pipes1, atol=0.01,
#                                   stop=8760)
#compare = pd.DataFrame({'direct1': res1.Qlpm,
#                        'mppt': res2.Qlpm})
#eff1 = pvps1.calc_efficiency()

pvps1.calc_flow()
print(pvps1.flow[6:16])
pvps1.calc_efficiency()


# ------------ FIGURES -----------------------

#plt.figure()
#plt.plot(pvps1.efficiency.index, pvps1.efficiency.electric_power)
#plt.title('Electric power in vs time')
#
#plt.figure()
#plt.plot(pvps1.efficiency.index, pvps1.modelchain.effective_irradiance)
#plt.title('Effective irradiance vs time')


# ------------ WATER VOLUME AND FLOW RATE VS TIME ----------

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


# ------------ POTENTIAL PATHS TO USE UNUSED ELECTRICITY  ---------------

total_unused_power_Wh = pvps1.flow.P_unused.sum()
total_pumped_water_L = (pvps1.flow.Qlpm).sum()*60

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
