# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 17:11:38 2019

@author: AP78430
"""

import matplotlib.pyplot as plt
import pvlib

import pump as pp
import pipenetwork as pn
import reservoir as rv
import consumption as cs
import pvpumpsystem as pvps


CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
pvsys1 = pvlib.pvsystem.PVSystem(
            surface_tilt=50, surface_azimuth=180,
            albedo=0, surface_type=None,
            module=CECMOD.Kyocera_Solar_KU270_6MCA,
            module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
                               **glass_params},
            modules_per_string=3, strings_per_inverter=1,
            inverter=None, inverter_parameters={'pdc0': 700},
            racking_model='open_rack_cell_glassback',
            losses_parameters=None, name=None
            )

weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(
    'https://energyplus.net/weather-download/' +
    'north_and_central_america_wmo_region_4/USA/CO/' +
    'USA_CO_Denver.Intl.AP.725650_TMY3/' +
    'USA_CO_Denver.Intl.AP.725650_TMY3.epw',
    coerce_year=2005)

locat1 = pvlib.location.Location.from_epw(metadata1)

chain1 = pvlib.modelchain.ModelChain(
            system=pvsys1, location=locat1,
            orientation_strategy=None,
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto', ac_model='pvwatts', aoi_model='physical',
            spectral_model='first_solar', temp_model='sapm',
            losses_model='pvwatts', name=None)

chain1.run_model(times=weatherdata1.index, weather=weatherdata1)

pump1 = pp.Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                model='SCB_10')
pipes1 = pn.PipeNetwork(40, 100, 0.08, material='glass', optimism=True)
consumption1 = cs.Consumption(constant_flow=1)
reservoir1 = rv.Reservoir(10000, 0)

pvps1 = pvps.PVPumpSystem(chain1, pump1, coupling='mppt', pipes=pipes1,
                          consumption=consumption1, reservoir=reservoir1)

#iv = pvps1.functioning_point_noiteration(plot=True)
pvps1.calc_flow()
pvps1.calc_efficiency()
pvps1.calc_reservoir()


#    print(chain1.effective_irradiance.loc[chain1.effective_irradiance<0])

#    res1 = calc_flow_directly_coupled(chain1, pump1, pipes1, atol=0.01,
#                                      stop=8760)
#    res2 = calc_flow_mppt_coupled(chain1, pump1, pipes1, atol=0.01,
#                                  stop=8760)
#    compare = pd.DataFrame({'direct1': res1.Qlpm,
#                            'mppt': res2.Qlpm})
#    eff1 = calc_efficiency(res1, chain1.effective_irradiance, pv_area)
#    eff2 = calc_efficiency(res2, chain1.effective_irradiance, pv_area)

#
##    plt.plot(pvps1.water_stored.index, pvps1.water_stored.volume)
##    plt.plot(pvps1.efficiency.index, pvps1.efficiency.electric_power)
##    plt.plot(pvps1.efficiency.index, pvps1.flow.Qlpm)
##    plt.plot(pvps1.efficiency.index, pvps1.modelchain.effective_irradiance)
#
#    fig, ax1 = plt.subplots()
#
#    ax1.set_xlabel('time')
#    ax1.set_ylabel('Water volume in tank [L]', color='r')
#    ax1.plot(pvps1.water_stored.index, pvps1.water_stored.volume, color='r')
#    ax1.tick_params(axis='y', labelcolor='r')
#
#    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#    ax2.set_ylabel('Pump output flow-rate [L/min]', color='b')
#    ax2.plot(pvps1.efficiency.index, pvps1.flow.Qlpm, color='b')
#    ax2.tick_params(axis='y', labelcolor='b')
#
#    fig.tight_layout()  # otherwise the right y-label is slightly clipped
#    plt.show()