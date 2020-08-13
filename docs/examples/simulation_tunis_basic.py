# -*- coding: utf-8 -*-
"""
Example of a very basic simulation with pvpumpingsystem package.

@author: Tanguy Lunel
"""

import matplotlib.pyplot as plt

import pvpumpingsystem.pvgeneration as pvgen
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.pvpumpsystem as pvps

# ------------ LOCATION & PV MODELING ----------------------

pvgen1 = pvgen.PVGeneration(
    # Weather data path
    weather_data_and_metadata=(
        '../../pvpumpingsystem/data/weather_files/'
        'TUN_Tunis.607150_IWEC.epw'),

    # PV array parameters
    pv_module_name='Canadian Solar CS5C 80M',
    modules_per_string=4,  # number of modules in series
    strings_in_parallel=1,  # number of strings in parallel
    albedo=0.3,  # in [0, 1]. Albedo of soil, 0.3 is typical of dry soils.

    # Models used (check pvlib.modelchain for all available models)
    orientation_strategy='south_at_latitude_tilt'  # or 'flat' or None
    )

# ------------ MPPT/DC-DC CONVERTER ---------

mppt1 = mppt.MPPT(efficiency=0.96,
                  price=410,
                  idname='PCA-120-BLS-M2'
                  )

# ------------ PUMPS -----------------

# For entering new pump data:
# 1) open the template at: "../data/pump_files/0_template_for_pump_specs.txt"
# 2) write your specs (watch the units!),
# 3) save it under a new name (like "name_of_pump.txt"),
# 4) and close the file.
#
# To use it here then, download it with the path as follows:
pump_sunpump = pp.Pump(path="../../pvpumpingsystem/data/"
                       "pump_files/SCB_10_150_120_BL.txt")

# ------------ PIPES ------------------------

pipes1 = pn.PipeNetwork(h_stat=20,  # static head [m]
                        l_tot=100,  # length of pipes [m]
                        diam=0.05,  # diameter [m]
                        material='plastic')


# ------------ PVPS DEFINITION -----------
# Here you gather all components of your PV pumping system previously defined:
pvps1 = pvps.PVPumpSystem(pvgen1,
                          pump_sunpump,
                          coupling='direct',  # to adapt: 'mppt' or 'direct',
                          pipes=pipes1)


# ------------ RUNNING MODEL -----------------

pvps1.run_model()

print(pvps1)
print('Total water pumped in the year = ', pvps1.flow.Qlpm.sum())
print('LLP = ', pvps1.llp)
print('Initial investment = {0} USD'.format(pvps1.initial_investment))
print('NPV = {0} USD'.format(pvps1.npv))
if pvps1.coupling == 'direct':
    pvps1.operating_point_noiteration(plot=True)

# ------------ GRAPHS -----------------------

# effective irradiance on PV array
plt.figure()
plt.plot(pvps1.efficiency.index[24:47].hour,
         pvps1.pvgeneration.modelchain.effective_irradiance[24:47])
plt.xlabel('Hour')
plt.title('Effective irradiance vs time')
plt.xlim([-1, 23])

# PV electric power
plt.figure()
plt.plot(pvps1.efficiency.index[24:47].hour,
         pvps1.efficiency.electric_power[24:47])
plt.xlabel('Hour')
plt.title('Electric power in vs time')

# Water volume in reservoir and output flow rate
plt.figure()
plt.plot(pvps1.efficiency.index[24:47].hour,
         pvps1.flow.Qlpm[24:47])
plt.xlabel('Hour')
plt.ylabel('Pump output flow-rate [L/min]')
plt.title('Output flow-rate vs time')
