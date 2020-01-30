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
from pvpumpingsystem import sizing


# ------------ MAIN INPUTS --------------------------------------------------

# Weather input
weather_path = (
    '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
        weather_path, coerce_year=2005)

# Consumption input
consumption_data = cs.Consumption(constant_flow=1,
                                  length=len(weather_data))

# Pipes set-up
pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                       material='plastic', optimism=True)

# Modeling method choices
coupling_method = 'mppt'  # other option is 'direct'
pump_modeling_method = 'arab'  # others are: 'kou', 'hamidat', 'theoretical'


pvps_fixture = pvps.PVPumpSystem(None, None, coupling=coupling_method,
                                 pipes=pipes, consumption=consumption_data)


# ------------ ELEMENTS TO SIZE----------------------------------------------

# Pump database:
pump_sunpump = pp.Pump(path="../data/pump_files/SCB_10_150_120_BL.txt",
                       idname='SCB_10',
                       modeling_method=pump_modeling_method)

pump_shurflo = pp.Pump(path="../data/pump_files/Shurflo_9325.txt",
                       idname='Shurflo_9325',
                       motor_electrical_architecture='permanent_magnet',
                       modeling_method=pump_modeling_method)
# TODO: reform pump_database as DataFrame to be consistent with pv_database
pump_database = [pump_sunpump, pump_shurflo]

# PV array database:
# use regex to add more than one provider.
# for example: provider = "Canadian_Solar|Zytech"
provider = "Canadian_Solar"
nb_elt_kept = 5
pv_database = sizing.shrink_pv_database(provider, nb_elt_kept)


# ------------ RUN SIZING ---------------------------------------------------

weather_short = sizing.shrink_weather(weather_data)
selection, total = sizing.sizing_maximize_flow(pv_database, pump_database,
                                               weather_short, weather_metadata,
                                               pvps_fixture)

print(selection)
