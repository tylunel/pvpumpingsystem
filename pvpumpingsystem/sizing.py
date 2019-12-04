# -*- coding: utf-8 -*-
"""
@author: Tanguy
"""

import numpy as np
import pandas as pd

import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
# from pvpumpingsystem import errors

# ------------ MAIN INPUTS -------------------------
# Weather input
weather_path = (
    'weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
weather_data, weather_metadata = pvlib.iotools.epw.read_epw(weather_path,
                                                            coerce_year=2005)
# Consumption input
consumption_data = cs.Consumption(constant_flow=1, length=len(weather_data))
# Modeling method choice
pump_modeling_method = 'arab'


# ------------ PUMP DATABASE ---------------------
pump_sunpump = pp.Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                       model='SCB_10',
                       modeling_method=pump_modeling_method)

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
                       modeling_method=pump_modeling_method)

pump_database = [pump_sunpump, pump_shurflo]

# ------------ PV DATABASE ---------------------

M_s = 6
M_p = 3

coupling_method = 'mppt'


def make_weather_short(weather_data, nb_elt=48):
    """
    Create a new weather_data object representing the range of weather that
    can be found in the weather_data given.

    Returns
    -------
    * pandas.DatFrame: weather object of nb_elt lines
    """
    # Remove rows with null irradiance
    sub_df = weather_data[weather_data.ghi != 0]

    # Get rows with minimum and maximum air temperature
    extreme_temp_df = sub_df[sub_df.temp_air == sub_df.temp_air.max()]
    extreme_temp_df = extreme_temp_df.append(
        sub_df[sub_df.temp_air == sub_df.temp_air.min()])

    # Sort DataFrame according to air temperature
    temp_sorted_df = sub_df.sort_values('temp_air')
    temp_sorted_df.reset_index(drop=True, inplace=True)
    index_array = np.linspace(0, temp_sorted_df.index.max(),
                              num=np.round(nb_elt/2)).round()
    temp_selected_df = temp_sorted_df.iloc[index_array]

    # Sort DataFrame according to GHI
    ghi_sorted_df = sub_df.sort_values('ghi')
    ghi_sorted_df.reset_index(drop=True, inplace=True)
    index_array = np.linspace(0, ghi_sorted_df.index.max(),
                              num=np.round(nb_elt/2)).round()
    ghi_selected_df = ghi_sorted_df.iloc[index_array]

    # Concatenation of df
    final_df = pd.concat([temp_selected_df, ghi_selected_df])
    time = weather_data.index[0]
    final_df.index = pd.date_range(time, periods=nb_elt, freq='h')

    return final_df


def total_water_pumped(pump, M_s, M_p, coupling_method,
                       consumption_data, weather_data, weather_metadata):
    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
                surface_tilt=45, surface_azimuth=180,
                albedo=0, surface_type=None,
                module=CECMOD.Kyocera_Solar_KU270_6MCA,
                module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
                                   **glass_params},
                modules_per_string=M_s, strings_per_inverter=M_p,
                inverter=None, inverter_parameters={'pdc0': 700},
                racking_model='open_rack_cell_glassback',
                losses_parameters=None, name=None
                )

    locat1 = pvlib.location.Location.from_epw(weather_metadata)

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

    chain1.run_model(times=weather_data.index, weather=weather_data)

    pipes1 = pn.PipeNetwork(h_stat=10, l_tot=100, diam=0.08,
                            material='plastic', optimism=True)
    reservoir1 = rv.Reservoir(1000000, 0)

    pvps1 = pvps.PVPumpSystem(chain1, pump, coupling=coupling_method,
                              pipes=pipes1,
                              consumption=consumption_data,
                              reservoir=reservoir1)
    pvps1.calc_flow()
    # TODO: fix issue on P and P_unused, it doesn't work properly
    return np.sum(pvps1.flow[['Qlpm', 'P', 'P_unused']])


if __name__ == '__main__':
    weather_short = make_weather_short(weather_data)
    print(weather_short[['ghi', 'temp_air']])

#    total = total_water_pumped(pump, M_s, M_p, coupling_method,
#                               consumption_data, weather_short,
#                               weather_metadata)
#    print(total)
