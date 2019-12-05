# -*- coding: utf-8 -*-
"""
@author: Tanguy
"""

import numpy as np
import pandas as pd
import tqdm

import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
# from pvpumpingsystem import errors


def shrink_pv_database(provider, nb_elt_kept=10):
    """
    Reduce the size of database by keeping only pv modules made by
    the given provider, and keep a certain number of pv modules spread
    in the range of power available.

    Parameters
    ----------
    provider: str, regex can be used
        Name of the provider(s) wanted.
        For example: "Canadian_Solar|Zytech"

    nb_elt_kept: integer
        Number of element kept in the shrunk database.
    """
    # transpose DataFrame
    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod').transpose()

    # keep only modules from specified provider
    pv_database_provider = CECMOD[CECMOD.index.str.contains(provider)]
    pv_database_provider_sorted = pv_database_provider.sort_values('STC')

    # change the index to numbers (former index kept in column 'index')
    pv_database_provider_sorted.reset_index(drop=False, inplace=True)
    index_array = np.linspace(0, pv_database_provider_sorted.index.max(),
                              num=nb_elt_kept).round()
    pv_database_kept = pv_database_provider_sorted.iloc[index_array]
    # re-change the index to pv module names
    pv_database_kept.index = pv_database_kept['index']
    del pv_database_kept['index']

    # re-tranpose DataFrame
    pv_database_kept = pv_database_kept.transpose()

    return pv_database_kept


def shrink_weather(weather_data, nb_elt=48):
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

    # Concatenation of two preceding df
    final_df = pd.concat([temp_selected_df, ghi_selected_df])
    time = weather_data.index[0]
    final_df.index = pd.date_range(time, periods=nb_elt, freq='h')

    return final_df


def run_pv_model(M_s, M_p, weather_data, weather_metadata, pv_module):
    """
    Runs the simulation of the photovoltaïc power generation.

    Returns
    -------
    * pvlib.modelchain.Modelchain:
        Modelchain object, gathering all properties of the pv array
        and the main output needed (power, diode_parameters, ...)
    """

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
                surface_tilt=45, surface_azimuth=180,
                albedo=0, surface_type=None,
                module=pv_module,
                module_parameters={**dict(pv_module),
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

    return chain1


def run_water_pumped(pv_modelchain, pump, coupling_method,
                     consumption_data, pipes_network):
    """
    Compute output flow from pv power available.

    Returns
    -------
    * pandas.DataFrame:
        contains the total volume pumped 'Qlpm', the total power generated 'P',
        and the total power unused by the pump 'P_unused'.

    """
    pipes1 = pipes_network
    reservoir1 = rv.Reservoir(1000000, 0)

    pvps1 = pvps.PVPumpSystem(pv_modelchain, pump, coupling=coupling_method,
                              pipes=pipes1,
                              consumption=consumption_data,
                              reservoir=reservoir1)
    # note: disable arg is for disabling the progress bar
    pvps1.calc_flow(disable=True)
    # TODO: fix issue on P and P_unused, it doesn't work properly when directly
    # coupled
    return np.sum(pvps1.flow[['Qlpm', 'P', 'P_unused']])


def sizing_maximize_flow(pv_database, pump_database,
                         weather_data, weather_metadata,
                         consumption_data, pipes_network):
    """
    Sizing procedure optimizing the output flow of the pumping station.

    Parameters
    ----------
    pv_database: pandas.DataFrame
        PV module database to explore.
    pump_database: list


    Note
    ----
    * Not very relevant in the case of mppt coupling as it will always be
        most powerful pv module with the highest number of module in series
        and parallel
    """
    # result dataframe
    result = pd.DataFrame()

    # Factorial computations
    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='Research of best combination: ',
                                 total=len(pv_database.columns)):
        # TODO add method to guess M_s from rated power of pump and of pv mod
        for M_s in np.arange(1, 8):
            pv_chain = run_pv_model(M_s, 1,
                                    weather_data, weather_metadata,
                                    pv_database[pv_mod_name])
            for pump in pump_database:
                output = run_water_pumped(pv_chain, pump,
                                          coupling_method,
                                          consumption_data,
                                          pipes_network)
                output = output.append(pd.Series({'pv_module': pv_mod_name,
                                                  'M_s': M_s,
                                                  'M_p': 1,
                                                  'pump': pump.model}))
                result = result.append(output, ignore_index=True)

    maximum_flow = result.Qlpm.max()
    selection = result[result.Qlpm > maximum_flow*0.99]

    if len(selection.index) > 1:
        minimum_p_unused = selection.P_unused.min()
        selection = selection[selection.P_unused == minimum_p_unused]

    return (selection, result)


def sizing_minimize_cost(acceptable_water_shortage_probability):
    """
    Sizing procedure optimizing the cost of the pumping station.
    """
    raise NotImplementedError


if __name__ == '__main__':
    # ------------ MAIN INPUTS -------------------------
    # Weather input
    weather_path = (
        'weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    # Consumption input
    consumption_data = cs.Consumption(constant_flow=1,
                                      length=len(weather_data))
    # Pipes set-up
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)
    # Modeling method choices
    pump_modeling_method = 'arab'
    coupling_method = 'mppt'

    # ------------ PUMP DATABASE ---------------------
    pump_sunpump = pp.Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                           model='SCB_10',
                           modeling_method=pump_modeling_method)

    pump_shurflo = pp.Pump(lpm={12: [212, 204, 197, 189, 186, 178,
                                     174, 166, 163, 155, 136],
                                24: [443, 432, 413, 401, 390, 382,
                                     375, 371, 352, 345, 310]},
                           tdh={12: [6.1, 12.2, 18.3, 24.4, 30.5,
                                     36.6, 42.7, 48.8, 54.9, 61.0, 70.1],
                                24: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6,
                                     42.7, 48.8, 54.9, 61.0, 70.1]},
                           current={12: [1.2, 1.5, 1.8, 2.0, 2.1, 2.4,
                                         2.7, 3.0, 3.3, 3.4, 3.9],
                                    24: [1.5, 1.7, 2.1, 2.4, 2.6, 2.8,
                                         3.1, 3.3, 3.6, 3.8, 4.1]
                                    },
                           model='Shurflo_9325',
                           motor_electrical_architecture='permanent_magnet',
                           modeling_method=pump_modeling_method)
    # TODO: reform pump_database as DataFrame to be consistent with pv_database
    pump_database = [pump_sunpump, pump_shurflo]

    # ------------ PV DATABASE ---------------------
    # use regex to add more than one provider
    provider = "Canadian_Solar|Zytech"
    nb_elt_kept = 5
    pv_database = shrink_pv_database(provider, nb_elt_kept)


    # -- TESTS (Temporary) --

    weather_short = shrink_weather(weather_data)
#    print(pv_database)
#    pv_mod = "Canadian_Solar_Inc__CS5C_80M"
#    run_pv_model(2, 1, weather_data, weather_metadata, pv_mod)

    selection, total = sizing_maximize_flow(pv_database, pump_database,
                                            weather_short, weather_metadata,
                                            consumption_data, pipes)

    print(selection)
