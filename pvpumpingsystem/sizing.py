# -*- coding: utf-8 -*-
"""
Module implementing sizing procedure to facilitate pv pumping station sizing.

@author: Tanguy Lunel
"""

import numpy as np
import pandas as pd
import tqdm
import warnings

import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pvgeneration as pvgen


def shrink_weather_representative(weather_data, nb_elt=48):
    """
    Create a new weather_data object representing the range of weather that
    can be found in the weather_data given. It allows to reduce
    the number of lines in the weather file from 8760 (if full year
    and hourly data) to 'nb_elt' lines, and eventually to greatly reduce
    the computation time.

    Parameters
    ----------
    weather_data: pandas.DataFrame
        The hourly data on irradiance, temperature, and others
        meteorological parameters.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    nb_elt: integer, default 48
        Number of line to keep in the weather_data file.

    Returns
    -------
    * pandas.DataFrame: weather object of nb_elt lines

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
                              num=int(np.round(nb_elt/2))).round()
    temp_selected_df = temp_sorted_df.iloc[index_array]

    # Sort DataFrame according to GHI
    ghi_sorted_df = sub_df.sort_values('ghi')
    ghi_sorted_df.reset_index(drop=True, inplace=True)
    index_array = np.linspace(0, ghi_sorted_df.index.max(),
                              num=int(np.round(nb_elt/2))).round()
    ghi_selected_df = ghi_sorted_df.iloc[index_array]

    # Concatenation of two preceding df
    final_df = pd.concat([temp_selected_df, ghi_selected_df])
    time = weather_data.index[0]
    final_df.index = pd.date_range(time, periods=nb_elt, freq='h')

    return final_df


def shrink_weather_worst_month(weather_data):
    """
    Create a new weather_data object with only the worst month of the
    weather_data given, according to the irradiance data.

    Parameters
    ----------
    weather_data: pandas.DataFrame
        The hourly data on irradiance, temperature, and others
        meteorological parameters.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    Returns
    -------
    * pandas.DataFrame: weather object of nb_elt lines

    """
    # DataFrame for results
    sum_irradiance = pd.DataFrame()

    for month in weather_data.month.drop_duplicates():
        weather_month = weather_data[weather_data.month == month]
        total_ghi = weather_month.ghi.sum()
        sum_irradiance = sum_irradiance.append({'month': month,
                                                'ghi': total_ghi},
                                               ignore_index=True)
    worst_month = sum_irradiance[sum_irradiance.ghi ==
                                 sum_irradiance.ghi.min()].month.iloc[0]

    weather_worst_month = weather_data[weather_data.month == worst_month]

    return weather_worst_month


def subset_respecting_llp(pv_database, pump_database,
                          weather_data, weather_metadata,
                          pvps_fixture,
                          llp_accepted=0.01,
                          M_s_guess=1):
    """
    Function returning the configurations of PV modules and pump
    that will minimize the net present value of the system and will insure
    the Loss of Load Probability (llp) is inferior to the one given.

    //!\\ Works fine only for MPPT coupling for now.

    Parameters
    ----------
    pv_database: list of strings,
        List of pv module names to try. If name is not eact, it will search
        a pv module database to find the best match.

    pump_database: list of pvpumpingssytem.Pump objects
        List of motor-pump to try.

    weather_data: pd.DataFrame
        Weather file of the location.
        Typically comes from pvlib.iotools.epw.read_epw()

    weather_metadata: dict
        Weather file metadata of the location.
        Typically comes from pvlib.iotools.epw.read_epw()

    pvps_fixture: pvpumpingsystem.PVPumpSystem object
        The PV pumping system to size.

    llp_accepted: float, between 0 and 1
        Maximum Loss of Load Probability that can be accepted.

    M_S_guess: integer,
        Estimated number of modules in series in the PV array. Will be sized
        by the function.

    Returns
    -------
    preselection: pd.Dataframe,
        All configurations tested respecting the LLP.
    """

    # TODO: add way of guessing M_s_guess, for example by calculating the
    # hydraulic power in water_demand and then finding the number of modules
    # for same power with an efficiency coefficient

    # TODO: insure the functioning for 'direct' coupling cases

    def funct_llp_for_Ms(pvps, M_s):
        pvps.pvgeneration.system.modules_per_string = M_s
        pvps.pvgeneration.run_model()
        pvps.run_model(starting_soc='morning', iteration=False)
        return pvps.llp

    # initalization of variables
    preselection = pd.DataFrame()

    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='Research of best combination: ',
                                 total=len(pv_database)):
        pvgen1 = pvgen.PVGeneration({'weather_data': weather_data,
                                     'metadata': weather_metadata},
                                    pv_module_name=pv_mod_name,
                                    modules_per_string=M_s_guess,
                                    strings_in_parallel=1)
        pvps_fixture.pvgeneration = pvgen1

        for pump in pump_database:

            pvps_fixture.motorpump = pump
            min_found = False
            M_s = M_s_guess
            llp_max = funct_llp_for_Ms(pvps_fixture, 0)
            llp_prev = 1.1
            while min_found is False:
                llp = funct_llp_for_Ms(pvps_fixture, M_s)
                print('module: {0} / pump: {1} / M_s: {2} / llp: {3}'.format(
                        pv_mod_name, pump.idname, M_s, llp))
                if llp <= llp_accepted and llp_prev > 1:  # first round
                    M_s -= 1
                elif llp <= llp_accepted and llp_prev <= llp_accepted:
                    M_s -= 1
                elif llp <= llp_accepted and 1 > llp_prev > llp_accepted \
                        and llp != llp_max:
                    min_found = True
                elif llp > llp_accepted and llp_prev != llp:
                    M_s += 1
                elif llp > llp_accepted and llp_prev == llp and llp != llp_max:
                    min_found = True
                elif llp > llp_accepted and llp_prev == llp and llp == llp_max:
                    M_s += 1
                # TODO: add case in which llp cannot be better because
                # of limitation from pump
                else:
                    raise Exception('This case had not been figured out'
                                    ' it could happen')
                llp_prev = llp

            preselection = preselection.append(
                    pd.Series({'pv_module': pvgen1.pv_module.name,
                               'M_s': M_s,
                               'M_p': 1,
                               'pump': pump.idname,
                               'llp': pvps_fixture.llp,
                               'npv': pvps_fixture.npv}),
                    ignore_index=True)

    return preselection


def sizing_minimize_npv(pv_database, pump_database,
                        weather_data, weather_metadata,
                        pvps_fixture,
                        llp_accepted=0.01,
                        M_s_guess=1):
    """
    Function returning the configurations of PV modules and pump
    that will minimize the net present value of the system and will insure
    that the Loss of Load Probability (llp) is inferior to the one given.
    It selects the pump and the pv module in databases and size the number
    of pv modules used.

    //!\\ Works fine only for MPPT coupling for now.

    Parameters
    ----------
    pv_database: list of strings,
        List of pv module names to try. If name is not eact, it will search
        a pv module database to find the best match.

    pump_database: list of pvpumpingssytem.Pump objects
        List of motor-pump to try.

    weather_data: pd.DataFrame
        Weather file of the location.
        Typically comes from pvlib.iotools.epw.read_epw()

    weather_metadata: dict
        Weather file metadata of the location.
        Typically comes from pvlib.iotools.epw.read_epw()

    pvps_fixture: pvpumpingsystem.PVPumpSystem object
        The PV pumping system to size.

    llp_accepted: float, between 0 and 1
        Maximum Loss of Load Probability that can be accepted.

    M_S_guess: integer,
        Estimated number of modules in series in the PV array. Will be sized
        by the function.

    Returns
    -------
    tuple with:
        selection: pd.DataFrame,
            The configurations that minimizes the net present value
            of the system.

        preselection: pd.Dataframe,
            All configurations tested respecting the LLP.
    """

    # TODO: add way of guessing M_s_guess, for example by calculating the
    # hydraulic power in water_demand and then finding the number of modules
    # for same power with an efficiency coefficient

    # TODO: insure the functioning for 'direct' coupling cases

    # TODO: check following for a discrete optimization:
    # https://towardsdatascience.com/linear-programming-and-discrete-optimization-with-python-using-pulp-449f3c5f6e99

    preselection = subset_respecting_llp(pv_database, pump_database,
                                         weather_data, weather_metadata,
                                         pvps_fixture,
                                         llp_accepted=llp_accepted,
                                         M_s_guess=M_s_guess)

    if np.isnan(preselection.npv).any():
        warnings.warn('The NPV could not be calculated, so optimized '
                      'sizing could not be found.')
        selection = preselection
    else:
        selection = preselection[preselection.npv == preselection.npv.min()]

    return (selection, preselection)


def sizing_Ms_vs_tank_size():
    """
    Function optimizing M_s and reservoir size as in Bouzidi.
    """
    raise NotImplementedError


if __name__ == '__main__':
    # ------------ MAIN INPUTS -------------------------
    # Weather input
    weather_path = (
        'data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_short = shrink_weather_representative(weather_data)
    weather_worst_month = shrink_weather_worst_month(weather_data)

    # Consumption input
    consumption_data = cs.Consumption(constant_flow=3,
                                      length=len(weather_short))

    # Pipes set-up
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)

    mppt1 = mppt.MPPT(efficiency=0.96,
                      price=1000)

    pvps1 = pvps.PVPumpSystem(None,
                              None,
                              coupling='mppt',
                              mppt=mppt1,
                              motorpump_model='arab',
                              pipes=pipes,
                              consumption=consumption_data)

    # ------------ PUMP DATABASE ---------------------
    pump_sunpump_120 = pp.Pump(path="data/pump_files/SCB_10_150_120_BL.txt",
                               idname='SCB_10_150_120_BL',
                               price=1100)

    pump_sunpump_180 = pp.Pump(path="data/pump_files/SCB_10_150_180_BL.txt",
                               idname='SCB_10_150_180_BL',
                               price=1200)

    pump_shurflo = pp.Pump(path="data/pump_files/Shurflo_9325.txt",
                           idname='Shurflo_9325',
                           price=700,
                           motor_electrical_architecture='permanent_magnet')

    pump_database = [pump_sunpump_120,
                     pump_sunpump_180,
                     pump_shurflo]

    # ------------ PV DATABASE ---------------------

    pv_database = ['Canadian_solar 340',
                   'Canadian_solar 200',
                   'Canadian solar 280']

    # -- TESTS (Temporary) --

#    selection, preselection1 = sizing_minimize_npv(
#           pv_database, pump_database,
#           weather_short, weather_metadata,
#           pvps1,
#           llp_accepted=0.05, M_s_guess=5)
#
#    preselection2 = subset_respecting_llp(
#           pv_database, pump_database,
#           weather_short, weather_metadata,
#           pvps1,
#           llp_accepted=0.05, M_s_guess=5)

    print(weather_worst_month)
