# -*- coding: utf-8 -*-
"""
Module implementing sizing procedure to facilitate pv pumping station sizing.

@author: Tanguy Lunel
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
import pvpumpingsystem.pvgeneration as pvgen
from pvpumpingsystem import errors


def shrink_weather(weather_data, nb_elt=48):
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


def run_pv_model(M_s, M_p, weather_data, weather_metadata, pv_module,
                 pv_array_tilt):
    """
    Runs the simulation of the photovoltaïc power generation.

    Parameters
    ----------
    M_s: integer,
        Number of pv modules in series.

    M_p: integer,
        Number of pv modules in parallel.

    weather_data: pandas.DataFrame
        The hourly data on irradiance, temperature, and others
        meteorological parameters.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    weather_metadata: dict
        The site metadata correponding to weather_data.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    pv_module: pandas.DataFrame
         Dataframe with one line containing the pv module characteristics.

    pv_array_tilt: float
        The tilt of the pv array.

    Returns
    -------
    * pvlib.modelchain.Modelchain:
        Modelchain object, gathering all properties of the pv array
        and the main output needed (power, diode_parameters, ...)
    """

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
            surface_tilt=pv_array_tilt, surface_azimuth=180,
            albedo=0, surface_type=None,
            module=pv_module,
            module_parameters={**dict(pv_module),
                               **glass_params},
            module_type='glass_polymer',
            modules_per_string=M_s, strings_per_inverter=M_p,
            inverter=None, inverter_parameters={'pdc0': 700},
            racking_model='open_rack',
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

    chain1.run_model(weather=weather_data)

    return chain1


def run_water_pumped(pvgeneration, pump, coupling_method,
                     consumption_data, pipes_network):
    """
    Compute output flow from the power produced by a pv generator.

    Parameters
    ----------
    pvgeneration: pvpumpingsystem.pvgeneration.PVGeneration,
        Object with the pv generator output characteristics (dc power,
        diode_parameters, ...)..

    pump: pvpumpingsystem.pump.Pump,
        Pump to use for modelling the output.

    coupling_method: str,
        How the pump and the pv array are linked. Can be 'mppt' or direct'.

    consumption_data: pvpumpingsystem.consumption.Consumption,
        The water consumption through time.

    pipes_network: pvpumpingsystem.pipenetwork.Pipenetwork,
        The pipes used in the system.

    Returns
    -------
    * pandas.DataFrame:
        contains the total volume pumped 'Qlpm', the total power generated 'P',
        and the total power unused by the pump 'P_unused'.

    """

    pipes1 = pipes_network
    reservoir1 = rv.Reservoir(1000000, 0)

    pvps1 = pvps.PVPumpSystem(pvgeneration, pump, coupling=coupling_method,
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
                         pvps_fixture):
    """
    Sizing procedure optimizing the output flow of the pumping station.

    Parameters
    ----------
    pv_database: pandas.DataFrame
        PV module database to explore.

    pump_database: list
        Pump database to explore.

    weather_data: pandas.DataFrame
        The hourly data on irradiance, temperature, and others
        meteorological parameters.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    weather_metadata: dict
        The site metadata correponding to weather_data.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    pvps_fixture: pvpumpingsystem.pvpumpsystem.PVPumpSystem,
        The rest of the system in which pv_database and pump_database will
        be tested to find the best set-up.


    Note
    ----
    * Not very relevant in the case of mppt coupling as it will always be
        most powerful pv module with the highest number of module in series
        and parallel
    """
    # get characteristic of pvps
    consumption_data = pvps_fixture.consumption
    pipes_network = pvps_fixture.pipes
    coupling_method = pvps_fixture.coupling

    # result dataframe
    result = pd.DataFrame()

    # Factorial computations
    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='Research of best combination: ',
                                 total=len(pv_database)):
        # TODO add method to guess M_s from rated power of pump and of pv mod
        for M_s in np.arange(1, 4):
            # TODO: add ways to look for the best tilt of arrays
            pvgen1 = pvgen.PVGeneration({'weather_data': weather_data,
                                         'metadata': weather_metadata},
                                        pv_module_name=pv_mod_name,
                                        modules_per_string=M_s,
                                        strings_in_parallel=1)
            pvgen1.run_model()

            for pump in pump_database:
                output = run_water_pumped(pvgen1,
                                          pump,
                                          coupling_method,
                                          consumption_data,
                                          pipes_network)
                output = output.append(pd.Series({
                      'pv_module': pvgen1.pv_module.name,
                      'M_s': M_s,
                      'M_p': 1,
                      'pump': pump.idname}))
                result = result.append(output, ignore_index=True)

    maximum_flow = result.Qlpm.max()
    # keep all solution that provide at least 99% of maximum output
    preselection = result[result.Qlpm > maximum_flow*0.99]

    # keep the system which wastes the minimum power among preselection
    if len(preselection.index) > 1:
        minimum_p_unused = preselection.P_unused.min()
        selection = preselection[preselection.P_unused == minimum_p_unused]
    else:
        selection = preselection

    return (selection, result)


def sizing_minimize_llp(pv_database, pump_database,
                        weather_data, weather_metadata,
                        pvps_fixture,
                        llp_accepted=0.01,
                        M_s_guess=1):
    """
    Function returning the configurations of PV modules and pump
    that insure a Loss of Load Probability (llp) inferior to the one given.

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
        total: pd.Dataframe,
            All configurations tested.

        selection: pd.DataFrame,
            The configurations that respect the maximum llp accepted.
    """
    result = pd.DataFrame()

    # TODO: add way of guessing M_s_guess, for example by calculating the
    # hydraulic power in water_demand and then finding the number of modules
    # for same power with an efficiency coefficient

    # TODO: add convergence method (bisection, or better?)
    # It would be possible to consider the modeling as a function and to
    # put it in scipy's root-finding algorithm to get the best number of

    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='Research of best combination: ',
                                 total=len(pv_database)):
        # initalization of variables
        M_s = M_s_guess
        llp = 1
        result_per_mod = pd.DataFrame()

        # process for each module
        while llp > llp_accepted:
            print('pv_mod: {0}, and M_s = {1}'.format(pv_mod_name, M_s))
            pvgen1 = pvgen.PVGeneration({'weather_data': weather_data,
                                         'metadata': weather_metadata},
                                        pv_module_name=pv_mod_name,
                                        modules_per_string=M_s,
                                        strings_in_parallel=1)
            pvgen1.run_model()

            pvps_fixture.pvgeneration = pvgen1

            for pump in pump_database:
                pvps_fixture.motorpump = pump
                print('For pump: ', pump.idname)
#                pvps_fixture.run_model(starting_soc='morning', disable=True)
                pvps_fixture.run_model(starting_soc='morning')

                result_per_mod = result_per_mod.append(
                        pd.Series({'pv_module': pvgen1.pv_module.name,
                                   'M_s': M_s,
                                   'M_p': 1,
                                   'pump': pump.idname,
                                   'llp': pvps_fixture.llp}),
                        ignore_index=True)
                print('llp = ', pvps_fixture.llp)

            # take the smallest llp of all results
            llp = result_per_mod.llp.min()

            if llp <= llp_accepted:
                result = result.append(result_per_mod,
                                       ignore_index=True)
            elif M_s > (M_s_guess + 3):
                # check that results are improving
                # TODO: comparison on result obtained 3 iterations is not ideal
                last_llp = result_per_mod[result_per_mod.M_s == M_s].llp
                prev_llp = result_per_mod[result_per_mod.M_s == (M_s-1)].llp
                if (last_llp.values == prev_llp.values).all():
                    raise errors.NoConvergenceError(
                        'The loss of load probability did not improve '
                        'during last 2 iterations. Check llp_accepted '
                        'or M_s_guess parameters')
            # increment number of module for next round
            M_s += 1

    selection = result[result.llp <= llp_accepted]

    return (selection, result)


def sizing_minimize_cost(llp):
    """
    Sizing procedure optimizing the cost of the pumping station.

    Parameter
    ---------
    llp: float between 0 and 1
        Acceptable loss of load probability (LLP) for the system,
        that is to say the acceptable water shortage probability.
        If the system is aimed at giving drinking water, it should be 0.
        If it is for agriculture, the acceptable shortage probability
        will depend on the type of crop.
    """
    raise NotImplementedError


if __name__ == '__main__':
    # ------------ MAIN INPUTS -------------------------
    # Weather input
    weather_path = (
        'data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_short = shrink_weather(weather_data)

    # Consumption input
    consumption_data = cs.Consumption(constant_flow=3,
                                      length=len(weather_short))

    # Pipes set-up
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)

    pvps1 = pvps.PVPumpSystem(None,
                              None,
                              coupling='mppt',
                              motorpump_model='arab',
                              pipes=pipes,
                              consumption=consumption_data)

    # ------------ PUMP DATABASE ---------------------
    pump_sunpump = pp.Pump(path="data/pump_files/SCB_10_150_120_BL.txt",
                           idname='SCB_10')

    pump_shurflo = pp.Pump(path="data/pump_files/Shurflo_9325.txt",
                           idname='Shurflo_9325',
                           motor_electrical_architecture='permanent_magnet')
    # TODO: reform pump_database as DataFrame to be consistent with pv_database
    pump_database = [pump_sunpump,
                     pump_shurflo]

    # ------------ PV DATABASE ---------------------

    pv_database = ['Canadian_solar 340',
                   'Canadian_solar 200']

    # -- TESTS (Temporary) --

#    selection, total = sizing_maximize_flow(pv_database, pump_database,
#                                            weather_short, weather_metadata,
#                                            pvps1)

    total, selection = sizing_minimize_llp(pv_database, pump_database,
                                           weather_short, weather_metadata,
                                           pvps1, 0.05, 1)

    print(total)
