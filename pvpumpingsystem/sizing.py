# -*- coding: utf-8 -*-
"""
Module implementing sizing procedure to facilitate pv pumping station sizing.

@author: Tanguy Lunel
"""

import numpy as np
import pandas as pd
import tqdm
import warnings

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
    pandas.DataFrame
        Weather data with (nb_elt) lines

    """
    # Remove rows with null irradiance
    sub_df = weather_data[weather_data.ghi != 0]

    # Get rows with minimum and maximum air temperature
    extreme_temp_df = sub_df[sub_df.temp_air == sub_df.temp_air.max()]
    extreme_temp_df = pd.concat([extreme_temp_df,
        sub_df[sub_df.temp_air == sub_df.temp_air.min()]])

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
    weather_data given, according to the global horizontal irradiance (ghi)
    data.

    Parameters
    ----------
    weather_data: pandas.DataFrame
        The hourly data on irradiance, temperature, and others
        meteorological parameters.
        Typically comes from pvlib.epw.read_epw() or pvlib.tmy.read.tmy().

    Returns
    -------
    pandas.DataFrame
        Weather data with (nb_elt) lines

    """
    # TODO: add attribute for selecting which criteria to use for considering
    # the worst month (could be ghi (like now), dni, dhi, temperature, etc)

    # DataFrame for results
    sum_irradiance = pd.DataFrame(columns=['month', 'ghi'])

    for month in weather_data.month.drop_duplicates():
        weather_month = weather_data[weather_data.month == month]
        total_ghi = weather_month.ghi.sum()
        sum_irradiance = pd.concat([sum_irradiance,
                                    pd.DataFrame([[month,total_ghi]], columns=['month','ghi'])]
                                    ,ignore_index=True)

    worst_month = sum_irradiance[sum_irradiance.ghi ==
                                 sum_irradiance.ghi.min()].month.iloc[0]

    weather_worst_month = weather_data[weather_data.month == worst_month]

    return weather_worst_month


def subset_respecting_llp_direct(pv_database, pump_database,  # noqa: C901
                                 weather_data, weather_metadata,
                                 pvps_fixture,
                                 llp_accepted=0.01,
                                 M_s_guess=None,
                                 M_p_guess=None,
                                 **kwargs):
    """
    Function returning the configurations of PV modules and pump
    that will minimize the net present value of the system and will insure
    the Loss of Load Probability (llp) is inferior to the one given.

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

    llp_accepted: float, default is 0.01
        Maximum Loss of Load Probability that can be accepted. Between 0 and 1

    M_S_guess: integer, default is None
        Estimated number of modules in series in the PV array. Will be sized
        by the function.

    Returns
    -------
    pandas.Dataframe
        All configurations tested respecting the LLP.
    """
    if pvps_fixture.coupling != 'direct':
        pvps_fixture.coupling = 'direct'
        warnings.warn("Pvps coupling method changed to 'direct'.")

    # initialization of variables
    preselection = pd.DataFrame()

    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='PV database exploration: ',
                                 total=len(pv_database)):

        # Sets the PV module
        pvps_fixture.pvgeneration = pvgen.PVGeneration(
            weather_data_and_metadata={
                    'weather_data': weather_data,
                    'weather_metadata': weather_metadata},
            pv_module_name=pv_mod_name,

            price_per_watt=2.5,  # in US dollars
            albedo=0,  # between 0 and 1
            modules_per_string=1,
            strings_in_parallel=1,
            # PV module glazing parameters (not always given in specs)
            glass_params={'K': 4,  # extinction coefficient [1/m]
                          'L': 0.002,  # thickness [m]
                          'n': 1.526},  # refractive index
            racking_model='open_rack',  # or'close_mount' or 'insulated_back'

            # Models used (check pvlib.modelchain for all available models)
            orientation_strategy='south_at_latitude_tilt',  # or 'flat' or None
            clearsky_model='ineichen',
            transposition_model='isotropic',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto',  # 'desoto' or 'cec'.
            ac_model='pvwatts',
            aoi_model='physical',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
            )
        all_series = [] 
        for pump in tqdm.tqdm(pump_database,
                              desc='Pump database exploration: ',
                              total=len(pump_database)):
            # check that pump can theoretically work for given tdh
            if pvps_fixture.pipes.h_stat > 0.9*pump.range.tdh['max']:
                warnings.warn('Pump {0} does not match '
                              'the required head'.format(pump.idname))
                continue  # skip this round

            # compute limits for number of modules in PV array
            # M_p
            I_sc_array_min = pump.range.current['min']
            I_sc_array_max = pump.range.current['max'] * 4  # arbitrary coeff
            M_p_max = (I_sc_array_max  # round down
                       // pvps_fixture.pvgeneration.pv_module.I_sc_ref)
            M_p_min = (I_sc_array_min  # round up
                       // pvps_fixture.pvgeneration.pv_module.I_sc_ref) + 1
            # M_s
            V_oc_array_min = pump.range.voltage['min']
            V_oc_array_max = pump.range.voltage['max'] * 1.2
            M_s_min = (V_oc_array_min  # round up
                       // pvps_fixture.pvgeneration.pv_module.V_oc_ref) + 1
            M_s_max = (V_oc_array_max  # round up
                       // pvps_fixture.pvgeneration.pv_module.V_oc_ref) + 1

            # Check that the current pump is in the range
            if pvps_fixture.pvgeneration.pv_module.V_oc_ref > V_oc_array_max:
                warnings.warn(('Pump {0} and PV module voltage of {1} '
                               'do not match').format(pump.idname,
                                                      pv_mod_name))
                continue  # skip this round
            if pvps_fixture.pvgeneration.pv_module.I_sc_ref > I_sc_array_max:
                warnings.warn(('Pump {0} and PV module current of {1} '
                               'do not match').format(pump.idname,
                                                      pv_mod_name))
                continue  # skip this round

            # If the range is ok, update the pump
            pvps_fixture.motorpump = pump

            M_s, M_p = size_nb_pv_direct(
                    pvps_fixture, llp_accepted,
                    M_s_min, M_s_max, M_p_min, M_p_max, **kwargs)

            new_row = pd.Series({
                'pv_module': pvps_fixture.pvgeneration.pv_module.name,
                'M_s': M_s,
                'M_p': M_p,
                'pump': pump.idname,
                'llp': pvps_fixture.llp,
                'npv': pvps_fixture.npv
            })
            
            all_series.append(new_row)
            # preselection = preselection.append(
            #     pd.Series({
            #             'pv_module': pvps_fixture.pvgeneration.pv_module.name,
            #             'M_s': M_s,
            #             'M_p': M_p,
            #             'pump': pump.idname,
            #             'llp': pvps_fixture.llp,
            #             'npv': pvps_fixture.npv}),
            #     ignore_index=True)

    # Remove not satifying LLP
    preselection = pd.concat(all_series, axis=1).T  # Transpose to make each Series a row
    preselection = preselection[preselection.llp <= llp_accepted]

    return preselection


def size_nb_pv_direct(pvps_fixture, llp_accepted,    # noqa: C901
                      M_s_min, M_s_max, M_p_min, M_p_max,
                      M_s_guess=None, M_p_guess=None,
                      **kwargs):
    """
    Function sizing the PV generator (i.e. the number of PV modules) for
    a specified llp_accepted.

    Returns
    -------
    tuple
        Number of modules in series and number of strings in parallel.
    """

    def funct_llp_for_Ms_Mp(pvps, M_s, M_p, **kwargs):
        pvps.pvgeneration.system.arrays[0].modules_per_string = M_s
        pvps.pvgeneration.system.arrays[0].strings = M_p
        pvps.pvgeneration.run_model()
        pvps.run_model(**kwargs)

        return pvps.llp

    # Guess a M_s to start with or take the given M_s_guess:
    if M_s_guess is None:
        M_s = M_s_min
    else:
        M_s = M_s_guess
    # Guess a M_p to start with or take the given M_p_guess:
    if M_p_guess is None:
        M_p = M_p_min
    else:
        M_p = M_p_guess

    # initialization of variable for first round
    llp_prev_Ms = 1.1
    llp_prev_Mp = 1.1
    M_s_prev = M_s_min

    while True:
        # new LLP
        llp = funct_llp_for_Ms_Mp(pvps_fixture, M_s, M_p, **kwargs)
        # printing (for debug):
        print('module: {0} / pump: {1} / M_s: {2} /'
              '/ M_p: {3} / llp: {4} / npv: {5}'.format(
                pvps_fixture.pvgeneration.pv_module_name,
                pvps_fixture.motorpump.idname,
                M_s, M_p,
                llp, pvps_fixture.npv))
        # changing M_s
        if llp <= llp_accepted:
            break
        elif llp > llp_accepted:
            if llp < llp_prev_Ms and M_s < M_s_max:
                M_s_prev = M_s
                M_s += 1
                M_s_change_efficient = True
            elif llp >= llp_prev_Ms:
                M_s = M_s_prev
                M_s_change_efficient = False
            else:
                pass  # keep current M_s
        llp_prev_Ms = llp

        if not M_s_change_efficient:  # then change M_p
            llp = funct_llp_for_Ms_Mp(pvps_fixture, M_s, M_p, **kwargs)
            print('module: {0} / pump: {1} / M_s: {2} /'
                  '/ M_p: {3} / llp: {4} / npv: {5}'.format(
                      pvps_fixture.pvgeneration.pv_module_name,
                      pvps_fixture.motorpump.idname,
                      M_s, M_p,
                      llp, pvps_fixture.npv))
            # changing M_p
            if llp <= llp_accepted:
                break
            elif llp > llp_accepted:
                if llp < llp_prev_Mp and M_p < M_p_max:
                    M_p += 1
                elif M_p == M_p_max:
                    break
                else:
                    break
            llp_prev_Mp = llp

    return M_s, M_p


# TODO: make this function work with the voltage range of mppt (once available)
# so as to size M_s and M_p. For now only M_s changes, but correponds
# actually more to the number of pv module than to nb of pv modules in series.
def subset_respecting_llp_mppt(pv_database, pump_database,    # noqa: C901
                               weather_data, weather_metadata,
                               pvps_fixture,
                               llp_accepted=0.01,
                               M_s_guess=None,
                               **kwargs):
    """
    Function returning the configurations of PV modules and pump
    that will minimize the net present value of the system and will ensure
    the Loss of Load Probability (llp) is inferior to the one given.

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

    llp_accepted: float, default is 0.01
        Maximum Loss of Load Probability that can be accepted. Between 0 and 1

    M_S_guess: integer, default is None
        Estimated number of modules in series in the PV array. Will be sized
        by the function.

    Returns
    -------
    pandas.Dataframe,
        All configurations tested respecting the LLP.
    """
    if pvps_fixture.coupling != 'mppt':
        pvps_fixture.coupling = 'mppt'
        warnings.warn("Pvps coupling method changed to 'mppt'.")

    # initalization of variables
    preselection = pd.DataFrame(columns=['pv_module',
                                         'M_s',
                                         'M_p',
                                         'pump',
                                         'llp',
                                         'npv'])

    for pv_mod_name in tqdm.tqdm(pv_database,
                                 desc='Research of best combination: ',
                                 total=len(pv_database)):
        # Sets the PV module
        pvps_fixture.pvgeneration = pvgen.PVGeneration(
            weather_data_and_metadata={
                    'weather_data': weather_data,
                    'weather_metadata': weather_metadata},
            pv_module_name=pv_mod_name,

            price_per_watt=2.5,  # in US dollars
            albedo=0,  # between 0 and 1
            modules_per_string=1,
            strings_in_parallel=1,
            # PV module glazing parameters (not always given in specs)
            glass_params={'K': 4,  # extinction coefficient [1/m]
                          'L': 0.002,  # thickness [m]
                          'n': 1.526},  # refractive index
            racking_model='open_rack',  # or'close_mount' or 'insulated_back'

            # Models used (check pvlib.modelchain for all available models)
            orientation_strategy='south_at_latitude_tilt',  # or 'flat' or None
            clearsky_model='ineichen',
            transposition_model='isotropic',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto',  # 'desoto' or 'cec'.
            ac_model='pvwatts',
            aoi_model='physical',
            spectral_model='no_loss',
            temperature_model='sapm',
            losses_model='no_loss'
            )

        for pump in pump_database:
            # check that pump can theoretically match
            if pvps_fixture.pipes.h_stat > 0.9*pump.range.tdh['max']:
                warnings.warn('Pump {0} does not match '
                              'the required head'.format(pump.idname))
                continue  # skip this round

            # Sets the motorpump
            pvps_fixture.motorpump = pump

            # Sizes the PV generator for respecting the llp_accepted
            M_s = size_nb_pv_mppt(pvps_fixture, llp_accepted, M_s_guess,
                                  **kwargs)

            preselection = pd.concat([
                preselection,
                pd.DataFrame(
                    [[ pvps_fixture.pvgeneration.pv_module.name,
                       M_s,
                       1,
                       pump.idname, 
                       pvps_fixture.llp, 
                       pvps_fixture.npv
                    ]],columns=['pv_module','M_s','M_p','pump','llp','npv'])
            ,], ignore_index=True, axis=0, join='outer')

    # Remove not satifying LLP
    preselection = preselection[preselection.llp <= llp_accepted]

    return preselection


def size_nb_pv_mppt(pvps_fixture, llp_accepted, M_s_guess, **kwargs):
    """
    Function sizing the PV generator (i.e. the number of PV modules) for
    a specified llp_accepted. Here 'M_s' represents the total number
    of PV module (because M_p = 1).

    Returns
    -------
    float
        Number of PV modules in the array, regardless of how they are
        arranged.
    """

    def funct_llp_for_Ms(pvps, M_s, **kwargs):
        pvps.pvgeneration.system.arrays[0].modules_per_string = M_s
        pvps.pvgeneration.system.arrays[0].strings = 1
        pvps.pvgeneration.run_model()
        pvps.run_model(**kwargs)
        return pvps.llp

    llp_max = funct_llp_for_Ms(pvps_fixture, 0, **kwargs)

    # Guess a M_s to start with:
    if M_s_guess is None:
        M_s = (pvps_fixture.motorpump.range.power['max'] //
               pvps_fixture.pvgeneration.pv_module.PTC)
    else:
        M_s = M_s_guess

    # initialization of variable for first round
    llp_prev = 1.1

    while True:  # while loop with break statement
        # new LLP:
        llp = funct_llp_for_Ms(pvps_fixture, M_s, **kwargs)
        # printing (useful for debug):
        print('module: {0} / pump: {1} / M_s: {2} /'
              ' llp: {3} / npv: {4}'.format(
                pvps_fixture.pvgeneration.pv_module_name,
                pvps_fixture.motorpump.idname,
                M_s,
                llp,
                pvps_fixture.npv))
        # decision tree:
        if llp <= llp_accepted and llp_prev > 1:  # first round
            M_s -= 1
        elif llp <= llp_accepted and llp_prev <= llp_accepted:
            M_s -= 1
        elif llp <= llp_accepted and 1 > llp_prev > llp_accepted \
                and llp != llp_max:
            break
        elif llp > llp_accepted and llp_prev != llp:
            M_s += 1
        elif llp > llp_accepted and llp_prev == llp and llp != llp_max:
            break  # unsatisfying llp, removed later
        elif llp > llp_accepted and llp_prev == llp and llp == llp_max:
            M_s += 1
        else:
            raise Exception('This case had not been figured out'
                            ' it could happen')
        # preparation for next turn
        llp_prev = llp

    return M_s


def sizing_minimize_npv(pv_database, pump_database,
                        weather_data, weather_metadata,
                        pvps_fixture,
                        llp_accepted=0.01,
                        M_s_guess=None,
                        M_p_guess=None,
                        **kwargs):
    """
    Function returning the configuration of PV modules and pump
    that minimizes the net present value (NPV) of the system and ensures
    that the Loss of Load Probability (llp) is inferior to the 'llp_accepted'.

    It proceeds by sizing the number of PV module needed to respect
    'llp_accepted' for each combination of pump and pv module. If the
    combination does not allow to respect 'llp_accepted' in any case,
    it is discarded. Then the combination with the lowest NPV is returned
    as the solution (first element of the tuple returned). All combinations
    details are also returned (second element of the tuple returned).

    Parameters
    ----------
    pv_database: list of strings,
        List of pv module names to try. If name is not eact, it will search
        a pv module database to find the best match.

    pump_database: list of pvpumpingssytem.Pump objects
        List of motor-pump to try.

    weather_data: pd.DataFrame
        Weather data of the location.
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

    **kwargs: dict,
        Keyword arguments internally given to
        py:func:`PVPumpSystem.run_model()`. Made for giving the financial
        parameters of the project.

    Returns
    -------
    tuple
        First element is a pandas.DataFrame containing the configuration
        that minimizes the net present value (NPV) of the system. This first
        element can contain more than one configuration if multiple
        configurations have the exact same NPV which also turns to be the
        minimum.
        Second element is a pandas.DataFrame containing all configurations
        tested respecting the LLP.
    """

    # TODO: Change structure of code so as the PVPumpSystem object accepts
    # lists as attribute (for ex.: list of pumps, pv modules, reservoir, etc)
    # and that the factorial design is made across all elements of each
    # list. The scope of the sizing would be not be restricted
    # to pv modules and pumps anymore.

    # TODO: check following for a discrete optimization:
    # https://towardsdatascience.com/linear-programming-and-discrete-optimization-with-python-using-pulp-449f3c5f6e99

    if pvps_fixture.coupling == 'direct':
        preselection = subset_respecting_llp_direct(pv_database,
                                                    pump_database,
                                                    weather_data,
                                                    weather_metadata,
                                                    pvps_fixture,
                                                    llp_accepted=llp_accepted,
                                                    M_s_guess=M_s_guess,
                                                    M_p_guess=M_p_guess,
                                                    **kwargs)
    elif pvps_fixture.coupling == 'mppt':
        preselection = subset_respecting_llp_mppt(pv_database,
                                                  pump_database,
                                                  weather_data,
                                                  weather_metadata,
                                                  pvps_fixture,
                                                  llp_accepted=llp_accepted,
                                                  M_s_guess=M_s_guess,
                                                  **kwargs)
    else:
        raise ValueError('Unknown coupling method.')

    if pd.isna(preselection.npv).any():
        warnings.warn('The NPV could not be calculated, so optimized '
                      'sizing could not be found.')
        selection = preselection
    else:
        selection = preselection[preselection.npv == preselection.npv.min()]

    return (selection, preselection)


def sizing_Ms_vs_tank_size():
    """
    Function optimizing M_s and reservoir size as in [1].

    References
    ----------

    [1] Bouzidi, 2013, 'New sizing method of PV water pumping systems',
    Sustainable Energy Technologies and Assessments
    """
    raise NotImplementedError


def sizing_tank_size():
    """
    Function optimizing reservoir size. Note that this value is continuous,
    so several optimization methods from scipy could be applied
    effiently on it.
    """
    raise NotImplementedError
