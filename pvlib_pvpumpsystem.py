# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:54:44 2019

@author: Tanguy

Defines a whole PVPS, with PV array, pump, pipes... and provide
functions for computing main output (water discharge,...) from input
(weather,...)
"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import tqdm
import time
import pvlib
import pvlib_pvps_tools

import pump as pp
import pipenetwork as pn
import reservoir as rv
import consumption as cs
import errors


class PVPumpSystem(object):
    """Class defining a PV pumping system made of :
            - modelchain : class pvlib.ModelChain
            - motorpump : class Pump
            - coupling: 'mppt' or 'direct'
                represents the type of coupling between pv generator and pump
            - pipes : class PipeNetwork
            - reservoir: class Reservoir
            - consumption: class Consumption

    """
    def __init__(self, modelchain, motorpump, coupling='mppt',
                 pipes=None, reservoir=None, consumption=None, idname=None):
        self.modelchain = modelchain  # instance of PVArray
        self.motorpump = motorpump  # instance of Pump
        self.coupling = coupling

        if pipes is None:
            self.pipes = pn.PipeNetwork(0, 0, 0.1)  # system with null length
        else:
            self.pipes = pipes  # instance of PipeNetwork

        if reservoir is None:
            self.reservoir = rv.Reservoir()  # instance of Reservoir
        else:
            self.reservoir = reservoir  # instance of Reservoir

        if consumption is None:
            self.consumption = cs.Consumption()  # instance of Consumption
        else:
            self.consumption = consumption  # instance of Consumption

        self.idname = idname  # instance of String

        # To be calculated later
        self.flow = None
        self.efficiency = None
        self.water_stored = None

    def __repr__(self):
        if self.idname is None:
            infos = ('PVPSystem made of : \n modelchain: {0} \npump: {1})'
                     .format(self.modelchain, self.motorpump.model))
            return infos
        else:
            return ('PV Pumping System :'+self.idname)

    def functioning_point_noiteration(self,
                                      plot=False, nb_pts=50, stop=8760):
        """Finds the IV functioning point(s) of the PV array and the load.

        cf pvlib_pvpumpsystem.functioning_point_noiteration for more details

        Parameters
        ----------

        plot: Boolean
            Allows or not the printing of IV curves of PV system and of
            the load.

        nb_pts: numeric
            number of points on graph

        stop: numeric
                number of data on which the computation is run

        Returns
        -------
        IV : pandas.DataFrame
            Current ('I') and voltage ('V') at the functioning point between
            load and pv array.

        Note / Issues
        -------------
        - takes ~5sec for computing 8760 iterations
        """
        params = self.modelchain.diode_params[0:stop]
        M_s = self.modelchain.system.modules_per_string
        M_p = self.modelchain.system.strings_per_inverter

        load_fctI, ectyp, intervalsVH = self.motorpump.functIforVH()
        load_fctV, ectyp, intervalsIH = self.motorpump.functVforIH()
        fctQwithVH, sigma2 = self.motorpump.functQforVH()

        tdh = self.pipes.h_stat

        pdresult = functioning_point_noiteration(
                params, M_s, M_p, load_fctV, load_fctI, intervalsVH['V'], tdh)

        if plot:
            plt.figure()
            # domain of interest on V
            # (*1.1 is for the case when conditions are better than stc)
            v_high_boundary = self.modelchain.system.module.V_oc_ref * M_s*1.1
            Vrange_pv = np.arange(0, v_high_boundary)
            # IV curve of PV array for good conditions
            params_good = self.modelchain.system.calcparams_desoto(1000, 25)
            Ivect_pv_good = pvlib_pvps_tools.function_i_with_v(Vrange_pv,
                                                               *params_good,
                                                               M_s, M_p)
            plt.plot(Vrange_pv, Ivect_pv_good,
                     label='pv array with S = 1000 W and Tcell = 25°C')
            # IV curve of PV array for poor conditions
            params_poor = self.modelchain.system.calcparams_desoto(100, 60)
            Ivect_pv_poor = pvlib_pvps_tools.function_i_with_v(Vrange_pv,
                                                               *params_poor,
                                                               M_s, M_p)
            plt.plot(Vrange_pv, Ivect_pv_poor,
                     label='pv array with S = 100 W and Tcell = 60°C')
            # IV curve of load (pump)
            Vrange_load = np.arange(*intervalsVH['V'](tdh))
            plt.plot(Vrange_load,
                     load_fctI(Vrange_load, tdh, error_raising=False),
                     label='load at static head = {0}'.format(tdh))

            plt.legend(loc='best')
            axes = plt.gca()
            axes.set_ylim([0, None])
            plt.xlabel('voltage (V)')
            plt.ylabel('current (A)')

        return pdresult

    def calc_flow(self, atol=0.1, stop=8760):
        """Function computing the flow at the output of the PVPS.
        cf pvlib_pvpumpsystem.calc_flow_directly_coupled for more details

        Parameters
        ----------
        atol: numeric
            absolute tolerance on the uncertainty of the flow in L/min

        stop: numeric
            number of data on which the computation is run

        Returns
        --------
        df : pandas.DataFrame
            pd.Dataframe with following attributes:
                'I': Current in A at the functioning point between
                    load and pv array.
                'V': Voltage in V at the functioning point.
                'Qlpm': Flow rate of water in L/minute

        Note / Issues
        ---------
        - takes ~15 sec for computing 8760 iterations with atol=0.1lpm
        """
        if self.coupling == 'mppt':
            self.flow = calc_flow_mppt_coupled(self.modelchain,
                                               self.motorpump,
                                               self.pipes,
                                               atol=atol, stop=stop)
        elif self.coupling == 'direct':
            self.flow = calc_flow_directly_coupled(self.modelchain,
                                                   self.motorpump,
                                                   self.pipes,
                                                   atol=atol, stop=stop)
        else:
            raise ValueError("Inappropriate value for argument coupling." +
                             "It should be 'mppt' or 'direct'.")

    def calc_efficiency(self):
        """Function computing the efficiencies between PV array output and
        motorpump output, between irradiance and PV output, and global
        efficiency.

        Parameters
        ----------
        self

        Returns
        -------
        self.efficiency: pd.DataFrame
            Dataframe with efficiencies

        """

        module_area = self.modelchain.system.module.A_c
        M_s = self.modelchain.system.modules_per_string
        M_p = self.modelchain.system.strings_per_inverter
        pv_area = module_area * M_s * M_p

        self.efficiency = calc_efficiency(self.flow,
                                          self.modelchain.effective_irradiance,
                                          pv_area)

    def calc_reservoir(self):
        """Wrapper of pvlib.pvpumpsystem.calc_reservoir.
        """
        if self.flow is None:
            self.calc_flow()
        self.water_stored = calc_reservoir(self.reservoir, self.flow.Qlpm,
                                           self.consumption.flow.consumption)


def functioning_point_noiteration(params, modules_per_string,
                                  strings_per_inverter,
                                  load_fctV, load_fctI=None,
                                  load_intervalV=None,
                                  tdh=0):
    """Finds the IV functioning point(s) of the PV array and the load.

    Parameters
    ----------
    params: pd.Dataframe
        Dataframe containing the 5 diode parameters. Typically comes from
        ModelChain.diode_params

    modules_per_string: numeric
        Number of modules in series in a string

    strings_per_inverter: numeric
        Number of strings in parallel

    load_fctV: function
        The function V=f(I) of the load directly coupled with the array.

    load_fctI: function
        The function I=f(V) of the load directly coupled with the array.
        Only useful if plot = True.

    load_intervalV: array-like
        Domain of V in load_fctI. Only useful if plot = True.

    tdh: numeric
        Total dynamic head

    Returns
    -------
    IV : pandas.DataFrame
        Current ('I') and voltage ('V') at the functioning point between
        load and pv array.

    Note / Issues
    ---------
    - takes ~5sec for computing 8760 iterations
    """
    result = []

    if type(params) is pd.Series:
        params = pd.DataFrame(params)
        params = params.transpose()

    for date, params_row in params.iterrows():

        I_L = params_row.I_L
        I_o = params_row.I_o
        R_s = params_row.R_s
        R_sh = params_row.R_sh
        nNsVth = params_row.nNsVth

        M_s = modules_per_string
        M_p = strings_per_inverter

        if (M_s, M_p) != (1, 1):
            I_L = M_p * I_L
            I_o = M_p * I_o
            nNsVth = nNsVth * M_s
            R_s = (M_s/M_p) * R_s
            R_sh = (M_s/M_p) * R_sh

        if np.isnan(I_L):
            result.append({'I': 0, 'V': 0})
        else:
            # def of equation of Single Diode Model with previous params:
            def I_pv_fct(I):
                """Returns the electrical equation of single-diode model in
                a way that it returns 0 when the equation is respected
                (i.e. the I is the good one)
                """
                V = load_fctV(I, tdh, error_raising=False)
                In = I_L - I_o*(np.exp((V + I*R_s)/nNsVth) - 1) - \
                    (V + I*R_s)/R_sh
                y = In - I
                return y

            # solver
            try:
                Im = opt.brentq(I_pv_fct, 0, I_L)
                Im = max(Im, 0)
                Vm = load_fctV(Im, tdh, error_raising=True)
            except ValueError:
                Im = float('nan')
                Vm = float('nan')
            except (errors.CurrentError, errors.HeadError):
                Im = float('nan')
                Vm = float('nan')

            result.append({'I': Im,
                           'V': Vm})
    # conversion in pd.DataFrame
    pdresult = pd.DataFrame(result)
    pdresult.index = params.index

    return pdresult


def calc_flow_directly_coupled(modelchain, motorpump, pipes,
                               atol=0.1,
                               stop=8760):
    """DEBUG VERSION of 'calc_flow_directly_coupled'

    Function computing the flow at the output of the PVPS.

    Parameters
    ----------
    modelchain: pvlib.modelchain.ModelChain object
        Chain of modeling of the PV generator.

    motorpump: pump.Pump object
        Pump associated with the PV generator

    pipes: pipenetwork.PipeNetwork object
        Hydraulic network linked to the pump

    atol: numeric
        absolute tolerance on the uncertainty of the flow in l/min

    stop: numeric
        number of data on which the computation is run

    Returns
    --------
    df : pandas.DataFrame
        pd.Dataframe with following attributes:
            'I': Current in A at the functioning point between
                load and pv array.
            'V': Voltage in V at the functioning point.
            'Qlpm': Flow rate of water in L/minute

    Note / Issues
    ---------
    - takes ~15 sec for computing 8760 iterations with atol=0.1lpm
    """
    result = []
    # retrieve specific functions of motorpump V(I,H) and Q(V,H)
    M_s = modelchain.system.modules_per_string
    M_p = modelchain.system.strings_per_inverter

    load_fctI, ectypI, intervalsVH = motorpump.functIforVH()
    load_fctV, ectypV, intervalsIH = motorpump.functVforIH()
    fctQwithPH, sigma2 = motorpump.functQforPH()

    for i, row in tqdm.tqdm(enumerate(
            modelchain.diode_params[0:stop].iterrows()),
                                      desc='Computing of Q',
                                      total=stop):

        params = row[1]
        Qlpm = 1
        Qlpmnew = 0

        # variables used in case of non-convergence in while loop
        t_init = time.time()
        mem = []

        while abs(Qlpm-Qlpmnew) > atol:  # loop to make Qlpm converge
            Qlpm = Qlpmnew
            # water temperature (random...)
            temp_water = 10
            # compute total head h_tot
            h_tot = pipes.h_stat + \
                pipes.dynamichead(Qlpm, T=temp_water)
            # compute functioning point
            iv_data = functioning_point_noiteration(params, M_s, M_p,
                                                    load_fctV, None, None,
                                                    h_tot)
            # consider losses
            if modelchain.losses != 1:
                power = iv_data.V*iv_data.I * modelchain.losses
            else:
                power = iv_data.V*iv_data.I
            # compute flow
            Qlpmnew = pvlib_pvps_tools.calc_flow_noiteration(fctQwithPH,
                                                             power,
                                                             h_tot)

            # code for exiting while loop if problem
            mem.append(Qlpmnew)
            if time.time()-t_init > 1:
                print('\niv:', iv_data)
                print('Q:', mem)
                raise ValueError('Loop too long to execute')
        result.append({'Qlpm': Qlpmnew,
                       'I': float(iv_data.I),
                       'V': float(iv_data.V),
                       'P': float(power),
                       'tdh': h_tot
                       })

    pdresult = pd.DataFrame(result)
    pdresult.index = modelchain.diode_params[0:stop].index
    return pdresult


def calc_flow_mppt_coupled(modelchain, motorpump, pipes,
                           atol=0.1,
                           stop=8760):
    """Function computing the flow at the output of the PVPS.

    Parameters
    ----------
    modelchain: pvlib.modelchain.ModelChain object
        Chain of modeling of the PV generator.

    motorpump: pump.Pump object
        Pump associated with the PV generator

    pipes: pipenetwork.PipeNetwork object
        Hydraulic network linked to the pump

    atol: numeric
        absolute tolerance on the uncertainty of the flow in l/min

    stop: numeric
        number of data on which the computation is run

    Returns
    --------
    df : pandas.DataFrame
        pd.Dataframe with following attributes:
            'I': Current in A at the functioning point between
                load and pv array.
            'V': Voltage in V at the functioning point.
            'Qlpm': Flow rate of water in L/minute

    Note / Issues
    ---------
    - takes ~15 sec for computing 8760 iterations with atol=0.1lpm
    """
    result = []

    load_fctI, ectypI, intervalsVH = motorpump.functIforVH()
    load_fctV, ectypV, intervalsIH = motorpump.functVforIH()
    fctQwithPH, sigma2 = motorpump.functQforPH()

    for i, power in tqdm.tqdm(enumerate(
            modelchain.dc.p_mp[0:stop]),
                                      desc='Computing of Q',
                                      total=stop):

        Qlpm = 1
        Qlpmnew = 0

        # variables used in case of non-convergence in while loop
        t_init = time.time()
        mem = []

        while abs(Qlpm-Qlpmnew) > atol:  # loop to make Qlpm converge
            Qlpm = Qlpmnew
            # water temperature (random...)
            temp_water = 10
            # compute total head h_tot
            h_tot = pipes.h_stat + \
                pipes.dynamichead(Qlpm, T=temp_water)
            # compute flow
            Qlpmnew = pvlib_pvps_tools.calc_flow_noiteration(fctQwithPH,
                                                             power,
                                                             h_tot)

            # code for exiting while loop if problem
            mem.append(Qlpmnew)
            if time.time() - t_init > 1:
                print('\nP:', power)
                print('Q:', mem)
                raise ValueError('Loop too long to execute.')

        result.append({'Qlpm': Qlpmnew,
                       'P': float(power),
                       'tdh': h_tot
                       })

    pdresult = pd.DataFrame(result)
    pdresult.index = modelchain.diode_params[0:stop].index
    return pdresult


def calc_efficiency(df, irradiance, pv_area):
    """Function computing the efficiencies between PV array output and
    motorpump output, between irradiance and PV output, and global
    efficiency.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing at least:
            electric power 'P'
            flow-rate 'Qlpm'
            total dynamic head 'tdh'

    irradiance: pd.DataFrame
        Dataframe containing irradiance on PV

    pv_area: numeric
        Suface of PV collectors

    Returns
    -------
    efficiencies_df: pd.DataFrame
        Dataframe with efficiencies

    """
    electric_power_in = df.P

    g = 9.804  # in m/s2
    density = 1000  # in kg/m3
    hydraulic_power = df.tdh*g*density * df.Qlpm/(60*1000)

    pump_efficiency = hydraulic_power/electric_power_in

    irrad_power = irradiance * pv_area

    pv_efficiency = electric_power_in/irrad_power

    return pd.concat({'electric_power': electric_power_in,
                      'hydraulic_power': hydraulic_power,
                      'irrad_power': irrad_power,
                      'pump_efficiency': pump_efficiency,
                      'pv_efficiency': pv_efficiency,
                      'total_efficiency': pump_efficiency*pv_efficiency},
                     axis=1)


def calc_reservoir(reservoir, Q_pumped, Q_consumption):
    """Function computing the water volume in the reservoir and the extra or
    lacking water compared to the consumption.

    Parameters
    ----------
    Q_pumped: pd.DataFrame
        Dataframe containing the reservoir input flow-rate in liter per minute

    Q_consumption: pd.DataFrame
        Dataframe containing the reservoir output flow-rate in liter per minute

    reservoir: Reservoir object
        The reservoir of the system.

    Returns
    -------
    level_df: pd.DataFrame
        Dataframe with water_volume in tank, and extra or lacking water.
    """
    level = []
    # timestep of in flowrate dataframe Q_lpm_df
    timestep = Q_pumped.index[1]-Q_pumped.index[0]
    timestep_minute = timestep.seconds/60

    # TODO: temporary: should be replaced by process in Consumption class
    timezone = Q_pumped.index.tz
    Q_consumption.index = Q_consumption.index.tz_localize(timezone)

    # diff in volume
    Q_diff = Q_pumped - Q_consumption

    # total change in volume during the timestep in liters
    volume_diff = Q_diff * timestep_minute

    for vol in volume_diff:
        level.append(reservoir.change_water_volume(vol))

    level_df = pd.DataFrame(level, columns=('volume', 'extra_water'))
    level_df.index = Q_pumped.index

    return level_df


#%% Code Test
if __name__ == '__main__':

    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
                surface_tilt=0, surface_azimuth=180,
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

#    print(chain1.effective_irradiance.loc[chain1.effective_irradiance<0])

    pump1 = pp.Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                    model='SCB_10')
    pipes1 = pn.PipeNetwork(40, 100, 0.08, material='glass', optimism=True)
    consumption1 = cs.Consumption(constant_flow=1)
    reservoir1 = rv.Reservoir(10000, 0)

    pvps1 = PVPumpSystem(chain1, pump1, coupling='direct', pipes=pipes1,
                         consumption=consumption1, reservoir=reservoir1)
#    iv = pvps1.functioning_point_noiteration(plot=True)
#    print(iv)

#    res1 = calc_flow_directly_coupled(chain1, pump1, pipes1, atol=0.01,
#                                      stop=8760)
#    res2 = calc_flow_mppt_coupled(chain1, pump1, pipes1, atol=0.01,
#                                  stop=8760)
#    compare = pd.DataFrame({'direct1': res1.Qlpm,
#                            'mppt': res2.Qlpm})
#    eff1 = calc_efficiency(res1, chain1.effective_irradiance, pv_area)
#    eff2 = calc_efficiency(res2, chain1.effective_irradiance, pv_area)

    pvps1.calc_flow()
    pvps1.calc_efficiency()
    pvps1.calc_reservoir()

#    plt.plot(pvps1.water_stored.index, pvps1.water_stored.volume)
#    plt.plot(pvps1.efficiency.index, pvps1.efficiency.electric_power)
#    plt.plot(pvps1.efficiency.index, pvps1.flow.Qlpm)
#    plt.plot(pvps1.efficiency.index, pvps1.modelchain.effective_irradiance)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('time')
    ax1.set_ylabel('Water volume in tank [L]', color='r')
    ax1.plot(pvps1.water_stored.index, pvps1.water_stored.volume, color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Pump output flow-rate [L/min]', color='b')
    ax2.plot(pvps1.efficiency.index, pvps1.flow.Qlpm, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
