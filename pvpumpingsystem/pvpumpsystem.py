# -*- coding: utf-8 -*-
"""
Defines a whole PVPS, with PV array, pump, pipes, etc, and provide
functions for computing main output (water discharge,...) from input
(weather, pv array, water consumption)

@author: Tanguy Lunel

"""
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import tqdm
import time
import pvlib
import warnings

import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.finance as fin
from pvpumpingsystem import errors


# TODO: add check on dc_model and coupling_method to insure that
# direct-coupling goes with SDM. (if no check, error can be hard to fine
# for user)

# TODO: add 'finance_params' parameters?? as dict?
# it would typically include opex, discount_rate, lifespan_pv,
# lifespan_pump, lifespan_mppt, labour_price_coefficient

class PVPumpSystem(object):
    """
    Class defining a PV pumping system made of:

    Attributes
    ----------
    pvgeneration: pvpumpingsystem.PVGeneration,
        Note that the weather file used here should ideally not
        smooth the extreme conditions (avoid TMY or IWEC for example).
        The pvgeneration.modelchain.dc_model must be a Single Diode
        model if the system is directly-coupled

    motorpump: pvpumpingsystem.Pump
        The pump used in the system.

    coupling: str,
        represents the type of coupling between pv generator and pump.
        Can be 'mppt' or 'direct'

    motorpump_model: str, default None
        The modeling method used to model the motorpump. Can be:
        'kou', 'arab', 'hamidat' or 'theoretical'.
        Overwrite the motorpump.modeling_method attribute if not None.

    mppt: pvpumpingsystem.MPPT
        Maximum power point tracker of the system.

    pipes: pvpumpingsystem.PipeNetwork

    reservoir: pvpumpingsystem.Reservoir

    consumption: pvpumpingsystem.Consumption

    llp: None or float,
        Loss of Load Probability, i.e. Water shortage probability.
        It is None until computed by run_model(), and then it
        ranges between 0 and 1.

    initial_investment: None or float,
        Cost of the system at the installation [USD].
        It is None until computed by run_model()

    """
    def __init__(self,
                 pvgeneration,
                 motorpump,
                 coupling='mppt',
                 motorpump_model=None,
                 mppt=None,
                 pipes=None,
                 reservoir=None,
                 consumption=None,
                 idname=None):
        self.pvgeneration = pvgeneration  # instance of PVArray
        self.motorpump = motorpump  # instance of Pump
        self.coupling = coupling
        self.mppt = mppt

        if motorpump_model is None and motorpump is not None:
            self.motorpump_model = self.motorpump.modeling_method
        elif motorpump is not None:
            self.motorpump_model = motorpump_model
            self.motorpump.modeling_method = motorpump_model
        else:  # motorpump is None (can happen in initialization of a sizing)
            pass

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
        infos = ('PVPSystem made of: \npvgeneration: {0} \npump: {1})'
                 .format(self.pvgeneration, self.motorpump))
        return infos

    # TODO: turn it into a decorator
    def define_motorpump_model(self, model):
        if model != self.motorpump_model:
            self.motorpump.modeling_method = model
            self.motorpump_model = model
        else:
            return 'Already defined motorpump model'

#    @property  # getter
#    def motorpump_model(self):
#        return self.motorpump_model
#
#    # setter: allows to recalculate pump coeffs when changing the model
#    @motorpump_model.setter
#    def motorpump_model(self, model):
#        if model != self.motorpump.modeling_method:
#            self.motorpump.modeling_method = model
#            self.motorpump_model = model

    def operating_point_noiteration(self, plot=False, nb_pts=50, stop=8760):
        """Finds the IV operating point(s) of the PV array and the pump
        (load).

        cf pvpumpsystem.operating_point_noiteration for more details

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
        pandas.DataFrame
            Current ('I') and voltage ('V') at the operating point between
            load and pv array.

        Note / Issues
        -------------
        Takes ~10sec to compute 8760 iterations
        """
        params = self.pvgeneration.modelchain.diode_params[0:stop]
        M_s = self.pvgeneration.system.modules_per_string
        M_p = self.pvgeneration.system.strings_per_inverter

        load_fctI, intervalsVH = self.motorpump.functIforVH()

        fctQwithVH, sigma2 = self.motorpump.functQforVH()

        tdh = self.pipes.h_stat

        pdresult = operating_point_noiteration(
                params,
                M_s, M_p,
                load_fctIfromVH=load_fctI,
                load_interval_V=intervalsVH['V'](tdh),
                pv_interval_V=[
                    0, self.pvgeneration.modelchain.dc.v_oc.max() * M_s],
                tdh=tdh)

        if plot:
            plt.figure()
            # domain of interest on V
            # (*1.1 is for the case when conditions are better than stc)
            v_high_boundary = self.pvgeneration.system.module.V_oc_ref * \
                M_s*1.1
            Vrange_pv = np.arange(0, v_high_boundary)

            # IV curve of PV array for good conditions
            IL, I0, Rs, Rsh, nNsVth = \
                self.pvgeneration.system.calcparams_desoto(1000, 25)
            if (M_s, M_p) != (1, 1):
                IL = M_p * IL
                I0 = M_p * I0
                nNsVth = nNsVth * M_s
                Rs = (M_s/M_p) * Rs
                Rsh = (M_s/M_p) * Rsh
            Ivect_pv_good = self.pvgeneration.system.i_from_v(Rsh, Rs,
                                                              nNsVth,
                                                              Vrange_pv,
                                                              I0, IL)
            plt.plot(Vrange_pv, Ivect_pv_good,
                     label='pv array with S = 1000 W and Tcell = 25°C')

            # IV curve of PV array for poor conditions
            IL, I0, Rs, Rsh, nNsVth = \
                self.pvgeneration.system.calcparams_desoto(100, 60)
            if (M_s, M_p) != (1, 1):
                IL = M_p * IL
                I0 = M_p * I0
                nNsVth = nNsVth * M_s
                Rs = (M_s/M_p) * Rs
                Rsh = (M_s/M_p) * Rsh
            Ivect_pv_poor = self.pvgeneration.system.i_from_v(Rsh, Rs,
                                                              nNsVth,
                                                              Vrange_pv,
                                                              I0, IL)
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

    def calc_flow(self, atol=0.1, stop=8760, **kwargs):
        """
        Computes the flow at the output of the PVPS, and
        assigns the value to the attribute 'flow'.

        cf :py:func:`calc_flow_directly_coupled` and
        :py:func:`calc_flow_mppt_coupled` for more details.

        Parameters
        ----------
        atol: numeric
            absolute tolerance on the uncertainty of the flow in L/min

        stop: numeric
            number of data on which the computation is run

        Returns
        -------
        None

        Notes
        -----
        Takes ~20 sec for computing 8760 iterations with mppt coupling and
            atol=0.1 lpm
        Takes ~60 sec for computing 8760 iterations with direct coupling and
            atol=0.1 lpm

        """
        if self.coupling == 'mppt':
            self.flow = calc_flow_mppt_coupled(self.pvgeneration,
                                               self.motorpump,
                                               self.pipes,
                                               self.mppt,
                                               atol=atol, stop=stop,
                                               **kwargs)
        elif self.coupling == 'direct':
            self.flow = calc_flow_directly_coupled(self.pvgeneration,
                                                   self.motorpump,
                                                   self.pipes,
                                                   atol=atol, stop=stop,
                                                   **kwargs)
        else:
            raise ValueError("Inappropriate value for argument coupling." +
                             "It should be 'mppt' or 'direct'.")

    def calc_efficiency(self):
        """
        Computes the efficiencies between PV array output and
        motorpump output, between irradiance and PV output, and global
        efficiency. Assigns the resulting data to the attribute 'efficiency'.

        cf :py:func:`calc_efficiency` for more details

        Returns
        -------
        None

        """

        module_area = self.pvgeneration.system.module.A_c
        M_s = self.pvgeneration.system.modules_per_string
        M_p = self.pvgeneration.system.strings_per_inverter
        pv_area = module_area * M_s * M_p

        if self.flow is None:
            self.calc_flow()

        self.efficiency = calc_efficiency(
            self.flow,
            self.pvgeneration.modelchain.effective_irradiance,
            pv_area)

    def calc_reservoir(self, starting_soc='morning'):
        """
        Computes the water volume in the reservoir and extra or
        lacking water compared to the consumption at any time step.
        Assigns the resulting data to the attribute 'flow'.

        cf :py:func:`calc_reservoir` for more details

        Parameters
        ----------
        starting_soc: str or float, default is 'morning'
            State of Charge of the reservoir at the beginning of
            the simulation [%].
            Available strings are 'empty' (no water in reservoir),
            'morning' (enough water for one morning consumption) and 'full'.

        Return
        ------
        None

        """
        # set an empty water reservoir at beginning
        if starting_soc == 'empty':
            self.reservoir.water_volume = 0

        # set reservoir with enough water to fulfil the need of one morning
        elif starting_soc == 'morning':
            # Get water needed in the first morning (until 12am)
            vol = float((self.consumption.flow_rate.iloc[0:12]*60).sum())
            # initialization of water in reservoir
            self.reservoir.water_volume = vol

        # set a full water reservoir at beginning
        elif starting_soc == 'full':
            self.reservoir.water_volume = self.reservoir.size

        # set water level to the one given
        elif isinstance(starting_soc, float) and self.reservoir.size != np.inf:
            self.reservoir.water_volume = self.reservoir.size * starting_soc

        # exception handling
        else:
            raise TypeError('starting_soc type is not correct, or is '
                            'incoherent with reservoir.size')

        if self.flow is None:
            self.calc_flow()

        self.water_stored = calc_reservoir(self.reservoir, self.flow.Qlpm,
                                           self.consumption.flow_rate.Qlpm)

    def run_model(self, iteration=False, starting_soc='morning', **kwargs):
        """
        Comprehensive modeling of the PVPS. Computes Loss of Power Supply (LLP)
        and stores it as an attribute. Re-run eveything even if already
        computed before.

        Parameters
        ----------
        iteration: boolean, default is False
            Decide if the friction head is taken into account in the
            computation. Turning it to True multiply by three the
            calculation time.

        **kwargs:
            Keyword arguments that apply to the financial analysis.
            kwargs are transfered to fin.net_present_value() function.

        """

        self.pvgeneration.run_model()

        # 'disable' removes the progress bar
        self.calc_flow(disable=True, iteration=iteration)
        self.calc_efficiency()
        self.calc_reservoir(starting_soc=starting_soc)

        self.consumption.flow_rate = cs.adapt_to_flow_pumped(
                self.consumption.flow_rate,
                self.flow.Qlpm)

        total_water_required = sum(self.consumption.flow_rate.Qlpm*60)
        total_water_lacking = -sum(self.water_stored.extra_water[
                self.water_stored.extra_water < 0])

        # water shortage probability
        try:
            self.llp = total_water_lacking / total_water_required
        except ZeroDivisionError:
            self.llp = np.nan

        # Price of motorpump, pv modules, reservoir, mppt
        self.initial_investment = fin.initial_investment(self, **kwargs)
        self.npv = fin.net_present_value(self, **kwargs)


def function_i_from_v(V, I_L, I_o, R_s, R_sh, nNsVth,
                      M_s=1, M_p=1):
    """
    Deprecated:
    'function_i_from_v' deprecated. Use pvlib.pvsystem.i_from_v instead

    Function I=f(V) coming from equation of Single Diode Model
    with parameters adapted to the irradiance and temperature.

    The adaptation of the 5 parameters from module parameters to array
    parameters is made according to [1].

    Parameters
    ----------
    V: numeric
        Voltage at which the corresponding current is to be calculated in volt.

    I_L: numeric
        The light-generated current (or photocurrent) in amperes.

    I_o: numeric
        The dark or diode reverse saturation current in amperes.

    nNsVth: numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    R_sh: numeric
        The shunt resistance in ohms.

    R_s: numeric
        The series resistance in ohms.

    M_s: numeric
        The number of module in series in the whole pv system.
        (= modules_per_strings)

    M_p: numeric
        The number of module in parallel in the whole pv system.
        (= strings_per_inverter)

    Returns
    -------
    numeric
        Output current of the whole pv source, in A.

    Notes
    -----
    According to the speed of the computations,
    it seems that the complexity of this function is cubic
    O(n^3), and therefore it takes too much time to compute this way for
    long vectors (around 45min for 8760 elements).

    Different from pvsystem.i_from_v because it includes M_s and M_p,
    so it gives the corresponding current at the output of the array,
    not only the module.

    References
    ----------
    [1] Petrone & al (2017), "Photovoltaic Sources Modeling", Wiley, p.5.
        URL: http://doi.wiley.com/10.1002/9781118755877
    """
    warnings.warn(
        "'function_i_from_v' deprecated. Use pvlib.pvsystem.i_from_v instead",
        DeprecationWarning)

    if (M_s, M_p) != (1, 1):
        I_L = M_p * I_L
        I_o = M_p * I_o
        nNsVth = nNsVth * M_s
        R_s = (M_s/M_p) * R_s
        R_sh = (M_s/M_p) * R_sh

    def I_pv_fct(I):  # noqa: E741
        return -I + I_L - I_o*(np.exp((V+I*R_s)/nNsVth) - 1) - (V+I*R_s)/R_sh

    Varr = np.array(V)
    Iguess = (np.zeros(Varr.size)) + I_L/2
    sol = opt.root(I_pv_fct, x0=Iguess)

    if (sol.x).any() < 0:
        warnings.warn('A current is negative. The PV module may be absorbing '
                      'energy and it can lead to unusual degradation.')

    return sol.x


# TODO: simplify function for removing C901 lint error
def operating_point_noiteration(  # noqa: C901
        params,
        modules_per_string,
        strings_per_inverter,
        load_fctIfromVH=None,
        load_interval_V=[-np.inf, np.inf],
        pv_interval_V=[-np.inf, np.inf],
        tdh=0):
    """
    Finds the IV operating point(s) between PV array and load (motor-pump).

    Parameters
    ----------
    params: pandas.Dataframe
        Dataframe containing the 5 diode parameters. Typically comes from
        PVGeneration.ModelChain.diode_params

    modules_per_string: numeric
        Number of modules in series in a string

    strings_per_inverter: numeric
        Number of strings in parallel

    load_fctIfromVH: function
        The function I=f(V, H) of the load directly coupled with the array.

    tdh: numeric
        Total dynamic head

    Returns
    -------
    pandas.DataFrame
        Current ('I') and voltage ('V') at the operating point between
        load and pv array. I and V are float. It is 0 when there is no
        irradiance, and np.nan when pv array and load don't match.

    Notes
    -----
    Takes ~10sec for computing 8760 iterations
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

        if np.isnan(I_L) or I_L == 0:
            Vm = 0
            Im = 0
        else:
            # attempt to use only load_fctI here
            def pv_fctI(V):  # does not work
                return pvlib.pvsystem.i_from_v(R_sh, R_s, nNsVth, V,
                                               I_o, I_L, method='lambertw')

            def load_fctI(V):
                return load_fctIfromVH(V, tdh, error_raising=False)

            # finds intersection. 10 is starting estimate, could be improved
#            Vm = opt.fsolve(lambda v: pv_fctI(v) - load_fctI(v), 10)
            try:
                Vm = opt.brentq(lambda v: pv_fctI(v) - load_fctI(v),
                                load_interval_V[0], pv_interval_V[1])
            except ValueError as e:
                if 'f(a) and f(b) must have different signs' in str(e):
                    # basically means that there is no operating point
                    Im = np.nan
                    Vm = np.nan
                else:
                    raise
            try:
                Im = load_fctIfromVH(Vm, tdh, error_raising=True)
            except (errors.VoltageError, errors.HeadError):
                Im = np.nan
                Vm = np.nan

        result.append({'I': Im,
                       'V': Vm})

    # conversion in pd.DataFrame
    pdresult = pd.DataFrame(result)
    pdresult.index = params.index

    return pdresult


def calc_flow_directly_coupled(pvgeneration, motorpump, pipes,
                               iteration=False,
                               atol=0.1,
                               stop=8760,
                               **kwargs):
    """
    Computes input electrical characteristics, total dynamic head,
    and flow at pump output.

    Parameters
    ----------
    pvgeneration: pvpumpingsystem.pvgeneration.PVGeneration object
        The PV generator object

    motorpump: pump.Pump object
        Pump associated with the PV generator

    pipes: pipenetwork.PipeNetwork object
        Hydraulic network linked to the pump

    iteration: boolean, default is False
        Decide if the system takes into account the friction head due to the
        flow rate of water pump (iteration = True) or if the system just
        considers the static head of the system (iteration = False).
        Often can be put to False if the pipes are well sized.

    atol: numeric
        absolute tolerance on the uncertainty of the flow in l/min.
        Used if iteration = True.

    stop: numeric
        number of data on which the computation is run

    Returns
    -------
    df: pandas.DataFrame,
        pd.Dataframe with following attributes:
            'I': Current in A at operating point
            'V': Voltage in V at operating point
            'Qlpm': Flow rate of water in L/minute
            'P': Input power to the pump in W
            'P_unused': Power unused (because too low or too high)
            'tdh': Total dynamic head in m

    Notes
    -----
    Takes ~15 sec for computing 8760 iterations with atol=0.1lpm

    """
    result = []
    modelchain = pvgeneration.modelchain
    # retrieve specific functions of motorpump V(I,H) and Q(V,H)
    M_s = modelchain.system.modules_per_string
    M_p = modelchain.system.strings_per_inverter

    load_fctIfromVH, intervalsVH = motorpump.functIforVH()
    fctQwithPH, sigma2 = motorpump.functQforPH()

    for i, row in tqdm.tqdm(enumerate(
            modelchain.diode_params[0:stop].iterrows()),
                                      desc='Computing of Q',
                                      total=stop,
                                      **kwargs):
        params = row[1]

        if iteration is True:
            Qlpm = 1
            Qlpmnew = 0

            # variables used in case of non-convergence in while loop
            t_init = time.time()
            mem = []

            while abs(Qlpm-Qlpmnew) > atol:  # loop to make Qlpm converge
                Qlpm = Qlpmnew
                # water temperature (arbitrary, should vary in future work)
                temp_water = 10
                # compute total head h_tot
                h_tot = pipes.h_stat + \
                    pipes.dynamichead(Qlpm, T=temp_water)
                # compute operating point
                iv_data = operating_point_noiteration(
                        params, M_s, M_p,
                        load_fctIfromVH=load_fctIfromVH,
                        load_interval_V=intervalsVH['V'](h_tot),
                        pv_interval_V=[0, modelchain.dc.v_oc[i] * M_s],
                        tdh=h_tot)
                # consider losses
                power = iv_data.V*iv_data.I * modelchain.losses
                # type casting
                power = float(power)
                # compute flow
                Qlpmnew = fctQwithPH(power, h_tot)['Q']

                # code for exiting while loop if problem
                mem.append(Qlpmnew)
                if time.time()-t_init > 1000:
                    print('\niv:', iv_data)
                    print('Q:', mem)
                    raise RuntimeError('Loop too long to execute')

            P_unused = fctQwithPH(power, h_tot)['P_unused']

            result.append({'Qlpm': Qlpmnew,
                           'I': float(iv_data.I),
                           'V': float(iv_data.V),
                           'P': power,
                           'P_unused': P_unused,
                           'tdh': h_tot
                           })
        else:  # iteration is False
            # compute operating point
            iv_data = operating_point_noiteration(
                    params, M_s, M_p,
                    load_fctIfromVH=load_fctIfromVH,
                    load_interval_V=intervalsVH['V'](pipes.h_stat),
                    pv_interval_V=[0, modelchain.dc.v_oc[i] * M_s],
                    tdh=pipes.h_stat)
            # consider losses
            power = iv_data.V*iv_data.I * modelchain.losses
            # type casting
            power = float(power)
            # compute flow
            res_dict = fctQwithPH(power, pipes.h_stat)
            Qlpm = res_dict['Q']
            P_unused = res_dict['P_unused']

            result.append({'Qlpm': Qlpm,
                           'I': float(iv_data.I),
                           'V': float(iv_data.V),
                           'P': power,
                           'P_unused': P_unused,
                           'tdh': pipes.h_stat
                           })

    pdresult = pd.DataFrame(result)
    pdresult.index = modelchain.diode_params[0:stop].index
    return pdresult


def calc_flow_mppt_coupled(pvgeneration, motorpump, pipes, mppt,
                           iteration=False,
                           atol=0.1,
                           stop=8760,
                           **kwargs):
    """
    Computes input electrical characteristics, total dynamic head,
    and flow at pump output.

    Parameters
    ----------
    pvgeneration: pvpumpingsystem.pvgeneration.PVGeneration
        The PV generator.

    motorpump: pump.Pump object
        Pump associated with the PV generator

    pipes: pipenetwork.PipeNetwork object
        Hydraulic network linked to the pump

    mppt: mppt.MPPT object,
        The maximum power point tracker of the system.

    iteration: boolean, default is False
        Decide if the system takes into account the friction head due to the
        flow rate of water pump (iteration = True) or if the system just
        considers the static head of the system (iteration = False).
        Often can be put to False if the pipes are well sized.

    atol: numeric
        absolute tolerance on the uncertainty of the flow in l/min.
        Used if iteration=True.

    stop: numeric
        number of data on which the computation is run

    Returns
    -------
    df: pandas.DataFrame
        pd.Dataframe with following attributes:
            'Qlpm': Flow rate of water in L/minute
            'P': Input power to the pump in W
            'P_unused': Power unused (because too low or too high)
            'tdh': Total dynamic head in m

    Notes
    -----
    Takes ~15 sec for computing 8760 iterations with atol=0.1lpm

    """
    result = []
    modelchain = pvgeneration.modelchain
    if mppt is None:
        # note that dc already includes losses from modelchain.losses_model
        power_available = modelchain.dc.p_mp[0:stop]
    else:
        power_available = modelchain.dc.p_mp[0:stop] * mppt.efficiency

    fctQwithPH, sigma2 = motorpump.functQforPH()

    for i, power in tqdm.tqdm(enumerate(power_available),
                              desc='Computing of Q',
                              total=stop,
                              **kwargs):

        if iteration is True:
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
                Qlpmnew = fctQwithPH(power, h_tot)['Q']

                # code for exiting while loop if problem happens
                mem.append(Qlpmnew)
                if time.time() - t_init > 0.1:
                    warnings.warn('Loop too long to execute. NaN returned.')
                    Qlpmnew = np.nan
                    break

            P_unused = fctQwithPH(power, h_tot)['P_unused']

            result.append({'Qlpm': Qlpmnew,
                           'P': float(power),
                           'P_unused': P_unused,
                           'tdh': h_tot
                           })
        else:  # iteration is False
            # simply computes flow
            res_dict = fctQwithPH(power, pipes.h_stat)
            Qlpm = res_dict['Q']
            P_unused = res_dict['P_unused']

            result.append({'Qlpm': Qlpm,
                           'P': float(power),
                           'P_unused': P_unused,
                           'tdh': pipes.h_stat
                           })

    pdresult = pd.DataFrame(result)
    pdresult.index = modelchain.diode_params[0:stop].index
    return pdresult


def calc_efficiency(df, irradiance, pv_area):
    """
    Computes the efficiencies between PV array output and
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
    pandas.DataFrame
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
    """
    Computes the water volume in the reservoir and the extra or
    lacking water compared to the consumption at any time step.

    Parameters
    ----------
    Q_pumped: pd.DataFrame
        Dataframe containing the reservoir input flow-rate [L/min]

    Q_consumption: pd.DataFrame
        Dataframe containing the reservoir output flow-rate [L/min]

    reservoir: Reservoir object
        The reservoir of the system.

    Returns
    -------
    pandas.DataFrame
        Dataframe with water_volume in tank, and extra or lacking water.
    """

    level = []
    # timestep of flowrate dataframe Q_lpm_df
    timestep = Q_pumped.index[1] - Q_pumped.index[0]
    timestep_minute = timestep.seconds/60

    Q_consumption = cs.adapt_to_flow_pumped(Q_consumption, Q_pumped)

    # replace nan by 0 for computation of Q_diff
    Q_pumped.fillna(value=0, inplace=True)

    # diff in volume
    Q_diff = Q_pumped - Q_consumption

    # total change in volume during the timestep in liters
    volume_diff = Q_diff * timestep_minute

    for vol in volume_diff:
        level.append(reservoir.change_water_volume(vol))

    water_stored = pd.DataFrame(level, columns=('volume', 'extra_water'))
    water_stored.index = Q_pumped.index

    return water_stored
