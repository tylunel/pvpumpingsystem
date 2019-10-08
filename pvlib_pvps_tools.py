# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:30:41 2019

@author: Tanguy
"""
import pvlib
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from warnings import warn
import pandas as pd
import pump
import tqdm
import errors


def function_i_with_v(V, I_L, I_o, R_s, R_sh, nNsVth,
                      M_s=1, M_p=1):
    """Function I=f(V) coming from equation of Single Diode Model
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
    I : numeric
        Output current of the whole pv source, in A.

    Notes / Issues
    --------
    - According to the speed of the computations,
    it seems that the complexity of this function is cubic
    O(n^3), and therefore it takes too much time to compute this way for
    long vectors (around 45min for 8760 elements).

    - Different from pvsystem.i_from_v because it includes M_s and M_p,
    so it gives the corresponding current at the output of the array,
    not only the module.

    References
    -------
    [1] Petrone & al (2017), "Photovoltaic Sources Modeling", Wiley, p.5.
        URL: http://doi.wiley.com/10.1002/9781118755877
    """
    if (M_s, M_p) != (1, 1):
        I_L = M_p * I_L
        I_o = M_p * I_o
        nNsVth = nNsVth * M_s
        R_s = (M_s/M_p) * R_s
        R_sh = (M_s/M_p) * R_sh

    def I_pv_fct(I):
        return -I + I_L - I_o*(np.exp((V+I*R_s)/nNsVth) - 1) - (V+I*R_s)/R_sh

    Varr = np.array(V)
    Iguess = (np.zeros(Varr.size)) + I_L/2
    sol = opt.root(I_pv_fct, x0=Iguess)

    if (sol.x).any() < 0:
        warn('A current is negative. The PV module may be '
             'absorbing energy and it can lead to unusual degradation.')

    return sol.x


def functioning_point(params, modules_per_string, strings_per_inverter,
                      modelchain,
                      load_fctV, load_fctI, load_intervalV,
                      tdh,
                      plot=False, nb_pts=50, stop=8760):
    """Finds the IV functioning point(s) of the PV array and the load.

    Parameters
    ----------
    modelchain: ModelChain object -> to keep for class function
        ModelChain object which have already been run with run_model.

        Changes : vvvvv
    params
    modules_per_string
    strings_per_inverter

    load_fctV: function
        The function V=f(I) of the load directly coupled with the array.

    load_fctI: function
        The function I=f(V) of the load directly coupled with the array.

    load_intervalV: array-like
        Domain of V in load_fctI

    plot: Boolean -> to keep for class function
        Allows or not the printing of IV curves of PV system and of the load.

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

    for date, params_row in tqdm.tqdm(params.iterrows(),
                                      desc='Computing of I for given voltage',
                                      total=len(params)):

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

    pdresult = pd.DataFrame(result)
    pdresult.index = params.index

    if plot:
        plt.figure
        # domain of interest on V
        # (*1.1 is for the case when conditions are better than stc)
        v_high_boundary = modelchain.system.module.V_oc_ref * M_s * 1.1
        Vrange_pv = np.arange(0, v_high_boundary)
        # IV curve of PV array for good conditions
        params_good = modelchain.system.calcparams_desoto(1000, 25)
        Ivect_pv_good = function_i_with_v(Vrange_pv, *params_good, M_s, M_p)
        plt.plot(Vrange_pv, Ivect_pv_good,
                 label='pv array with S = 1000 W and Tcell = 25°C')
        # IV curve of PV array for poor conditions
        params_poor = modelchain.system.calcparams_desoto(100, 60)
        Ivect_pv_poor = function_i_with_v(Vrange_pv, *params_poor, M_s, M_p)
        plt.plot(Vrange_pv, Ivect_pv_poor,
                 label='pv array with S = 100 W and Tcell = 60°C')
        # IV curve of load (pump)
        Vrange_load = np.arange(load_intervalV[0], load_intervalV[1])
        plt.plot(Vrange_load,
                 load_fctI(Vrange_load, tdh, error_raising=False),
                 label='load')

        plt.legend(loc='best')
        axes = plt.gca()
        axes.set_ylim([0, None])
        plt.xlabel('voltage (V)')
        plt.ylabel('current (A)')

    return pdresult


def calc_flow_noiteration(fct_Q_from_inputs, input_1, static_head):
    """Function computing the flow at the output of the PVPS according
    to the input_1 at functioning point.

    Parameters
    ----------
    fct_Q_from_inputs: function
        Function computing flow rate of the pump in liter/minute.
        Should preferentially come from 'Pump.fct_Q_from_inputs()'

    input_1: numeric
        Voltage or Power at the functioning point.

    static_head: numeric
        Static head of the hydraulic network

    Returns
    ---------
    Q: water discharge in liter/timestep of input_1 data (typically
                                                           hours)
    """
    if not type(input_1) is float:
        input_1 = float(input_1)

    if input_1 == 0:
        Qlpm = 0
    else:
        # compute total head h_tot
        h_tot = static_head
        # compute discharge Q
        try:
            Qlpm = fct_Q_from_inputs(input_1, h_tot)
        except (errors.VoltageError, errors.PowerError):
            Qlpm = 0
    if Qlpm < 0:
        Qlpm = 0

    discharge = Qlpm  # temp: 60 should be replaced by timestep

    return discharge


#%% Code Test
if __name__ == '__main__':

    pump1 = pump.Pump(path="fichiers_pompes/SCB_10_150_120_BL.txt",
                      model='SCB_10')
    load_fctI, ectyp, intervalsVH = pump1.functIforVH()

    load_fctV, ectyp, intervalsIH = pump1.functVforIH()

    fctQwithVH, sigma2 = pump1.functQforVH()

    # Using the 5 parameters model of DeSoto through this function
    # instead of retrieving the 6 parameters from CEC database underestimate
    # the result of p_mp of 0.6% in average.
    # -> np.mean((p_mp_6param - p_mp_5param)/p_mp_6param)
#    five_params = pvlib.singlediode.getparams_from_specs_desoto(
#        I_sc=9.43, V_oc=38.3, I_mp=8.71, V_mp=31.0, alpha_sc=0.06, N_s=60,
#        beta_oc=-0.36)
#    five_params = {'I_L': 9.452306515003592,
#                   'I_o': 3.244967717785086e-10,
#                   'nNsVth': 1.59170284124638,
#                   'R_sh': 125.85822206469133,
#                   'R_s': 0.2977156014170881}
    # pvmod1=pm.PVModule(Brand='Kyocera',Model='KU270-6MCA',Ac=1.645,Price=0,
    #                    I_sc=9.43,V_oc=38.3,I_mp=8.71,V_mp=31.0,
    #                    beta_oc=-0.36,alpha_sc=0.06,N_s=60)

    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
                surface_tilt=0, surface_azimuth=180,
                albedo=0, surface_type=None,
                module=CECMOD.Kyocera_Solar_KU270_6MCA,
                module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
                                   **glass_params},
#                module_parameters={**five_params, **glass_params,
#                                   'Technology': 'c-Si'},
                modules_per_string=3, strings_per_inverter=1,
                inverter=None, inverter_parameters={'pdc0': 700},
                racking_model='open_rack_cell_glassback',
                losses_parameters=None, name=None
                )

    weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(
        './fichiers_epw/USA_CO_Denver.Intl.AP.725650_TMY3.epw')

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
                losses_model='no_loss', name=None)

    chain1.run_model(times=weatherdata1.index, weather=weatherdata1)

    stop = 8760
    fp = functioning_point(chain1.diode_params[0:stop],
                           chain1.system.modules_per_string,
                           chain1.system.strings_per_inverter,
                           chain1,
                           load_fctV, load_fctI, intervalsVH['V'],
                           40,
                           stop=8760, plot=True)

    Qtot = calc_flow_noiteration(fp.V[12], fctQwithVH, 40)
    print(Qtot)
