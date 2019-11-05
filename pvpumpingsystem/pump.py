# -*- coding: utf-8 -*-
"""
Created on Fri May 17 07:54:42 2019

@author: Sergio Gualteros, Tanguy Lunel

module defining class and functions for modeling the pump.

"""
import collections
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog as tkfile
from itertools import count
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for plotting in 3d
import scipy.optimize as opt
import scipy.interpolate as spint

from pvpumpingsystem import errors
from pvpumpingsystem import function_models


class Pump:
    """
    Class representing a pump.

    Attributes
    ----------
        path: str,
            The path to the txt file with specifications. Can be given
            through constructor or through pop-up window.

        lpm: dict
            Dictionary of flow rate (values) [liter per minute] according to
            voltage (keys) [V]
        tdh: dict
            Dictionary of total dynamic head (values) [m]
            according to voltage (keys) [V]
        current: dict
            Dictionary of current (values) [A]
            according to voltage (keys) [V]
        voltage: list
            list of voltage (the keys of preceding dictionaries) [V]

        motor_electrical_architecture: str,
            'permanent_magnet', 'series_excited', 'shunt_excited',
            'separately_excited'.
        pump_category: str,
            centrifugal or positive displacement
        model: str
            name of the pump
        price: numeric
            The price of the pump
        power_rating: numeric
            Power rating of the pump (in fact)
        controler: str
            Name of controller
        diameter_output: numeric
            output diameter

        data extracted from datasheet :
            (voltage, lpm, tdh, current, watts, efficiency ).


    """
    _ids = count(1)

    def __init__(self, path='',
                 lpm=None, tdh=None, current=None,
                 motor_electrical_architecture=None,
                 pump_category=None, model=None,
                 price=None, power_rating=None,
                 controler=None, diameter_output=None):
        # TODO: improve dataset readibility by putting it as DataFrame

        if None not in (lpm, tdh, current):
            # use input data to create pump object
            self.lpm = lpm
            self.tdh = tdh
            self.current = current
            self.voltage = list(self.lpm.keys())
            self.watts = get_watts_from_current(current)
        else:
            # retrieve pump data from txt datasheet given by path
            try:
                self.voltage, self.lpm, self.tdh, self.current, \
                    self.watts = get_data_pump(path)
            except IOError:
                print('The mentionned path does not exist, please select'
                      'another in the pop-up window.')
                tk.Tk().withdraw()
                filepath = tkfile.askopenfilename()
                self.path = filepath
                self.voltage, self.lpm, self.tdh, self.current, \
                    self.watts = get_data_pump(path)

        self.motor_electrical_architecture = motor_electrical_architecture
        self.pump_category = pump_category
        self.model = model
        self.price = price
        self.power_rating = power_rating
        self.controler = controler
        self.diameter_output = diameter_output

        self.data_completeness = specs_completeness(
            self.voltage, self.lpm, self.tdh, self.current,
            self.motor_electrical_architecture)

        # data in the form of DataFrame
        vol = []
        head = []
        cur = []
        flow = []
        power = []
        for V in self.current:
            for i, Idata in enumerate(self.current[V]):
                vol.append(V)
                head.append(self.tdh[V][i])
                cur.append(Idata)
                flow.append(self.lpm[V][i])
                power.append(V*Idata)
        self.specs_df = pd.DataFrame({'voltage': vol,
                                      'tdh': head,
                                      'current': cur,
                                      'flow': flow,
                                      'power': power})

        # coeffs not really directly used previously, changed for new ones
        coeffs = _curves_coeffs_old(self.lpm, self.tdh, self.watts)
        self.coeff_pow_with_lpm = coeffs['pow_with_lpm']
        self.coeff_pow_with_tdh = coeffs['pow_with_tdh']
        self.coeff_tdh_with_lpm = coeffs['tdh_with_lpm']

        # new ones
        if self.data_completeness['voltage_number'] >4:
            self.coeffs = _curves_coeffs_Kou98(
                    self.specs_df, self.data_completeness['data_number'],
                    self.data_completeness['voltage_number'])
        else:
            self.coeffs = None
#        self.coeff_pow_with_lpm = coeffs[1]
#        self.coeff_pow_with_tdh = coeffs['pow_with_tdh']
#        self.coeff_tdh_with_lpm = coeffs['tdh_with_lpm']

        self.id = next(self._ids)

    def __repr__(self):
        affich = "model :" + str(self.model) + \
                 "\npump_category :" + str(self.pump_category) + \
                 "\nprice :" + str(self.price) + \
                 "\npower rating (HP) :" + str(self.power_rating) + \
                 "\ncontroler :" + str(self.controler) + \
                 "\noutput diameter (inches) :" + str(self.diameter_output)
        return affich

    def starting_characteristics(self, tdh, motor_electrical_architecture):
        """
        ------------------------- TO CHECK !! -------------------------
        --------- consistant with results from functVforIH ??? --------

        Returns the required starting voltage, power and current
        for a specified tdh.

        motor_electrical_architecture: str,
            'permanent_magnet', 'series_excited', 'shunt_excited',
            'separately_excited'. Names selected according to [1]

        returns :
            {'V':vmin,'P':pmin,'I':imin}
            vmin is :
                None: if tdh out of the range of the pump
                float: value of minimum starting voltage

        References
        ----------
        [1] Singer and Appelbaum,"Starting characteristics of direct
        current motors powered by solar cells", IEEE transactions on
        energy conversion, vol8, 1993

        """
        raise NotImplementedError

#        if self.coeff_tdh is None:
#            self._curves_coeffs_old()
#
#        tdhmax = {}
#        powmin = {}
#        for V in self.voltage:
#            tdhmax[V] = self.coeff_tdh[V][0]  # y-intercept of tdh vs lpm
#            powmin[V] = self.coeff_pow[V][0]
#
#        # interpolation:
#        # of Vmin vs tdh
#        newf_t = spint.interp1d(list(tdhmax.values()), list(tdhmax.keys()),
#                                kind='cubic')
#        # of power vs V
#        newf_p = spint.interp1d(list(powmin.keys()), list(powmin.values()),
#                                kind='cubic')
#
#        if tdh < min(tdhmax.values()):
#            print('The resqueted tdh is out of the range for the pump,'
#                  'it is below the minimum tdh.')
#            vmin = 'below'
#            pmin = None
#            imin = None
#        elif tdh > max(tdhmax.values()):
#            print('The resqueted tdh is out of the range for the pump,'
#                  'it is above the maximum tdh delivered by the pump.')
#            vmin = 'above'
#            pmin = None
#            imin = None
#        else:
#            vmin = newf_t(tdh)
#            pmin = newf_p(vmin)
#            imin = pmin/vmin
#        return {'V': vmin, 'P': pmin, 'I': imin}

    def plot_tdh_Q(self):
        """Print the graph of tdh(in m) vs Q(in lpm)
        """

        coeffs = _curves_coeffs_old(self.lpm, self.tdh, self.watts)

        tdh_x = {}
        # greatest value of lpm encountered in data
        lpm_max = max(self.lpm[max(self.voltage)])
        lpm_x = np.arange(0, lpm_max, step=lpm_max/10)  # vector of lpm

        for V in self.voltage:

            def tdh_funct(x):
                # function tdh
                return (coeffs['tdh'][V][0]
                        + coeffs['tdh'][V][1]*x
                        + coeffs['tdh'][V][2]*x**2
                        + coeffs['tdh'][V][3]*x**3
                        + coeffs['tdh'][V][4]*x**4)

            # vectors of tdh and efficiency with lpm - ready to be printed
            tdh_x[V] = tdh_funct(lpm_x)

        fig = plt.figure(facecolor='White')
        # add space in height between the subplots:
        fig.subplots_adjust(hspace=0.5)
        ax1 = plt.subplot(2, 1, 1)

        for i, V in enumerate(self.voltage):  # for each voltage available :
            # get the next color to have the same color by voltage:
            col = next(ax1._get_lines.prop_cycler)['color']
            plot(lpm_x, tdh_x[V], linestyle='--', linewidth=1.5, color=col,
                 label=str(V)+'VDC extrapolated')
            plot(self.lpm[V], self.tdh[V], linestyle='-', linewidth=2,
                 color=col, label=str(V)+'VDC from specs')
        ax1.set_title(str(self.model) +
                      ' Flow rate curves Vs. Head')
        ax1.set_xlabel('lpm')
        ax1.set_ylabel('Head (m)')
        ax1.set_ylim(0, max(tdh_x[max(self.voltage)])*1.1)
        ax1.legend(loc='best')
        ax1.grid(True)

        ax2 = plt.subplot(2, 1, 2)
        for V in self.voltage:
            plot(self.lpm[V], self.watts[V], linewidth=2,
                 label=str(V) + ' VDC')
        ax2.set_xlabel('lpm')
        ax2.set_ylabel('watts')
        ax2.set_title(str(self.model) +
                      'Flow rate Vs. electrical power')
        ax2.grid(True)
        ax2.legend(loc='best')

        plt.show()

    def functVforIH(self):
        """
        Returns:
        --------
        * Tuple containing :
            - the function giving V according to I and H static for the pump :
                V = f1(I, H)
            - the standard deviation on V between real data points and data
                computed with this function
            - a dict containing the domains of I and H
                (Now the control is done inside the function by raising errors)
        """
        if (len(self.current[self.voltage[0]]) >= 4
                and len(self.tdh[self.voltage[0]]) >= 4):
            funct_mod = function_models.polynomial_multivar_3_3_1
        else:
            raise errors.InsufficientDataError('Information on current '
                                               'for pump lacks.')

        # gathering of data
        vol = []
        tdh = []
        cur = []
        for V in self.voltage:
            for i, Idata in enumerate(self.current[V]):
                vol.append(V)
                tdh.append(self.tdh[V][i])
                cur.append(Idata)

        dataxy = [np.array(cur), np.array(tdh)]
        dataz = np.array(np.array(vol))
        # computing of linear regression
        param, covmat = opt.curve_fit(funct_mod, dataxy, dataz)
        # comparison between linear reg and actual data
        datacheck = funct_mod(dataxy, *param)
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        # domain computing
        dom = _domain_I_H(self.current, self.tdh, self.data_completeness)
        intervals = {'I': dom[0],
                     'H': dom[1]}

        def functV(i, H, error_raising=True):
            """Function giving voltage V according to current I and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not intervals['I'](H)[0] <= i <= intervals['I'](H)[1]:
                    raise errors.CurrentError(
                            'I (={0}) is out of bounds. For this specific '
                            'head H (={1}), current I should be in the '
                            'interval {2}'.format(i, H, intervals['I'](H)))
                if not intervals['H'](I)[0] <= H <= intervals['H'](I)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'current I (={1}), H should be in the interval {2}'
                            .format(H, i, intervals['H'](I)))
            return funct_mod([i, H], *param)

        return functV, ectyp, intervals

    def functIforVH(self):
        """Returns a tuple containing :
            - the function giving I according to V and H static for the pump :
                I = f1(V, H)
            - the standard deviation on I between real data points and data
                computed with this function
            - a dict containing the domains of V and H
                (Now the control is done inside the function by raising errors)
        """

        if self.data_completeness['voltage'] >= 4:
            funct_mod = function_models.polynomial_multivar_3_3_1
        elif self.data_completeness['voltage'] == 3:
            funct_mod = function_models.polynomial_multivar_2_2_0
        elif self.data_completeness['voltage'] == 2:
            funct_mod = function_models.polynomial_multivar_1_1_0
        elif self.data_completeness['voltage'] == 1:
            funct_mod = function_models.polynomial_multivar_0_1_0
        else:
            raise errors.InsufficientDataError('Information on accepted '
                                               'voltage for pump lacks.')

        # loading of data
        vol = []  # voltage
        tdh = []  # total dynamic head
        cur = []  # current
        for V in self.voltage:
            for i, I in enumerate(self.current[V]):
                vol.append(V)
                cur.append(I)
                tdh.append(self.tdh[V][i])

        dataxy = [np.array(vol), np.array(tdh)]
        dataz = np.array(np.array(cur))

        # curve-fitting of linear regression
        param, covmat = opt.curve_fit(funct_mod, dataxy, dataz)
        datacheck = funct_mod(dataxy, *param)
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        # domain of V and tdh and gathering in one single variable
        dom = _domain_V_H(self.tdh, self.data_completeness)
        intervals = {'V': dom[0],
                     'H': dom[1]}

        def functI(V, H, error_raising=True):
            """Function giving voltage V according to current I and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not intervals['V'](H)[0] <= V <= intervals['V'](H)[1]:
                    raise errors.VoltageError(
                            'V (={0}) is out of bounds. For this specific '
                            'head H (={1}), V should be in the interval {2}'
                            .format(V, H, intervals['V'](H)))
                if not intervals['H'](V)[0] <= H <= intervals['H'](V)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'voltage V (={1}), H should be in the interval {2}'
                            .format(H, V, intervals['H'](V)))
            return funct_mod([V, H], *param)

        return functI, ectyp, intervals

    def functIforVH_Kou(self):
        """Returns a tuple containing :
            - the function giving I according to V and H static for the pump :
                I = f1(V, H)
            - a dict containing the domains of V and H
                (Now the control is done inside the function by raising errors)
        """

        if self.coeffs is not None:
            coeffs = self.coeffs['coeffs_f1']
            if self.coeffs['model'] == 'kou':
                funct_mod = function_models.polynomial_multivar_3_3_4

        # domain of V and tdh and gathering in one single variable
        dom = _domain_V_H(self.tdh, self.data_completeness)
        intervals = {'V': dom[0],
                     'H': dom[1]}

        def functI(V, H, error_raising=True):
            """Function giving voltage V according to current I and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not intervals['V'](H)[0] <= V <= intervals['V'](H)[1]:
                    raise errors.VoltageError(
                            'V (={0}) is out of bounds. For this specific '
                            'head H (={1}), V should be in the interval {2}'
                            .format(V, H, intervals['V'](H)))
                if not intervals['H'](V)[0] <= H <= intervals['H'](V)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'voltage V (={1}), H should be in the interval {2}'
                            .format(H, V, intervals['H'](V)))
            return funct_mod([V, H], *coeffs)

        return functI, intervals

    def functQforVH(self):
        """Returns a tuple containing :
            -the function giving Q according to V and H static for the pump :
                Q = f2(V,H)
            -the standard deviation on Q between real data points and data
                computed with this function
        """
        if self.data_completeness['voltage'] >= 4:
            funct_mod = function_models.polynomial_multivar_3_3_1
        elif self.data_completeness['voltage'] == 3:
            funct_mod = function_models.polynomial_multivar_2_2_0
        elif self.data_completeness['voltage'] == 2:
            funct_mod = function_models.polynomial_multivar_1_1_0
        elif self.data_completeness['voltage'] == 1:
            funct_mod = function_models.polynomial_multivar_0_1_0
        else:
            raise errors.InsufficientDataError('Information on accepted '
                                               'voltage for pump lacks.')
        # gathering of data needed
        vol = []
        tdh = []
        lpm = []
        for V in self.voltage:
            for i, Q in enumerate(self.lpm[V]):
                vol.append(V)
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        dataxy = [np.array(vol), np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        param, covmat = opt.curve_fit(funct_mod, dataxy, dataz)

        datacheck = funct_mod(dataxy, *param)
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        def functQ(V, H):
            if not min(vol) <= V <= max(vol):
                raise errors.VoltageError('V (={0}) is out of bounds. It '
                                          'should be in the interval {1}'
                                          .format(V, [min(vol), max(vol)]))
            if not min(tdh) <= H <= max(tdh):
                raise errors.HeadError('H (={0}) is out of bounds. It should'
                                       ' be in the interval {1}'
                                       .format(H, [min(tdh), max(tdh)]))
            return funct_mod([V, H], *param)

        return functQ, ectyp

    def functQforPH(self):
        """Returns a tuple containing :
            -the function giving Q according to P and H static for the pump :
                Q = f2(P,H)
            -the standard deviation on Q between real data points and data
                computed with this function
        """

        if (len(self.watts[self.voltage[0]]) >= 4
                and len(self.tdh[self.voltage[0]]) >= 4):
            funct_mod = function_models.polynomial_multivar_3_3_1
        else:
            raise errors.InsufficientDataError('Information on power '
                                               'for pump lacks.')
        # gathering of data needed
        power = []
        tdh = []
        lpm = []
        for V in self.voltage:
            for i, Q in enumerate(self.lpm[V]):
                power.append(self.watts[V][i])
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        dataxy = [np.array(power), np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        param, covmat = opt.curve_fit(funct_mod, dataxy, dataz)

        datacheck = funct_mod(dataxy, *param)
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        def functQ(P, H):
            if not min(power) <= P <= max(power):
                raise errors.PowerError('P (={0}) is out of bounds. It '
                                        'should be in the interval {1}'
                                        .format(P, [min(power), max(power)]))
            if not min(tdh) <= H <= max(tdh):
                raise errors.HeadError('H (={0}) is out of bounds. It should'
                                       ' be in the interval {1}'
                                       .format(H, [min(tdh), max(tdh)]))
            return funct_mod([P, H], *param)

        return functQ, ectyp

    def functQforPH_Kou(self):
        """Returns a tuple containing :
            -the function giving Q according to P and H static for the pump :
                Q = f2(P,H)
            -the standard deviation on Q between real data points and data
                computed with this function
        """

        if self.coeffs is not None:
            coeffs = self.coeffs['coeffs_f2']
            if self.coeffs['model'] == 'kou':
                funct_mod = function_models.polynomial_multivar_3_3_4

        # domain of V and tdh and gathering in one single variable
        dom = _domain_P_H(self.tdh, self.data_completeness)
        intervals = {'P': dom[0],
                     'H': dom[1]}

        def functQ(P, H):
            if not intervals['P'](H)[0] <= P <= intervals['P'](H)[1]:
                raise errors.PowerError('P (={0}) is out of bounds. It '
                                        'should be in the interval {1}'
                                        .format(P, intervals['P'](H)))
            if not intervals['H'](P)[0] <= H <= intervals['H'](P)[1]:
                raise errors.HeadError('H (={0}) is out of bounds. It should'
                                       ' be in the interval {1}'
                                       .format(H, intervals['H'](P)))
            return funct_mod([P, H], *coeffs)

        return functQ, intervals

    def IVcurvedata(self, head, nbpoint=40):
        """Function returning the data needed for plotting the IV curve at
        a given head.

        Return
        ------
            -dict with keys I and V, and the corresponding list of values
        """

        fctV, sigma, inter = self.functVforIH()
        if head > max(self.tdh):
            print('h_tot is not in the range of the pump')
            return {'I': 0, 'V': 0}

        Itab = np.linspace(min(inter['I'](head)), max(inter['I'](head)),
                           nbpoint)
        Vtab = np.zeros(nbpoint)

        for i, I in enumerate(Itab):
            try:
                Vtab[i] = fctV(I, head)
            except errors.HeadError:
                Vtab[i] = -1

        return {'I': Itab, 'V': Vtab}


def get_watts_from_current(current_dict):
    """Compute electric power.

    Parameter
    ---------
    current_dict: dict
        Dictionary containing list of currents (values) drawn by the pump
        according to the voltages (keys).

    Return
    ------
    * dictionary with voltage as keys and with power drawn by the pump
        as values.

    """
    power_dict = {}
    for voltage in current_dict:
        power_list = list(np.array(current_dict[voltage])*voltage)
        power_dict[voltage] = power_list

    return power_dict


def get_data_pump(path):
    """
    This function is used to load the pump data from the .txt file
    designated by the path. This .txt files has the
    characteristics of the datasheets. The data is returned in the
    form of 6 tables (list containing lists in real):
    voltage, lpm, tdh, current, watts, efficiency.

    Parameters:
    -----------
    path: str
        path to the file of the pump data

    Returns:
    --------
    tuple
        tuple containing list

    """
    # Import data
    data = np.loadtxt(path, dtype={'names': ('voltage', 'tdh', 'current',
                                             'lpm', 'watts', 'efficiency'),
                      'formats': (float, float, float, float, float, float)},
                      skiprows=1)

    # sorting of data
    volt = np.zeros(data.size)  # array filled with 0
    for i in range(0, data.size):
        volt[i] = data[i][0]
    # create dict with voltage as keys and with number of occurence as values
    counter = collections.Counter(volt)
    keys_sorted = sorted(list(counter.keys()))  # voltages in increasing order

    # Creation and filling of data lists
    voltage = keys_sorted
    # main dict, containing sub-list per voltage
    lpm = {}
    tdh = {}
    current = {}
    watts = {}

    k = 0
    for V in voltage:
        tdh_temp = []
        current_temp = []
        lpm_temp = []
        watts_temp = []
        for j in range(0, counter[V]):
            tdh_temp.append(data[k][1])
            current_temp.append(data[k][2])
            lpm_temp.append(data[k][3])
            watts_temp.append(data[k][4])
            k = k+1
        tdh[V] = tdh_temp
        current[V] = current_temp
        lpm[V] = lpm_temp
        watts[V] = watts_temp

    return voltage, lpm, tdh, current, watts


def specs_completeness(voltage, lpm, tdh, current,
                       motor_electrical_architecture):
    """
    Evaluates the completeness of the data of a pump.

    Returns
    -------
    dictionary with following keys:
        * voltage_number: float
            number of voltage for which data are given
        * data_number: float
            number of points for which lpm, current, voltage and head are
            given
        * lpm_min: float
            Ratio between min flow_rate given and maximum.
            Should be ideally 0.
        * head_min:float
            Ratio between min head given and maximum.
            Should be ideally 0.
        * elec_archi: boolean
            A valid electrical architecture for the motor is given
    """

    valid_elec_archi = (motor_electrical_architecture in (
            'permanent_magnet', 'series_excited', 'shunt_excited',
            'separately_excited'))

    volt_nb = len(voltage)

    # computing of mean lpm completeness
    lpm_ratio = []
    for v in lpm:
        lpm_ratio.append(min(lpm[v])/max(lpm[v]))
    mean_lpm_ratio = np.mean(lpm_ratio)

    head_ratio = min_in_values(tdh)/max_in_values(tdh)

    data_number = 0
    for volt in lpm:
        for i in lpm[volt]:
            data_number += 1
    # TODO: should add some lines to check that current and tdh have same shape
    # than lpm


    return {'voltage_number': volt_nb,
            'lpm_min': mean_lpm_ratio,
            'head_min': head_ratio,
            'elec_archi': valid_elec_archi,
            'data_number': data_number}


def max_in_values(data):
    """Finds the maximum in the values of a dict"""
    data_flat = []
    for key in data.keys():
        for elt in data[key]:
            data_flat.append(elt)
    return max(data_flat)


def min_in_values(data):
    """Finds the minimum in the values of a dict"""
    data_flat = []
    for key in data.keys():
        for elt in data[key]:
            data_flat.append(elt)
    return min(data_flat)


def _curves_coeffs(lpm, tdh, watts, method):
    """Compute curve-fitting coefficient with different methods : Kou,
    Hadj Arab or theoretical.

    """
    return None


def _curves_coeffs_Kou98(specs_df, data_number, voltage_number):
    """Compute curve-fitting coefficient with method of Kou [1]

    Parameters
    ----------
    specs_df: pd.DataFrame
        DataFrame with specs.

    Reference
    ---------
    [1] Kou Q, Klein S.A. & Beckman W.A., "A method for estimating the
    long-term performance of direct-coupled PV pumping systems", 1998,
    Solar Energy

    """

    if data_number >= 10 and voltage_number > 3:
        funct_mod = function_models.polynomial_multivar_3_3_4
    else:
        raise errors.InsufficientDataError('Lack of information on lpm, '
                                           'current or tdh for pump.')
    # f1: I(V, H)
    datax_df = specs_df['voltage']
    datay_df = specs_df['tdh']
    dataz_df = specs_df['current']
    dataxy_ar = [np.array(datax_df),
                 np.array(datay_df)]
    dataz_ar = np.array(dataz_df)

    param_f1, covmat_f1 = opt.curve_fit(
            funct_mod, dataxy_ar, dataz_ar
#            , bounds=([0, 0, -np.inf, -np.inf, -np.inf],
#                    [np.inf, np.inf, 0, 0, 0])
            )
    # computing of RMSE and of normalized RMSE for f1
    datacheck = funct_mod(dataxy_ar, *param_f1)
    rmse_f1 = np.sqrt(sum((dataz_ar-datacheck)**2)/len(dataz_ar))
    nrmse_f1 = rmse_f1/np.mean(datacheck)

    # f2: Q(P, H)
    datax_df = specs_df['power']
    dataz_df = specs_df['flow']
    dataxy_ar = [np.array(datax_df),
                 np.array(datay_df)]
    dataz_ar = np.array(dataz_df)

    param_f2, covmat_f2 = opt.curve_fit(funct_mod, dataxy_ar,
                                        dataz_ar)
    # computing of RMSE and of normalized RMSE for f2
    datacheck = funct_mod(dataxy_ar, *param_f2)
    rmse_f2 = np.sqrt(sum((dataz_ar-datacheck)**2)/len(dataz_ar))
    nrmse_f2 = rmse_f2/np.mean(datacheck)

    return {'coeffs_f1': param_f1,
            'rmse_f1': rmse_f1,
            'nrmse_f1': nrmse_f1,
            'coeffs_f2': param_f2,
            'rmse_f2': rmse_f2,
            'nrmse_f2': nrmse_f2,
            'model': 'kou'}


def _curves_coeffs_old(lpm, tdh, watts):
    """
    Compute curve-fitting coefficient from data for :
        - tdh vs lpm
        - power vs lpm

    returns a dict of sub-dict :
        -the first dict contains the 2 curves as keys : 'tdh','pow'
        resp. for total dynamic head and power
            -the sub-dicts contain the available voltage as keys, typically
            '60','75','90','105','120'
    These same 3 dictionnary are saved as attributes in the pump object,
    under the name 'self.coeff_tdh', 'self.coeff_pow'
    """
    func_model = function_models.polynomial_4

    coeff_tdh_with_lpm = {}  # coeff from curve-fitting of tdh vs lpm
    coeff_pow_with_lpm = {}  # coeff from curve-fitting of power vs lpm
    coeff_pow_with_tdh = {}  # coeff from curve-fitting of power vs tdh

    for V in lpm:
        # curve-fit of tdh vs lpm
        coeffs_tdh, matcov = opt.curve_fit(
            func_model, lpm[V], tdh[V],
            p0=[10, -1, -1, 0, 0],
            bounds=([0, -np.inf, -np.inf, -np.inf, -np.inf],
                    [np.inf, 0, 0, 0, 0]))
        coeff_tdh_with_lpm[V] = coeffs_tdh

        # curve-fit of power vs lpm
        coeffs_P, matcov = opt.curve_fit(func_model, lpm[V], watts[V])
        coeff_pow_with_lpm[V] = coeffs_P

        # curve-fit of power vs lpm
        coeffs_P, matcov = opt.curve_fit(func_model, tdh[V], watts[V])
        coeff_pow_with_tdh[V] = coeffs_P

    return {'tdh_with_lpm': coeff_tdh_with_lpm,
            'pow_with_lpm': coeff_pow_with_lpm,
            'pow_with_tdh': coeff_pow_with_tdh}


def _domain_I_H(data_cur, data_tdh, data_completeness):
    """
    Function giving the domain of cur depending on tdh, and vice versa.

    """

    funct_mod = function_models.polynomial_2

    # domains of I and tdh depending on each other
    x_tips = []
    y_tips = []
    for v in data_tdh:
        x_tips.append(min(data_cur[v]))
        y_tips.append(max(data_tdh[v]))

    if data_completeness['voltage'] > 2 and data_completeness['lpm_min'] == 0:
        # case working fine for SunPumps - not sure about complete data from
        # other manufacturer
        param_y, pcov_y = opt.curve_fit(funct_mod,
                                        x_tips, y_tips)
        param_x, pcov_x = opt.curve_fit(funct_mod,
                                        y_tips, x_tips)

        def interval_cur(y):
            "Interval on x depending on y"
            return [max(funct_mod(y, *param_x),
                        min_in_values(data_cur)),
                    max_in_values(data_cur)]

        def interval_tdh(x):
            "Interval on y depending on x"
            return [0, min(max(funct_mod(x, *param_y),
                               0),
                           max_in_values(data_tdh))]

    else:
        # Would need deeper work to fully understand what are the limits
        # on I and V depending on tdh, and how it affects lpm
        def interval_cur(*args):
            "Interval on current, independent of tdh"
            return [min_in_values(data_cur), max_in_values(data_cur)]

        def interval_tdh(*args):
            "Interval on tdh, independent of current"
            return [min_in_values(data_tdh), max_in_values(data_tdh)]

    return interval_cur, interval_tdh


def _domain_V_H(data_tdh, data_completeness):
    """
    Function giving the range of voltage and head in which the pump will
    work.
    """
    funct_mod = function_models.polynomial_2

    data_v = []
    tdh_tips = []
    for v in data_tdh:
        data_v.append(v)
        tdh_tips.append(max(data_tdh[v]))

    if data_completeness['voltage_number'] > 2 \
            and data_completeness['lpm_min'] == 0:
        # case working fine for SunPumps - not sure about complete data from
        # other manufacturer
        param_tdh, pcov_tdh = opt.curve_fit(funct_mod,
                                            data_v, tdh_tips)
        param_v, pcov_v = opt.curve_fit(funct_mod,
                                        tdh_tips, data_v)

        def interval_vol(tdh):
            "Interval on v depending on tdh"
            return [max(funct_mod(tdh, *param_v), min(data_v)),
                    max(data_v)]

        def interval_tdh(v):
            "Interval on tdh depending on v"
            return [0, min(max(funct_mod(v, *param_tdh), 0),
                           max(data_tdh))]

    else:
        # Would need deeper work to fully understand what are the limits
        # on I and V depending on tdh, and how it affects lpm
        def interval_vol(*args):
            "Interval on vol, independent of tdh"
            return [min(data_v), max(data_v)]

        def interval_tdh(*args):
            "Interval on tdh, independent of vol"
            return [min_in_values(data_tdh), max_in_values(data_tdh)]

    return interval_vol, interval_tdh


def _domain_P_H(specs_df, data_completeness):
    """
    Function giving the range of power and head in which the pump will
    work.
    """
    funct_mod = function_models.polynomial_1

    if data_completeness['voltage_number'] >= 2 \
            and data_completeness['lpm_min'] == 0:
        # case working fine for SunPumps - not sure about complete data from
        # other manufacturer
        df_flow_null = specs_df[specs_df.flow == 0]

        datapower_df = df_flow_null['power']
        datatdh_df = df_flow_null['tdh']

        datapower_ar = np.array(datapower_df)
        datatdh_ar = np.array(datatdh_df)

        param_tdh, pcov_tdh = opt.curve_fit(funct_mod,
                                            datapower_ar, datatdh_ar)
        param_pow, pcov_pow = opt.curve_fit(funct_mod,
                                            datatdh_ar, datapower_ar)

        def interval_power(tdh):
            "Interval on power depending on tdh"
            return [max(funct_mod(tdh, *param_pow), min(datapower_ar)),
                    np.inf]

        def interval_tdh(power):
            "Interval on tdh depending on v"
            return [0, min(max(funct_mod(power, *param_tdh), 0),
                           max(datatdh_ar))]

    elif data_completeness['voltage_number'] >= 2:
        tdhmax_df = specs_df[specs_df.tdh == max(specs_df.tdh)]
        power_min_tdhmax = min(tdhmax_df.power)
        tdhmin_df = specs_df[specs_df.tdh == min(specs_df.tdh)]
        power_min_tdhmin = min(tdhmin_df.power)

        datapower_ar = np.array([power_min_tdhmin, power_min_tdhmax])
        datatdh_ar = np.array(
            [float(specs_df[specs_df.power == power_min_tdhmin].tdh),
             float(specs_df[specs_df.power == power_min_tdhmax].tdh)])
        param_tdh, pcov_tdh = opt.curve_fit(funct_mod,
                                            datapower_ar, datatdh_ar)
        param_pow, pcov_pow = opt.curve_fit(funct_mod,
                                            datatdh_ar, datapower_ar)

        def interval_power(tdh):
            "Interval on power depending on tdh"
            return [max(funct_mod(tdh, *param_pow), min(datapower_ar)),
                    np.inf]

        def interval_tdh(power):
            "Interval on tdh depending on v"
            return [0, min(max(funct_mod(power, *param_tdh), 0),
                           max(datatdh_ar))]

    else:
        # Would need deeper work to fully understand what are the limits
        # on I and V depending on tdh, and how it affects lpm

        datax_ar = np.array(specs_df.power)
        datay_ar = np.array(specs_df.tdh)

        def interval_power(*args):
            "Interval on power, independent of tdh"
            return [min(datax_ar), max(datax_ar)]

        def interval_tdh(*args):
            "Interval on tdh, independent of power"
            return [min(datay_ar), max(datay_ar)]

    return interval_power, interval_tdh


if __name__ == "__main__":
    # %% pump creation
    pump1 = Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                 model='SCB_10')

    pump2 = Pump(lpm={12: [212, 204, 197, 189, 186, 178, 174, 166, 163, 155,
                          136],
                     24: [443, 432, 413, 401, 390, 382, 375, 371, 352, 345,
                          310]},
                 tdh={12: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
                           54.9, 61.0, 70.1],
                      24: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
                           54.9, 61.0, 70.1]},
                 current={12: [1.2, 1.5, 1.8, 2.0, 2.1, 2.4, 2.7, 3.0, 3.3,
                               3.4, 3.9],
                          24: [1.5, 1.7, 2.1, 2.4, 2.6, 2.8, 3.1, 3.3, 3.6,
                               3.8, 4.1]
                          }, model='Shurflo_9325',
                 motor_electrical_architecture='permanent_magnet')

#    pump2 = Pump(lpm={12: [212, 204, 197, 189, 186, 178, 174, 166, 163, 155],
#                      24: [443, 432, 413, 401, 390, 382, 375, 371, 352, 345,
#                           310]},
#                 tdh={12: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
#                           54.9, 61.0],
#                      24: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
#                           54.9, 61.0, 70.1]},
#                 current={12: [1.2, 1.5, 1.8, 2.0, 2.1, 2.4, 2.7, 3.0, 3.3,
#                               3.4],
#                          24: [1.5, 1.7, 2.1, 2.4, 2.6, 2.8, 3.1, 3.3, 3.6,
#                               3.8, 4.1],
#                          }, model='Shurflo_9325')

#    pump1.plot_tdh_Q()
#    pump2.plot_tdh_Q()

    print(pump2.coeffs)
    domains = _domain_P_H(pump2.specs_df, pump2.data_completeness)
    res_f1 = pump2.functIforVH_Kou()
    print(res_f1)

## %% set-up for following plots
#    vol = []
#    tdh = []
#    cur = []
#    lpm = []
#    power = []
#    for V in pump1.voltage:
#        for i, I in enumerate(pump1.current[V]):
#            vol.append(V)
#            cur.append(I)
#            tdh.append(pump1.tdh[V][i])
#            lpm.append(pump1.lpm[V][i])
#            power.append(pump1.watts[V][i])
#
#
## %% plot of functVforIH
#    f1, stddev, intervals = pump1.functVforIH()
#    vol_check = []
#    for i, I in enumerate(cur):
#        vol_check.append(f1(I, tdh[i], error_raising=False))
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', title='Voltage as a function of'
#                         ' current (A) and static head (m)')
#    ax.scatter(cur, tdh, vol, label='from data')
#    ax.scatter(cur, tdh, vol_check, label='from curve fitting')
#    ax.set_xlabel('current')
#    ax.set_ylabel('head')
#    ax.set_zlabel('voltage V')
#    ax.legend(loc='lower left')
#    plt.show()
#    print('std dev on V:', stddev)
##    print('V for IH=(4,25): {0:.2f}'.format(f1(4, 25)))
#
## %% plot of functIforVH
#    f2, stddev, intervals = pump1.functIforVH()
#    cur_check = []
#    for i, V in enumerate(vol):
#        cur_check.append(f2(V, tdh[i], error_raising=False))
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', title='Current as a function of'
#                         ' voltage (V) and static head (m)')
#    ax.scatter(vol, tdh, cur, label='from data')
#    ax.scatter(vol, tdh, cur_check, label='from curve fitting')
#    ax.set_xlabel('voltage')
#    ax.set_ylabel('head')
#    ax.set_zlabel('current I')
#    ax.legend(loc='lower left')
#    plt.show()
#    print('std dev on I: ', stddev)
##    print('I for VH=(20, 25): {0:.2f}'.format(f2(20, 25)))
#
#
## %% plot of functQforVH
#    f3, stddev = pump1.functQforVH()
#    lpm_check = []
#    for i, v in enumerate(vol):
#        try:
#            Q = f3(v, tdh[i])
#        except (errors.VoltageError, errors.HeadError):
#            Q = 0
#        lpm_check.append(Q)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', title='Q (lpm) as a function of'
#                         ' voltage (V) and static head (m)')
#    ax.scatter(vol, tdh, lpm, label='from data')
#    ax.scatter(vol, tdh, lpm_check, label='from curve fitting')
#    ax.set_xlabel('voltage')
#    ax.set_ylabel('head')
#    ax.set_zlabel('discharge Q')
#    ax.legend(loc='lower left')
#    plt.show()
#    print('std dev on Q calculated from V:', stddev)
#
#
## %% plot of functQforPH
#    f4, stddev = pump1.functQforPH()
#    lpm_check = []
#    for i, po in enumerate(power):
#        try:
#            Q = f4(po, tdh[i])
#        except (errors.PowerError, errors.HeadError):
#            Q = 0
#        lpm_check.append(Q)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', title='Q (lpm) as a function of'
#                         ' power (W) and static head (m)')
#    ax.scatter(power, tdh, lpm, label='from data')
#    ax.scatter(power, tdh, lpm_check, label='from curve fitting')
#    ax.set_xlabel('power')
#    ax.set_ylabel('head')
#    ax.set_zlabel('discharge Q')
#    ax.legend(loc='lower left')
#    plt.show()
#    print('std dev on Q calculated from P: ', stddev)
