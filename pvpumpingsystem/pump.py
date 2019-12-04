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
from sklearn.metrics import r2_score
import warnings

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
                 controler=None, diameter_output=None,
                 modeling_method='arab'):

        self.id = next(self._ids)

        if None not in (lpm, tdh, current):
            # use input data to create pump object
            self.lpm = lpm
            self.tdh = tdh
            self.current = current
            self.voltage = list(self.lpm.keys())
            # TODO: Delete use of dict and put everything as DataFrame
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
        self.modeling_method = modeling_method

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

        self.data_completeness = specs_completeness(
                self.specs_df,
                self.motor_electrical_architecture)

        # new ones
        if self.modeling_method.lower() == 'kou':
            self.coeffs = _curves_coeffs_Kou98(
                    self.specs_df, self.data_completeness)
        elif self.modeling_method.lower() == 'arab':
            self.coeffs = _curves_coeffs_Arab06(
                    self.specs_df, self.data_completeness)
        elif self.modeling_method.lower() == 'hamidat':
            self.coeffs = _curves_coeffs_Hamidat08(
                    self.specs_df, self.data_completeness)
        else:
            self.coeffs = None
#        self.coeff_pow_with_lpm = coeffs[1]
#        self.coeff_pow_with_tdh = coeffs['pow_with_tdh']
#        self.coeff_tdh_with_lpm = coeffs['tdh_with_lpm']

        # coeffs not really directly used previously, changed for new ones
#        coeffs = _curves_coeffs_Gualteros17(self.lpm, self.tdh, self.watts)
#        self.coeff_pow_with_lpm = coeffs['pow_with_lpm']
#        self.coeff_pow_with_tdh = coeffs['pow_with_tdh']
#        self.coeff_tdh_with_lpm = coeffs['tdh_with_lpm']

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
#            self._curves_coeffs_Gualteros17()
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

        coeffs = _curves_coeffs_Gualteros17(self.lpm, self.tdh, self.watts)

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

    def functVforIH_old(self):
        """
        Returns:
        --------
        * Tuple containing :
            - the function giving V according to I and H static for the pump :
                V = f1(I, H)
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

        # domain computing
        dom = _domain_I_H(self.specs_df, self.data_completeness)
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
                if not intervals['H'](i)[0] <= H <= intervals['H'](i)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'current I (={1}), H should be in the interval {2}'
                            .format(H, i, intervals['H'](i)))
            return funct_mod([i, H], *param)

        return functV, intervals

    def functVforIH(self):
        """
        Function whose goal is to inverse functIforVH to find I from V and H
        analytically.
        """
        raise NotImplementedError

    def functIforVH(self):
        """
        Function computing the IV characteristics of the pump
        depending on head H.

        Returns
        -------
        * a tuple containing :
            - the function giving I according to voltage V and head H
            for the pump : I = f1(V, H)
            - the domains of validity for V and H. Can be functions, so as the
            range of one depends on the other, or fixed ranges.
        """

        if self.modeling_method == 'kou':
            return self.functIforVH_Kou()
        if self.modeling_method == 'arab':
            return self.functIforVH_Arab()
        if self.modeling_method == 'theoretical':
            return self.functIforVH_theoretical()
        if self.modeling_method == 'hamidat':
            raise NotImplementedError(
                "Hamidat method does not provide model for functIforVH, "
                "it is made to model only mppt coupled systems, which "
                "don't need this function.")
        else:
            raise NotImplementedError(
                "The function functIforVH corresponding to the requested "
                "modeling method is not available yet, need to "
                "implemented another valid method.")

    def functIforVH_Arab(self):
        """
        Function using [1] for modeling I vs V of pump.

        References
        ---------
        [1] Hadj Arab A., Benghanem M. & Chenlo F.,
        "Motor-pump system modelization", 2006, Renewable Energy
        [2] Djoudi Gherbi, Hadj Arab A., Salhi H., "Improvement and validation
        of PV motor-pump model for PV pumping system performance analysis",
        2017
        """

        coeffs_a = self.coeffs['a']
        coeffs_b = self.coeffs['b']
        funct_mod = function_models.compound_polynomial_1_3

        # domain of V and tdh and gathering in one single variable
        dom = _domain_V_H(self.specs_df, self.data_completeness)
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
            return funct_mod([V, H], *coeffs_a, *coeffs_b)

        return functI, intervals

    def functIforVH_Kou(self):
        """
        Function using [1] for modeling I vs V of pump.

        Reference
        ---------
        [1] Kou Q, Klein S.A. & Beckman W.A., "A method for estimating the
        long-term performance of direct-coupled PV pumping systems", 1998,
        Solar Energy
        """

        coeffs = self.coeffs['coeffs_f1']
        funct_mod = function_models.polynomial_multivar_3_3_4

        # domain of V and tdh and gathering in one single variable
        dom = _domain_V_H(self.specs_df, self.data_completeness)
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

# TODO: change this function in functIforVH for standardizing
    def functVforIH_theoretical(self):
        """
        Function using electrical architecture for modeling V vs I of pump.

        Reference
        ---------
        [1]
        """

        coeffs = self.coeffs['coeffs_f1']

        def funct_mod(input_values, R_a, beta_0, beta_1, beta_2):
            """Returns the equation v(i, h).
            """
            i, h = input_values
            funct_mod_beta = function_models.polynomial_2
            beta = funct_mod_beta(h, beta_0, beta_1, beta_2)
            return R_a*i + beta*np.sqrt(i)

        # domain of V and tdh and gathering in one single variable
        dom = _domain_I_H(self.specs_df, self.data_completeness)
        intervals = {'I': dom[0],
                     'H': dom[1]}

        def functV(I, H, error_raising=True):
            """Function giving current I according to voltage V and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not intervals['I'](H)[0] <= I <= intervals['I'](H)[1]:
                    raise errors.CurrentError(
                            'I (={0}) is out of bounds. For this specific '
                            'head H (={1}), I should be in the interval {2}'
                            .format(I, H, intervals['V'](H)))
                if not intervals['H'](I)[0] <= H <= intervals['H'](I)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'current I (={1}), H should be in the interval {2}'
                            .format(H, I, intervals['H'](I)))
            return funct_mod([I, H], *coeffs)

        return functV, intervals

#    def functQforVH(self):
#        """Returns a tuple containing :
#            -the function giving Q according to V and H static for the pump :
#                Q = f2(V,H)
#            -the standard deviation on Q between real data points and data
#                computed with this function
#        """
#        if self.data_completeness['voltage'] >= 4:
#            funct_mod = function_models.polynomial_multivar_3_3_1
#        elif self.data_completeness['voltage'] == 3:
#            funct_mod = function_models.polynomial_multivar_2_2_0
#        elif self.data_completeness['voltage'] == 2:
#            funct_mod = function_models.polynomial_multivar_1_1_0
#        elif self.data_completeness['voltage'] == 1:
#            funct_mod = function_models.polynomial_multivar_0_1_0
#        else:
#            raise errors.InsufficientDataError('Information on accepted '
#                                               'voltage for pump lacks.')
#        # gathering of data needed
#        vol = []
#        tdh = []
#        lpm = []
#        for V in self.voltage:
#            for i, Q in enumerate(self.lpm[V]):
#                vol.append(V)
#                tdh.append(self.tdh[V][i])
#                lpm.append(Q)
#
#        dataxy = [np.array(vol), np.array(tdh)]
#        dataz = np.array(np.array(lpm))
#        # computing of linear regression
#        param, covmat = opt.curve_fit(funct_mod, dataxy, dataz)
#
#        datacheck = funct_mod(dataxy, *param)
#        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#
#        def functQ(V, H):
#            if not min(vol) <= V <= max(vol):
#                raise errors.VoltageError('V (={0}) is out of bounds. It '
#                                          'should be in the interval {1}'
#                                          .format(V, [min(vol), max(vol)]))
#            if not min(tdh) <= H <= max(tdh):
#                raise errors.HeadError('H (={0}) is out of bounds. It should'
#                                       ' be in the interval {1}'
#                                       .format(H, [min(tdh), max(tdh)]))
#            return funct_mod([V, H], *param)
#
#        return functQ, ectyp

    def functQforVH(self):
        """
        Function redirecting to functQforPH
        """

        def functQ(V, H):
            f1, _ = self.functIforVH()
            f2, _ = self.functQforPH()
            cur = f1(V, H)
            return f2(V*cur, H)

        dom = _domain_V_H(self.specs_df, self.data_completeness)
        intervals = {'V': dom[0],
                     'H': dom[1]}

        return functQ, intervals

    def functQforPH(self):
        """
        Function computing the output flow rate of the pump.

        Returns
        -------
        * a tuple containing :
            - the function giving Q according to power P and  head H
            for the pump : Q = f2(P, H)
            - the domains of validity for P and H. Can be functions, so as the
            range of one depends on the other, or fixed ranges.
        """

        if self.modeling_method == 'kou':
            return self.functQforPH_Kou()
        if self.modeling_method == 'arab':
            return self.functQforPH_Arab()
        if self.modeling_method == 'hamidat':
            return self.functQforPH_Hamidat()
        else:
            raise NotImplementedError(
                "The function functQforPH corresponding to the requested "
                "modeling method is not available yet, need to "
                "implemented another valid method.")

    def functQforPH_Hamidat(self):
        """

        """
        coeffs_a = self.coeffs['a']
        coeffs_b = self.coeffs['b']
        coeffs_c = self.coeffs['c']
        coeffs_d = self.coeffs['d']
        funct_mod_P = function_models.compound_polynomial_3_3

        def funct_P(Q, power, head):
            return funct_mod_P([Q, head], *coeffs_a, *coeffs_b, *coeffs_c,
                               *coeffs_d) - power

        dom = _domain_P_H(self.specs_df, self.data_completeness)
        intervals = {'P': dom[0],
                     'H': dom[1]}

        def functQ(P, H):
            if P < intervals['P'](H)[0]:
                Q = 0
                P_unused = P
            elif intervals['P'](H)[0] < P < intervals['P'](H)[1]:
                # Newton-Raphson numeraical method:
                # actually fprime should be given for using Newton-Raphson
                Q = opt.newton(funct_P, 5, args=(P, H))
                P_unused = 0  # power unused for pumping
            elif intervals['P'](H)[1] < P:
                Pmax = intervals['P'](H)[1]
                Q = opt.newton(funct_P, 5, args=(Pmax, H))
                P_unused = P - Pmax
            else:  # for example P is NaN
                Q = np.nan
                P_unused = np.nan
            return {'Q': Q, 'P_unused': P_unused}

        return functQ, intervals

    def functQforPH_Arab(self):
        """
        Function using [1] and [2] for output flow rate modeling.

        References
        ---------
        [1] Hadj Arab A., Benghanem M. & Chenlo F.,
        "Motor-pump system modelization", 2006, Renewable Energy
        [2] Djoudi Gherbi, Hadj Arab A., Salhi H., "Improvement and validation
        of PV motor-pump model for PV pumping system performance analysis",
        2017
        """

        coeffs_c = self.coeffs['c']
        coeffs_d = self.coeffs['d']
        coeffs_e = self.coeffs['e']
        funct_mod = function_models.compound_polynomial_2_3

        # domain of V and tdh and gathering in one single variable
        dom = _domain_P_H(self.specs_df, self.data_completeness)
        intervals = {'P': dom[0],
                     'H': dom[1]}

        def functQ(P, H):
            if P < intervals['P'](H)[0]:
                Q = 0
                P_unused = P
            elif intervals['P'](H)[0] < P < intervals['P'](H)[1]:
                Q = funct_mod([P, H], *coeffs_c, *coeffs_d, *coeffs_e)
                P_unused = 0
            elif intervals['P'](H)[1] < P:
                Pmax = intervals['P'](H)[1]
                Q = funct_mod([Pmax, H], *coeffs_c, *coeffs_d, *coeffs_e)
                P_unused = P - Pmax
            else:  # for example P is NaN
                Q = np.nan
                P_unused = np.nan
            return {'Q': Q, 'P_unused': P_unused}

        return functQ, intervals

    def functQforPH_Kou(self):
        """
        Function using [1] for output flow rate modeling.

        Reference
        ---------
        [1] Kou Q, Klein S.A. & Beckman W.A., "A method for estimating the
        long-term performance of direct-coupled PV pumping systems", 1998,
        Solar Energy
        """

        coeffs = self.coeffs['coeffs_f2']
        funct_mod = function_models.polynomial_multivar_3_3_4

        # domain of V and tdh and gathering in one single variable
        dom = _domain_P_H(self.specs_df, self.data_completeness)
        intervals = {'P': dom[0],
                     'H': dom[1]}

        def functQ(P, H):
            if P < intervals['P'](H)[0]:
                Q = 0
                P_unused = P
            elif intervals['P'](H)[0] < P < intervals['P'](H)[1]:
                Q = funct_mod([P, H], *coeffs)
                P_unused = 0
            elif intervals['P'](H)[1] < P:
                Pmax = intervals['P'](H)[1]
                Q = funct_mod([P, H], *coeffs)
                P_unused = P - Pmax
            else:  # for example P is NaN
                Q = np.nan
                P_unused = np.nan
            return {'Q': Q, 'P_unused': P_unused}

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


def specs_completeness(specs_df,
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

    voltages = specs_df.voltage.drop_duplicates()
    volt_nb = len(voltages)

    # computing of mean lpm completeness
    lpm_ratio = []
    for v in voltages:
        lpm_ratio.append(min(specs_df[specs_df.voltage == v].flow)
                         / max(specs_df[specs_df.voltage == v].flow))
    mean_lpm_ratio = np.mean(lpm_ratio)

    head_ratio = min(specs_df.tdh)/max(specs_df.tdh)

    data_number = 0
    for v in voltages:
        for i in specs_df[specs_df.voltage == v].flow:
            data_number += 1

    return {'voltage_number': volt_nb,
            'lpm_min': mean_lpm_ratio,
            'head_min': head_ratio,
            'elec_archi': valid_elec_archi,
            'data_number': data_number}


def _curves_coeffs_Arab06(specs_df, data_completeness):
    """
    Compute curve-fitting coefficient with method of Hadj Arab [1] and
    Djoudi Gherbi [2].

    It uses a 3rd order polynomial to model Q(P) and
    a 1st order polynomial to model V(I). Each corresponding
    coefficient depends on TDH through a 3rd order polynomial.

    Parameters
    ----------
    specs_df: pd.DataFrame
        DataFrame with specs.

    Reference
    ---------
    [1] Hadj Arab A., Benghanem M. & Chenlo F.,
    "Motor-pump system modelization", 2006, Renewable Energy
    [2] Djoudi Gherbi, Hadj Arab A., Salhi H., "Improvement and validation
    of PV motor-pump model for PV pumping system performance analysis", 2017

    """
    funct_mod_I = function_models.polynomial_1
    funct_mod_coeffs = function_models.polynomial_3
# TODO: make the regression directly on the compound function, as in
# the theoretical case.
#    funct_mod_I_compound = function_models.compound_polynomial_1_3

    # TODO: add check on number of head available (for lin. reg. of coeffs)
    if data_completeness['data_number'] >= 10 \
            and data_completeness['voltage_number'] >= 3:
        funct_mod_Q = function_models.polynomial_2
        funct_mod_Q_order = 2
    elif data_completeness['data_number'] >= 10 \
            and data_completeness['voltage_number'] >= 2:
        funct_mod_Q = function_models.polynomial_1
        funct_mod_Q_order = 1
    else:
        raise errors.InsufficientDataError('Lack of information on lpm, '
                                           'current or tdh for pump.')

    # linear equation I = a + b*V
    a_ls = []
    b_ls = []
    tdh_1 = []
    for H in specs_df.tdh:
        if H not in tdh_1:
            sub_df = specs_df[specs_df.tdh == H]
            if len(sub_df) >= 2:
                params_a_b, _ = opt.curve_fit(
                    funct_mod_I, sub_df.power, sub_df.current)
                a_ls.append(params_a_b[0])
                b_ls.append(params_a_b[1])
                tdh_1.append(H)

    # TODO: compare Q(P) or Q(V) (original article uses the latter)
    # quadratic equation Q = c + d*P + e*P**2
    c_ls = []
    d_ls = []
    e_ls = []
    tdh_2 = []
    for H in specs_df.tdh:
        if H not in tdh_2:
            sub_df = specs_df[specs_df.tdh == H]
            if len(sub_df) > funct_mod_Q_order:
                params_c_d_e, _ = opt.curve_fit(
                    funct_mod_Q, sub_df.power, sub_df.flow)
                c_ls.append(params_c_d_e[0])
                d_ls.append(params_c_d_e[1])
                tdh_2.append(H)
                if data_completeness['voltage_number'] >= 3:  # useless ?
                    e_ls.append(params_c_d_e[2])
                else:
                    e_ls.append(0)

    # dependance of a, b, c, d, e on H
    coeffs_dict = {}
    for coeff in ['a', 'b']:
        params, _ = opt.curve_fit(
            funct_mod_coeffs, tdh_1, eval(coeff + '_ls'))
        coeffs_dict[coeff] = params
    for coeff in ['c', 'd', 'e']:  # two loops needed because tdh_2 != tdh_1
        params, _ = opt.curve_fit(
            funct_mod_coeffs, tdh_2, eval(coeff + '_ls'))
        coeffs_dict[coeff] = params

    return coeffs_dict

# TODO: standardize output with statistical values as well
#    param_f1 = coeffs_dict['a', 'b']
#    stats_f1 = function_models.correlation_stats(funct_mod_I_compound,
#                                                 (param_f1['a'],
#                                                 param_f1['b']),
#                                                 dataxy_ar, dataz_ar)
#    param_f2 = coeffs_dict['c', 'd', 'e']
#
#    return {'coeffs_f1': param_f1,
#            'rmse_f1': stats_f1['rmse'],
#            'nrmse_f1': stats_f1['nrmse'],
#            'r_squared_f1': stats_f1['r_squared'],
#            'coeffs_f2': param_f2,
#            'rmse_f2': stats_f2['rmse'],
#            'nrmse_f2': stats_f2['nrmse'],
#            'r_squared_f2': stats_f2['r_squared']}


def _curves_coeffs_Kou98(specs_df, data_completeness):
    """Compute curve-fitting coefficient with method of Kou [1].

    It uses a 3rd order multivariate polynomial with cross terms to model
    V(I, TDH) and Q(V, TDH) from the data.

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

    if data_completeness['data_number'] >= 10 \
            and data_completeness['voltage_number'] > 3:
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
            funct_mod, dataxy_ar, dataz_ar)
#            , bounds=([0, 0, -np.inf, -np.inf, -np.inf],
#                    [np.inf, np.inf, 0, 0, 0]))
    # computing of RMSE and of normalized RMSE for f1
    stats_f1 = function_models.correlation_stats(funct_mod, param_f1,
                                                 dataxy_ar, dataz_ar)

    # f2: Q(P, H)
    datax_df = specs_df['power']
    dataz_df = specs_df['flow']
    dataxy_ar = [np.array(datax_df),
                 np.array(datay_df)]
    dataz_ar = np.array(dataz_df)

    param_f2, covmat_f2 = opt.curve_fit(funct_mod, dataxy_ar,
                                        dataz_ar)
    # computing of RMSE and of normalized RMSE for f2
    stats_f2 = function_models.correlation_stats(funct_mod, param_f2,
                                                 dataxy_ar, dataz_ar)

    return {'coeffs_f1': param_f1,
            'rmse_f1': stats_f1['rmse'],
            'nrmse_f1': stats_f1['nrmse'],
            'r_squared_f1': stats_f1['r_squared'],
            'coeffs_f2': param_f2,
            'rmse_f2': stats_f2['rmse'],
            'nrmse_f2': stats_f2['nrmse'],
            'r_squared_f2': stats_f2['r_squared']}


def _curves_coeffs_Hamidat08(specs_df, data_completeness):
    """
    Compute curve-fitting coefficient with method of Hamidat [1].
    It uses a 3rd order polynomial to model Q(P) and each corresponding
    coefficient depends on TDH through a 3rd order polynomial as well.

    Parameters
    ----------
    specs_df: pd.DataFrame
        DataFrame with specs.

    Returns
    -------
    * dict with coefficients of the curve fit (a1, a2, a3, a4, b1, ..., d4)

    Reference
    ---------
    [1] Hamidat A., ..., 2008, Renewable Energy

    """

    if data_completeness['data_number'] >= 10 \
            and data_completeness['voltage_number'] > 3:
        funct_mod_Q = function_models.polynomial_3
        funct_mod_Q_order = 3
        funct_mod_coeffs = function_models.polynomial_3
    else:
        raise errors.InsufficientDataError('Lack of information on lpm, '
                                           'current or tdh for pump.')

    # cubic equation P = a + b*Q + c*Q**2 + d*Q**3
    a_ls = []
    b_ls = []
    c_ls = []
    d_ls = []
    tdh = []
    for H in specs_df.tdh:
        if H not in tdh:
            sub_df = specs_df[specs_df.tdh == H]
            if len(sub_df) > funct_mod_Q_order:
                params, _ = opt.curve_fit(
                    funct_mod_Q, sub_df.flow, sub_df.power)
                a_ls.append(params[0])
                b_ls.append(params[1])
                c_ls.append(params[2])
                d_ls.append(params[3])
                tdh.append(H)

    # dependance of a, b, c, d, e on H
    coeffs_dict = {}
    for coeff in ['a', 'b', 'c', 'd']:
        params, _ = opt.curve_fit(
            funct_mod_coeffs, tdh, eval(coeff + '_ls'))
        coeffs_dict[coeff] = params

# TODO: standardize output with statistical values as well
    return coeffs_dict


def _curves_coeffs_theoretical(specs_df, data_completeness, elec_archi):
    """Compute curve-fitting coefficient following theoretical analysis of
    motor architecture.

    It uses a equation of the form V = R_a*i + beta(H)*np.sqrt(i) to model
    V(I, TDH) and an equation of the form Q = (a + b*H) * (c + d*P) to model
    Q(P, TDH) from the data.

    This kind of equation is used in [1], [2], ...


    Parameters
    ----------
    specs_df: pd.DataFrame
        DataFrame with specs.

    Reference
    ---------
    [1] Mokkedem & al, ...

    """
    if elec_archi != 'permanent_magnet':
        raise NotImplementedError(
            'This model is not implemented yet for electrical architecture '
            'different from permanent magnet motor.')

    if not data_completeness['data_number'] >= 10 \
            and data_completeness['voltage_number'] > 3:
        raise errors.InsufficientDataError('Lack of information on lpm, '
                                           'current or tdh for pump.')

    # gives f1: V=f1(i, H)
    def funct_mod_v(input_values, R_a, beta_0, beta_1, beta_2):
        """Returns the equation v(i, h).
        """
        i, h = input_values
        funct_mod_beta = function_models.polynomial_2
        beta = funct_mod_beta(h, beta_0, beta_1, beta_2)
        return R_a*i + beta*np.sqrt(i)

    dataxy = [np.array(specs_df.current),
              np.array(specs_df.tdh)]
    dataz = np.array(specs_df.voltage)
    param_f1, matcov = opt.curve_fit(funct_mod_v, dataxy, dataz, maxfev=10000)

    stats_f1 = function_models.correlation_stats(funct_mod_v, param_f1,
                                                 dataxy, dataz)

    # gives f2; Q=f2(P, H)
    def funct_mod_Q(input_values, a, b, c, d):
        P, H = input_values
        return (a + b*H) * (c + d*P)
        # theoretically it should be the following formula, but doesn't work
        # return (a + b*H + c*H**2) * P/H

    dataxy = [np.array(specs_df.power),
              np.array(specs_df.tdh)]
    dataz = np.array(specs_df.flow)
    param_f2, matcov = opt.curve_fit(funct_mod_Q, dataxy, dataz, maxfev=10000)

    stats_f2 = function_models.correlation_stats(funct_mod_Q, param_f2,
                                                 dataxy, dataz)

    return {'coeffs_f1': param_f1,
            'rmse_f1': stats_f1['rmse'],
            'nrmse_f1': stats_f1['nrmse'],
            'r_squared_f1': stats_f1['r_squared'],
            'coeffs_f2': param_f2,
            'rmse_f2': stats_f2['rmse'],
            'nrmse_f2': stats_f2['nrmse'],
            'r_squared_f2': stats_f2['r_squared']}


def _curves_coeffs_Gualteros17(lpm, tdh, watts):
    """
    Sergio Gualteros method. In his method the pump is modeled only for the
    static head of the hydraulic circuit.

    Based on Hamidat & Benyoucef, 2008 ->
    Model with polynomial_3rd_order P(Q) and then uses Newton-Raphson method

    Compute curve-fitting coefficient from data for :
        - tdh vs lpm
        - power vs lpm
        - power vs tdh

    returns a dict of sub-dict :
        -the first dict contains the 2 curves as keys : 'tdh','pow'
        resp. for total dynamic head and power
            -the sub-dicts contain the available voltage as keys, typically
            '60','75','90','105','120'
    These same 3 dictionnary are saved as attributes in the pump object,
    under the name 'self.coeff_tdh', 'self.coeff_pow'
    """
#    raise NotImplementedError('Even though this function exists, it '
#                              'cannot be used in later process')

    func_model = function_models.polynomial_4

    # this function allows to simplify the next equation
#    coeff_pow_with_tdh = coeff_pow_with_tdh_at_static_head(static_head,
#                                                           hydraulic_circuit)
    coeff_pow_with_tdh = {}  # coeff from curve-fitting of power vs tdh

    coeff_tdh_with_lpm = {}  # coeff from curve-fitting of tdh vs lpm
    coeff_pow_with_lpm = {}  # coeff from curve-fitting of power vs lpm

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


def _domain_I_H(specs_df, data_completeness):
    """
    Function giving the domain of cur depending on tdh, and vice versa.

    """

    funct_mod = function_models.polynomial_2

    # domains of I and tdh depending on each other
    data_v = specs_df.voltage.drop_duplicates()

    cur_tips = []
    tdh_tips = []
    for v in data_v:
        cur_tips.append(min(specs_df[specs_df.voltage == v].current))
        tdh_tips.append(max(specs_df[specs_df.voltage == v].tdh))

    if data_completeness['voltage_number'] > 2 \
            and data_completeness['lpm_min'] == 0:
        # case working fine for SunPumps - not sure about complete data from
        # other manufacturer
        param_tdh, pcov_tdh = opt.curve_fit(funct_mod,
                                            cur_tips, tdh_tips)
        param_cur, pcov_cur = opt.curve_fit(funct_mod,
                                            tdh_tips, cur_tips)

        def interval_cur(y):
            "Interval on x depending on y"
            return [max(funct_mod(y, *param_cur), min(specs_df.current)),
                    max(specs_df.current)]

        def interval_tdh(x):
            "Interval on y depending on x"
            return [0, min(max(funct_mod(x, *param_tdh),
                               0),
                           max(specs_df.tdh))]

    else:
        # Would need deeper work to fully understand what are the limits
        # on I and V depending on tdh, and how it affects the flow rate
        def interval_cur(*args):
            "Interval on current, independent of tdh"
            return [min(specs_df.current), max(specs_df.current)]

        def interval_tdh(*args):
            "Interval on tdh, independent of current"
            return [min(specs_df.tdh), max(specs_df.tdh)]

    return interval_cur, interval_tdh


def _domain_V_H(specs_df, data_completeness):
    """
    Function giving the range of voltage and head in which the pump will
    work.
    """
    funct_mod = function_models.polynomial_2

    data_v = specs_df.voltage.drop_duplicates()
    tdh_tips = []
    for v in data_v:
        tdh_tips.append(max(specs_df[specs_df.voltage == v].tdh))

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
                           max(specs_df.tdh))]

    else:
        # Would need deeper work to fully understand what are the limits
        # on I and V depending on tdh, and how it affects lpm
        def interval_vol(*args):
            "Interval on vol, independent of tdh"
            return [min(data_v), max(data_v)]

        def interval_tdh(*args):
            "Interval on tdh, independent of vol"
            return [0, max(specs_df.tdh)]

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
            power_max_for_tdh = max(specs_df[specs_df.tdh <= tdh].power)
            return [max(funct_mod(tdh, *param_pow), min(datapower_ar)),
                    power_max_for_tdh]

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
            power_max_for_tdh = max(specs_df[specs_df.tdh <= tdh].power)
            return [max(funct_mod(tdh, *param_pow), min(datapower_ar)),
                    power_max_for_tdh]

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
            return [0, max(datay_ar)]

    return interval_power, interval_tdh


if __name__ == "__main__":
    # %% pump creation
    pump1 = Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                 model='SCB_10', modeling_method='arab',
                 motor_electrical_architecture='permanent_magnet')

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
                 motor_electrical_architecture='permanent_magnet',
                 modeling_method='arab')

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

    coeffs_1 = _curves_coeffs_Hamidat08(pump1.specs_df, pump1.data_completeness)
    coeffs_2 = _curves_coeffs_theoretical(pump1.specs_df,
                                          pump1.data_completeness,
                                          pump1.motor_electrical_architecture)
    coeffs_3 = _curves_coeffs_Arab06(pump1.specs_df, pump1.data_completeness)
    coeffs_4 = _curves_coeffs_Kou98(pump1.specs_df, pump1.data_completeness)

#    print(pump1.coeffs)
#    f2, _ = pump1.functQforPH()
#    print(f2(400, 10))
#    intervals = _domain_I_H(pump1.specs_df, pump1.data_completeness)
#    print(intervals)


#    print(pump1.coeffs)
#    print(pump2.coeffs)
#    f11, _ = pump1.functIforVH()
#    f21, _ = pump1.functQforPH()
#    f12, _ = pump2.functIforVH()
#    f22, _ = pump2.functQforPH()

#    pump1.plot_tdh_Q()
#    pump2.plot_tdh_Q()

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
#    pump_concerned = pump2
#    f4, stddev = pump_concerned.functQforPH()
#    lpm_check = []
#
#    for index, row in pump_concerned.specs_df.iterrows():
#        try:
#            Q = f4(row.power, row.tdh)
#        except (errors.PowerError, errors.HeadError):
#            Q = 0
#        lpm_check.append(Q)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d', title='Q (lpm) as a function of'
#                         ' power (W) and static head (m)')
#    ax.scatter(pump_concerned.specs_df.power, pump_concerned.specs_df.tdh,
#               pump_concerned.specs_df.flow,
#               label='from data')
#    ax.scatter(pump_concerned.specs_df.power, pump_concerned.specs_df.tdh,
#               lpm_check,
#               label='from curve fitting with modeling method {0}'.format(
#                       pump_concerned.modeling_method))
#    ax.set_xlabel('power')
#    ax.set_ylabel('head')
#    ax.set_zlabel('discharge Q')
#    ax.legend(loc='lower left')
#    plt.show()
#    print('std dev on Q calculated from P: ', stddev)
