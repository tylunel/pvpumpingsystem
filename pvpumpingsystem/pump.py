# -*- coding: utf-8 -*-
"""
Created on Fri May 17 07:54:42 2019

@author: Sergio Gualteros, Tanguy Lunel

module defining class and functions for modeling the pump.

"""
import collections
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkfile
from itertools import count
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for plotting in 3d
from matplotlib.collections import PatchCollection
import scipy.optimize as opt
import scipy.interpolate as spint
import pandas as pd

from pvpumpingsystem import errors


class Pump:
    """
    Class representing a pump.

    Attribute:
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


        category: str,
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
                 model=None, category=None,
                 price=None, power_rating=None,
                 controler=None, diameter_output=None):
        # use input data to create pump object
        if None not in (lpm, tdh, current):
            self.lpm = lpm
            self.tdh = tdh
            self.current = current
            self.voltage = list(self.lpm.keys())
            self.watts = get_watts_from_current(current)

        # retrieve pump data from txt datasheet given by path
        try:
            self.voltage, self.lpm, self.tdh, self.current, \
                self.watts = getdatapump(path)
        except IOError:
            print('The mentionned path does not exist, please select another'
                  ' in the pop-up window.')
            tk.Tk().withdraw()
            filepath = tkfile.askopenfilename()
            self.path = filepath
            self.voltage, self.lpm, self.tdh, self.current, \
                self.watts = getdatapump(path)

        self.model = model
        self.category = category
        self.price = price
        self.power_rating = power_rating
        self.controler = controler
        self.diameter_output = diameter_output

        self.coeff_pow = None
        self.coeff_tdh = None

        self.id = next(self._ids)

    def __repr__(self):
        affich = "model :" + str(self.model) + \
                 "\ncategory :" + str(self.category) + \
                 "\nprice :" + str(self.price) + \
                 "\npower rating (HP) :" + str(self.power_rating) + \
                 "\ncontroler :" + str(self.controler) + \
                 "\noutput diameter (inches) :" + str(self.diameter_output)
        return affich

    def curves_coeffs(self):
        """Compute curve-fitting coefficient from data for :
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
        def func_model(x, a, b, c, d, e):
            return a*x**4+b*x**3+c*x**2+d*x+e

        self.coeff_tdh = {}  # coeff from curve-fitting of tdh vs lpm
        self.coeff_pow = {}  # coeff from curve-fitting of power vs lpm

        for V in self.voltage:
            # curve-fit of tdh vs lpm
            coeffs_tdh, matcov = opt.curve_fit(func_model, self.lpm[V],
                                               self.tdh[V])
            self.coeff_tdh[V] = coeffs_tdh

            # curve-fit of power vs lpm
            coeffs_P, matcov = opt.curve_fit(func_model, self.lpm[V],
                                             self.watts[V])
            self.coeff_pow[V] = coeffs_P

        return {'tdh': self.coeff_tdh,
                'pow': self.coeff_pow}


    def startingVPI(self, tdh):
        """
        ------------------------- TO CHECK !! -------------------------
        --------- consistant with results from functVforIH ??? --------

        Returns the required starting voltage, power and current
        for a specified tdh.

        returns :
            {'V':vmin,'P':pmin,'I':imin}
            vmin is :
                None: if tdh out of the range of the pump
                float: value of minimum starting voltage

        """
        raise NotImplementedError

        if self.coeff_tdh is None:
            self.curves_coeffs()

        tdhmax = {}
        powmin = {}
        for V in self.voltage:
            tdhmax[V] = self.coeff_tdh[V][4]  # y-intercept of tdh vs lpm
            powmin[V] = self.coeff_pow[V][4]

        # interpolation:
        # of Vmin vs tdh
        newf_t = spint.interp1d(list(tdhmax.values()), list(tdhmax.keys()),
                                kind='cubic')
        # of power vs V
        newf_p = spint.interp1d(list(powmin.keys()), list(powmin.values()),
                                kind='cubic')

        if tdh < min(tdhmax.values()):
            print('The resqueted tdh is out of the range for the pump,'
                  'it is below the minimum tdh.')
            vmin = 'below'
            pmin = None
            imin = None
        elif tdh > max(tdhmax.values()):
            print('The resqueted tdh is out of the range for the pump,'
                  'it is above the maximum tdh delivered by the pump.')
            vmin = 'above'
            pmin = None
            imin = None
        else:
            vmin = newf_t(tdh)
            pmin = newf_p(vmin)
            imin = pmin/vmin
        return {'V': vmin, 'P': pmin, 'I': imin}


    def plot_tdh_Q(self):
        """Print the graph of tdh(in m) vs Q(in lpm)
        """
        if self.coeff_eff is None:
            self.curves_coeffs()
            self.opti_zone()

        tdh_x = {}
        eff_x = {}
        # greatest value of lpm encountered in data
        lpm_max = max(self.lpm[max(self.voltage)])
        self.lpm_x = np.arange(0, lpm_max)  # vector of lpm

        for V in self.voltage:
            def eff_funct(x):
                # efficiency function
                return self.coeff_eff[V][0]*x**4 + self.coeff_eff[V][1]*x**3 +\
                        self.coeff_eff[V][2]*x**2 + self.coeff_eff[V][3]*x + \
                        self.coeff_eff[V][4]

            def tdh_funct(x):
                # function tdh
                return self.coeff_tdh[V][0]*x**4 + self.coeff_tdh[V][1]*x**3 +\
                        self.coeff_tdh[V][2]*x**2 + self.coeff_tdh[V][3]*x + \
                        self.coeff_tdh[V][4]

            # vectors of tdh and efficiency with lpm - ready to be printed
            tdh_x[V] = tdh_funct(self.lpm_x)
            eff_x[V] = eff_funct(self.lpm_x)

        fig = plt.figure(facecolor='White')
        # add space in height between the subplots:
        fig.subplots_adjust(hspace=0.5)
        ax1 = plt.subplot(2, 1, 1)

        for i, V in enumerate(self.voltage):  # for each voltage available :
            # get the next color to have the same color by voltage:
            col = next(ax1._get_lines.prop_cycler)['color']
            plot(self.lpm_x, tdh_x[V], linestyle='-', linewidth=2, color=col,
                 label=str(V)+'VDC')
            plot(self.lpm_x, eff_x[V], linestyle='--', linewidth=1, color=col,
                 label='efficiency')
        ax1.set_title(str(self.model) +
                      ' Courbes Debit Vs. Hauteur manometrique et efficiency')
        ax1.set_xlabel('lpm')
        ax1.set_ylabel('Hauteur manometrique (m) / (%)')
        ax1.set_ylim(0, max(tdh_x[max(self.voltage)]))
        ax1.legend(loc='best')
        ax1.grid(True)
        patches = []
        patches.append(self.polygon)
        collection = PatchCollection(patches, alpha=0.5)
        ax1.add_collection(collection)

        ax2 = plt.subplot(2, 1, 2)
        for V in self.voltage:
            plot(self.lpm[V], self.watts[V], linewidth=2,
                 label=str(V) + ' VDC')
        ax2.set_xlabel('lpm')
        ax2.set_ylabel('watts')
        ax2.set_title(str(self.model) +
                      'Courbes Debit Vs. Puissance electrique')
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
        def funct_mod(input_val, a, c1, c2, h1, h2, t1):
            '''model for linear regression'''
            x, y = input_val[0], input_val[1]
            return a + c1*x + c2*x**2 + h1*y + h2*y**2 + t1*x*y

        def funct_model_intervals(input_val, a, b, c):
            '''model for linear regression of tdh(V) and V(tdh)'''
            x = input_val
            return a + b*x + c*x**2
        # gathering of data
        vol = []
        tdh = []
        cur = []
        for V in self.voltage:
            for i, Idata in enumerate(self.current[V]):
                vol.append(V)
                tdh.append(self.tdh[V][i])
                cur.append(Idata)

        datax = [np.array(cur), np.array(tdh)]
        dataz = np.array(np.array(vol))
        # computing of linear regression
        para, covmat = opt.curve_fit(funct_mod, datax, dataz)
        # comparison between linear reg and actual data
        datacheck = funct_mod(datax, para[0], para[1], para[2], para[3],
                              para[4], para[5])
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        # domains of I and tdh depending on each other
        data_i = []
        data_tdh = []
        for key in self.tdh.keys():
            data_i.append(min(self.current[key]))
            data_tdh.append(max(self.tdh[key]))
        param_tdh, pcov_tdh = opt.curve_fit(funct_model_intervals,
                                            data_i, data_tdh)
        param_i, pcov_i = opt.curve_fit(funct_model_intervals,
                                        data_tdh, data_i)

        def interval_i(tdh):
            "Interval on i depending on tdh"
            return [max(funct_model_intervals(tdh, *param_i), min(cur)),
                    max(cur)]

        def interval_tdh(i):
            "Interval on tdh depending on i"
            return [0, min(max(funct_model_intervals(i, *param_tdh), 0),
                           max(tdh))]
        # domain of I and tdh and gathering in one single variable
        intervals = {'I': interval_i,
                     'H': interval_tdh}

        def functV(I, H, error_raising=True):
            """Function giving voltage V according to current I and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not interval_i(H)[0] <= I <= interval_i(H)[1]:
                    raise errors.CurrentError(
                            'I (={0}) is out of bounds. For this specific '
                            'head H (={1}), I should be in the interval {2}'
                            .format(I, H, interval_i(H)))
                if not interval_tdh(I)[0] <= H <= interval_tdh(I)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'current I (={1}), H should be in the interval {2}'
                            .format(H, I, interval_tdh(I)))
            return para[0] + para[1]*I + para[2]*I**2 + para[3]*H + \
                para[4]*H**2 + para[5]*I*H

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
        def funct_model(input_val, a, v1, v2, v3, h1, h2, h3, t1):
            """model for linear regression"""
            x, y = input_val[0], input_val[1]
            return a + v1*x + v2*x**2 + v3*x**3 + \
                h1*y + h2*y**2 + h3*y**3 + t1*x*y

        def funct_model_intervals(input_val, a, b, c):
            '''model for linear regression of tdh(V)'''
            x = input_val
            return a + b*x + c*x**2

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
        para, covmat = opt.curve_fit(funct_model, dataxy, dataz)
        datacheck = funct_model(dataxy, para[0], para[1], para[2], para[3],
                                para[4], para[5], para[6], para[7])
        ectyp = np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

        # domains of I and tdh depending on each other
        data_v = []
        data_tdh = []
        for key in self.tdh.keys():
            data_v.append(key)
            data_tdh.append(max(self.tdh[key]))
        param_tdh, pcov_tdh = opt.curve_fit(funct_model_intervals,
                                            data_v, data_tdh)
        param_v, pcov_v = opt.curve_fit(funct_model_intervals,
                                        data_tdh, data_v)

        def interval_v(tdh):
            "Interval on v depending on tdh"
            return [max(funct_model_intervals(tdh, *param_v), min(vol)),
                    max(vol)]

        def interval_tdh(v):
            "Interval on tdh depending on v"
            return [0, min(max(funct_model_intervals(v, *param_tdh), 0),
                           max(tdh))]
        # domain of V and tdh and gathering in one single variable
        intervals = {'V': interval_v,
                     'H': interval_tdh}

        def functI(V, H, error_raising=True):
            """Function giving voltage V according to current I and tdh H.

            Error_raising parameter allows to check the given values
            according to the possible intervals and to raise errors if not
            corresponding.
            """
            if error_raising is True:
                if not interval_v(H)[0] <= V <= interval_v(H)[1]:
                    raise errors.VoltageError(
                            'V (={0}) is out of bounds. For this specific '
                            'head H (={1}), V should be in the interval {2}'
                            .format(V, H, interval_v(H)))
                if not interval_tdh(V)[0] <= H <= interval_tdh(V)[1]:
                    raise errors.HeadError(
                            'H (={0}) is out of bounds. For this specific '
                            'voltage V (={1}), H should be in the interval {2}'
                            .format(H, V, interval_tdh(V)))
            return para[0] + para[1]*V + para[2]*V**2 + para[3]*V**3 + \
                para[4]*H + para[5]*H**2 + para[6]*H**3 + para[7]*V*H

        return functI, ectyp, intervals

    def functQforVH(self):
        """Returns a tuple containing :
            -the function giving Q according to V and H static for the pump :
                Q = f2(V,H)
            -the standard deviation on Q between real data points and data
                computed with this function
        """
        def funct_mod(inp, a, v1, v2, h1, h2, t1):# model for linear regression
            x, y = inp[0], inp[1]
            return a + v1*x + v2*x**2 + h1*y + h2*y**2 + t1*x*y
        # gathering of data needed
        vol = []
        tdh = []
        lpm = []
        for V in self.voltage:
            for i, Q in enumerate(self.lpm[V]):
                vol.append(V)
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        datax = [np.array(vol), np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        para, covmat = opt.curve_fit(funct_mod, datax, dataz)

        datacheck = funct_mod(datax, para[0], para[1], para[2], para[3],
                              para[4], para[5])
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

            return para[0] + para[1]*V + para[2]*V**2 + para[3]*H + \
                para[4]*H**2 + para[5]*V*H

        return functQ, ectyp

    def functQforPH(self):
        """Returns a tuple containing :
            -the function giving Q according to P and H static for the pump :
                Q = f2(P,H)
            -the standard deviation on Q between real data points and data
                computed with this function
        """

        def funct_mod(inp, a, v1, v2, h1, h2, t1):
            # model for linear regression
            x, y = inp[0], inp[1]
            return a + v1*x + v2*x**2 + h1*y + h2*y**2 + t1*x*y
        # gathering of data needed
        power = []
        tdh = []
        lpm = []
        for V in self.voltage:
            for i, Q in enumerate(self.lpm[V]):
                power.append(self.watts[V][i])
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        datax = [np.array(power), np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        para, covmat = opt.curve_fit(funct_mod, datax, dataz)

        datacheck = funct_mod(datax, para[0], para[1], para[2], para[3],
                              para[4], para[5])
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

            return para[0] + para[1]*P + para[2]*P**2 + para[3]*H + \
                para[4]*H**2 + para[5]*P*H

        return functQ, ectyp

    def IVcurvedata(self, head, nbpoint=40):
        """Function returning the data needed for plotting the IV curve at
        a given head.

        returns:
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

    current_df = pd.DataFrame(current_dict)
    watts_df = current_df*current_df.columns
    watts_dict = wattdf.to_dict('list')

    return watts_dict


def getdatapump(path):
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


if __name__=="__main__":
#%% pump creation
    pump1 = Pump(path="pumps_files/SCB_10_150_120_BL.txt",
                 model='SCB_10')

#%% set-up for following plots
    vol = []
    tdh = []
    cur = []
    lpm = []
    power = []
    for V in pump1.voltage:
        for i, I in enumerate(pump1.current[V]):
            vol.append(V)
            cur.append(I)
            tdh.append(pump1.tdh[V][i])
            lpm.append(pump1.lpm[V][i])
            power.append(pump1.watts[V][i])

#    # alternative way (faster ?): convert in array and flatten it
#    volflat=np.array(list(vol.values())).flat[:]
#    tdhflat=np.array(list(tdh.values())).flat[:]
#    lpmflat=np.array(list(lpm.values())).flat[:]

#%% plot of functVforIH
    f1, stddev, intervals = pump1.functVforIH()
    vol_check = []
    for i, I in enumerate(cur):
        vol_check.append(f1(I, tdh[i], error_raising=False))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='Voltage as a function of'
                         ' current (A) and static head (m)')
    ax.scatter(cur, tdh, vol, label='from data')
    ax.scatter(cur, tdh, vol_check, label='from curve fitting')
    ax.set_xlabel('current')
    ax.set_ylabel('head')
    ax.set_zlabel('voltage V')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on V:', stddev)
    print('V for IH=(4,25): {0:.2f}'.format(f1(4, 25)))

#%% plot of functIforVH
    f1, stddev, intervals = pump1.functIforVH()
    cur_check = []
    for i, V in enumerate(vol):
        cur_check.append(f1(V, tdh[i], error_raising=False))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='Current as a function of'
                         ' voltage (V) and static head (m)')
    ax.scatter(vol, tdh, cur, label='from data')
    ax.scatter(vol, tdh, cur_check, label='from curve fitting')
    ax.set_xlabel('voltage')
    ax.set_ylabel('head')
    ax.set_zlabel('current I')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on I: ', stddev)
    print('I for VH=(89,25): {0:.2f}'.format(f1(89, 25)))


#%% plot of functQforVH
    f2, stddev = pump1.functQforVH()
    lpm_check = []
    for i, v in enumerate(vol):
        try:
            Q = f2(v, tdh[i])
        except (errors.VoltageError, errors.HeadError):
            Q = 0
        lpm_check.append(Q)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='Q (lpm) as a function of'
                         ' voltage (V) and static head (m)')
    ax.scatter(vol, tdh, lpm, label='from data')
    ax.scatter(vol, tdh, lpm_check, label='from curve fitting')
    ax.set_xlabel('voltage')
    ax.set_ylabel('head')
    ax.set_zlabel('discharge Q')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on Q calculated from V:', stddev)
    print('Q for VH=(74,25): {0:.2f}'.format(f2(74, 25)))


#%% plot of functQforPH
    f2, stddev = pump1.functQforPH()
    lpm_check = []
    for i, po in enumerate(power):
        try:
            Q = f2(po, tdh[i])
        except (errors.PowerError, errors.HeadError):
            Q = 0
        lpm_check.append(Q)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title='Q (lpm) as a function of'
                         ' power (W) and static head (m)')
    ax.scatter(power, tdh, lpm, label='from data')
    ax.scatter(power, tdh, lpm_check, label='from curve fitting')
    ax.set_xlabel('power')
    ax.set_ylabel('head')
    ax.set_zlabel('discharge Q')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on Q calculated from P: ', stddev)
    print('Q for PH=(100,25): {0:.2f}'.format(f2(100, 25)))

