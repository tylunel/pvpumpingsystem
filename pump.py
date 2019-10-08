# -*- coding: utf-8 -*-
"""
Created on Fri May 17 07:54:42 2019

@author: Sergio Gualteros, Tanguy Lunel

module defining a pump and functions for modeling of the pump.

----Still to come :
    - change functVforIH and functQforVH so as they accept array as input


"""
import collections
import numpy as np
import tkinter as tk
import tkinter.filedialog as tkfile
from itertools import count
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.optimize import curve_fit, fmin
import scipy.optimize as opt
import scipy.interpolate as spint

import errors


class Pump:
    """
    Class representing a pump.

    The minimum attributes are :

        path (str) : given through constructor or through window
            -> contains the path to the txt file with specifications
    optionnal attributes are :
        cat (str) : centrifugal or positive displacement
        model (str) : name of the pump
        price (numeric)
        power (numeric)
        controler
        output diameter (numeric)
        data extracted from datasheet :
            (tension, lpm, tdh, Courant, watts, efficacite ).
    """
    _ids = count(1)

    def __init__(self,model=None,cat=None,price=None,path=None
                 ,power=None,controler=None,diam_out=None):
        if path == None:
            tk.Tk().withdraw()
            filepath = tkfile.askopenfilename()
            self.path=filepath

        try :
            self.tension, self.lpm, self.tdh, self.courant, self.watts, \
            self.efficacite = getdatapump(path) # getdatapump gather data  \
                                        #   from txt datasheet given by path
        except IOError:
            print('The mentionned path does not exist, please select another')
            tk.Tk().withdraw()
            filepath = tkfile.askopenfilename()
            self.path=filepath
            self.tension, self.lpm, self.tdh, self.courant, self.watts, \
            self.efficacite = getdatapump(filepath) # getdatapump gather data  \
                                            #   from txt datasheet given by path
        self.model = model
        self.cat = cat
        self.price = price
        self.power = power
        self.controler = controler
        self.diam_out = diam_out

        self.coeff_eff=None
        self.coeff_pow=None
        self.coeff_tdh=None

        self.id   = next(self._ids)

## Message confirming the correct creation of the object
#        print('\nPump created successfully')
#        print('Pump created with :')
#        for attr in self.__dict__:
#            print(attr,':', self.__dict__[attr],', ')
#        print('---------')


    def __repr__(self):
        affich= "model :" +str(self.model) + \
                "\ncategory :" +str(self.cat) + \
                "\nprice :" +str(self.price) + \
                "\npower (HP) :" +str(self.power) + \
                "\ncontroler :" +str(self.controler) + \
                "\noutput diameter (inches) :" +str(self.diam_out)
        return affich

    def curves_coeffs(self):
        """Compute curve-fitting coefficient from data for :
            - efficiency vs lpm
            - tdh vs lpm
            - power vs lpm

        returns a dict of sub-dict :
            -the first dict contains the 3 curves as keys : 'eff','tdh','pow'
            resp. for efficiency, total dynamic head and power
                -the sub-dicts contain the available voltage as keys, typically
                '60','75','90','105','120'
        These same 3 dictionnary are saved as attributes in the pump object,
        under the name 'self.coeff_eff', 'self.coeff_tdh', 'self.coeff_pow'
        """
        def func_model(x, a, b, c, d, e):
            return a*x**4+b*x**3+c*x**2+d*x+e

        self.coeff_eff  = {} # coeff from curve-fitting of efficiency vs lpm
        self.coeff_tdh  = {}# coeff from curve-fitting of tdh vs lpm
        self.coeff_pow  = {}# coeff from curve-fitting of power vs lpm

        for V in self.tension:
            # curve-fit of efficiency vs lpm
            coeffs_eff,matcov = curve_fit(func_model,self.lpm[V],
                                          self.efficacite[V])
            self.coeff_eff[V]=coeffs_eff# save the coeffs in dict

            # curve-fit of tdh vs lpm
            coeffs_tdh,matcov = curve_fit(func_model,self.lpm[V],
                                          self.tdh[V])
            self.coeff_tdh[V]=coeffs_tdh

            # curve-fit of power vs lpm
            coeffs_P,matcov = curve_fit(func_model,self.lpm[V],
                                        self.watts[V])
            self.coeff_pow[V]=coeffs_P

        return {'eff':self.coeff_eff,'tdh':self.coeff_tdh,'pow':self.coeff_pow}


    def startingVPI(self,tdh):
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
        if self.coeff_tdh==None:
            self.curves_coeffs()

        tdhmax={}
        powmin={}
        for V in self.tension:
            tdhmax[V] = self.coeff_tdh[V][4] # y-intercept of tdh vs lpm
            powmin[V] = self.coeff_pow[V][4]

        #interpolation:
        # of Vmin vs tdh
        newf_t=spint.interp1d(list(tdhmax.values()),list(tdhmax.keys()),
                              kind='cubic')
        # of power vs V
        newf_p=spint.interp1d(list(powmin.keys()),list(powmin.values()),
                              kind='cubic')

        if tdh<min(tdhmax.values()):
            print('The resqueted tdh is out of the range for the pump,'
                  'it is below the minimum tdh.')
            vmin='below'
            pmin=None
            imin=None
        elif tdh>max(tdhmax.values()):
            print('The resqueted tdh is out of the range for the pump,'
                  'it is above the maximum tdh delivered by the pump.')
            vmin='above'
            pmin=None
            imin=None
        else:
            vmin=newf_t(tdh)
            pmin=newf_p(vmin)
            imin=pmin/vmin

        return {'V':vmin,'P':pmin,'I':imin}

    def opti_zone(self,lpm_tol=0.05):
        """Fonction qui permet de calculer et tracer la zone d'opération
        préférentielle de la pompe.

        """

        # dict having the voltages as keys
        self.best_eff_pt   = {} # Best Efficiency Points (lpm,tdh)
        self.lim_min= {}# low limit of efficiency accepted
        self.lim_max= {}# high limit of efficiency accepted

        for V in self.tension:
            # efficiency function
            def eff_funct(x):
                return self.coeff_eff[V][0]*x**4+self.coeff_eff[V][1]*x**3+ \
                        self.coeff_eff[V][2]*x**2+self.coeff_eff[V][3]*x+ \
                        self.coeff_eff[V][4]

            # function tdh
            def tdh_funct(x):
                return self.coeff_tdh[V][0]*x**4+self.coeff_tdh[V][1]*x**3+ \
                        self.coeff_tdh[V][2]*x**2+self.coeff_tdh[V][3]*x+ \
                        self.coeff_tdh[V][4]

            # optimal efficiency : maximizing of eff_funct
            lpm_at_eff_max =fmin(lambda x: -eff_funct(x), 0)
            self.best_eff_pt[V] = (lpm_at_eff_max,tdh_funct(lpm_at_eff_max))

            # range of acceptability for lpm around efficiency max
            lpm_min=lpm_at_eff_max*(1 - lpm_tol) # low limit
            lpm_max=lpm_at_eff_max*(1 + lpm_tol) # high limit
            self.lim_min[V] = (lpm_min , tdh_funct(lpm_min))
            self.lim_max[V] = (lpm_max , tdh_funct(lpm_max))

        point        = np.zeros((4,2))
        point[0]     = self.lim_min[min(self.tension)]
        point[1]     = self.lim_min[max(self.tension)]
        point[2]     = self.lim_max[max(self.tension)]
        point[3]     = self.lim_max[min(self.tension)]
        self.polygon = Polygon(point,closed='True')
        # This polygon does not guarantee a good efficiency anywhere in it,
        # indeed, efficiency at 120V will be way higher than at 60V. It should
        # be changed for another kind of polygon in which the efficiency could
        # be at least a certain value.
        return self.polygon

    def plot_tdh_Q(self):
        """Print the graph of tdh(in m) vs Q(in lpm)

        """

        if self.coeff_eff==None:
            self.curves_coeffs()
            self.opti_zone()

        tdh_x = {}
        eff_x = {}
        #greatest value of lpm encountered in data
        lpm_max    = max(self.lpm[max(self.tension)])
        self.lpm_x = np.arange(0,lpm_max) # vector of lpm

        for V in self.tension:
            # efficiency function
            def eff_funct(x):
                return self.coeff_eff[V][0]*x**4+self.coeff_eff[V][1]*x**3+ \
                        self.coeff_eff[V][2]*x**2+self.coeff_eff[V][3]*x+ \
                        self.coeff_eff[V][4]
            # function tdh
            def tdh_funct(x):
                return self.coeff_tdh[V][0]*x**4+self.coeff_tdh[V][1]*x**3+ \
                        self.coeff_tdh[V][2]*x**2+self.coeff_tdh[V][3]*x+ \
                        self.coeff_tdh[V][4]

            # vectors of tdh and efficiency with lpm - ready to be printed
            tdh_x[V]   = tdh_funct(self.lpm_x)
            eff_x[V]   = eff_funct(self.lpm_x)

        fig = plt.figure(facecolor='White')
        # add space in height between the subplots:
        fig.subplots_adjust(hspace=0.5)

        ax1 = plt.subplot(2,1,1)

        for i,V in enumerate(self.tension):# for each voltage available :
            # get the next color to have the same color by voltage:
            col = next(ax1._get_lines.prop_cycler)['color']
            plot(self.lpm_x,tdh_x[V],linestyle='-',linewidth=2,color=col,
                 label=str(V)+'VDC')
            plot(self.lpm_x,eff_x[V],linestyle='--',linewidth=1,color=col,
                 label='efficiency')
        ax1.set_title(str(self.model)+ \
                      ' Courbes Debit Vs. Hauteur manometrique et efficacite')
        ax1.set_xlabel('lpm')
        ax1.set_ylabel('Hauteur manometrique (m) / (%)')
        ax1.set_ylim(0,max(tdh_x[max(self.tension)]))
        ax1.legend(loc='best')
        ax1.grid(True)
        patches = []
        patches.append(self.polygon)
        collection = PatchCollection(patches, alpha=0.5)
        ax1.add_collection(collection)

        ax2 = plt.subplot(2,1,2)
        for V in self.tension:
            plot(self.lpm[V],self.watts[V],linewidth=2,
                 label=str(V)+' VDC')
        ax2.set_xlabel('lpm')
        ax2.set_ylabel('watts')
        ax2.set_title(str(self.model)+'Courbes Debit Vs. Puissance electrique')
        ax2.grid(True)
        ax2.legend(loc='best')

        plt.show()


#    def plot_P_Q_atV(self,V):
#        """-----------------------not finished-------------------------
#        """
#
#        #greatest value of lpm encountered in data
#        lpm_max    = max(self.lpm.values())
#        self.lpm_x = np.arange(0,lpm_max) # vector of lpm
#        # efficiency function
#        def eff_funct(x):
#            return self.coeff_eff[V][0]*x**4+self.coeff_eff[V][1]*x**3+ \
#                    self.coeff_eff[V][2]*x**2+self.coeff_eff[V][3]*x+ \
#                    self.coeff_eff[V][4]
#        # function tdh
#        def tdh_funct(x):
#            return self.coeff_tdh[V][0]*x**4+self.coeff_tdh[V][1]*x**3+ \
#                    self.coeff_tdh[V][2]*x**2+self.coeff_tdh[V][3]*x+ \
#                    self.coeff_tdh[V][4]
#        # power function
#        def pow_funct(x):
#            return self.coeff_pow[V][0]*x**4+self.coeff_pow[V][1]*x**3+ \
#                    self.coeff_pow[V][2]*x**2+self.coeff_pow[V][3]*x+ \
#                    self.coeff_pow[V][4]
#
#        # vectors of tdh and efficiency with lpm - ready to be printed
#        tdh_x   = tdh_funct(self.lpm_x)
#        eff_x   = eff_funct(self.lpm_x)
#        pow_x   = pow_funct(self.lpm_x)
#
#
#        num   = np.size(self.tension)
#
#        fig = plt.figure(facecolor='White')
#        # add space in height between the subplots
#        fig.subplots_adjust(hspace=0.5)
#
#        ax1 = plt.subplot(2,1,1)
#
#        for i,V in enumerate(self.tension):# for each voltage available :
#            # get the next color to have the same color by voltage:
#            col = next(ax1._get_lines.prop_cycler)['color']
#            plot(self.lpm_x,self.tdh_x[V],linestyle='-',linewidth=2,color=col,
#                 label=str(V)+'VDC')
#            plot(self.lpm_x,self.eff_x[V],linestyle='--',linewidth=1,color=col,
#                 label='efficiency')
#        ax1.set_title(str(self.model)+ \
#                      ' Courbes Debit Vs. Hauteur manometrique et efficacite')
#        ax1.set_xlabel('lpm')
#        ax1.set_ylabel('Hauteur manometrique (m) / (%)')
#        ax1.set_ylim(0,max(tdh_x[max(self.tension)]))
#        ax1.legend(loc='best')
#        ax1.grid(True)
#        patches = []
#        patches.append(self.polygon)
#        collection = PatchCollection(patches, alpha=0.5)
#        ax1.add_collection(collection)
#
#        ax2 = plt.subplot(2,1,2)
#        for i in range (0,num):
#            plot(self.lpm[i],self.watts[i],linewidth=2,
#                 label=str(int(self.tension[i]))+' VDC')
#        ax2.set_xlabel('lpm')
#        ax2.set_ylabel('watts')
#        ax2.set_title(str(self.model)+'Courbes Debit Vs. Puissance electrique')
#        ax2.grid(True)
#        ax2.legend(loc='best')
#
#        plt.show()

    def functVforIH(self):
        """Returns a tuple containing :
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
        for V in self.tension:
            for i, Idata in enumerate(self.courant[V]):
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
            data_i.append(min(self.courant[key]))
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
        for V in self.tension:
            for i, I in enumerate(self.courant[V]):
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
        def funct_mod(inp,a,v1,v2,h1,h2,t1):# model for linear regression
            x,y = inp[0],inp[1]
            return a + v1*x + v2*x**2 + h1*y + h2*y**2 + t1*x*y
        # gathering of data needed
        vol=[]
        tdh=[]
        lpm=[]
        for V in self.tension:
            for i, Q in enumerate(self.lpm[V]):
                vol.append(V)
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        datax = [np.array(vol),np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        para, covmat = curve_fit(funct_mod, datax, dataz)

        datacheck=funct_mod(datax,para[0],para[1],para[2],para[3],
                            para[4],para[5])
        ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))

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
        def funct_mod(inp, a, v1, v2, h1, h2, t1):# model for linear regression
            x, y = inp[0], inp[1]
            return a + v1*x + v2*x**2 + h1*y + h2*y**2 + t1*x*y
        # gathering of data needed
        power = []
        tdh = []
        lpm = []
        for V in self.tension:
            for i, Q in enumerate(self.lpm[V]):
                power.append(self.watts[V][i])
                tdh.append(self.tdh[V][i])
                lpm.append(Q)

        datax = [np.array(power), np.array(tdh)]
        dataz = np.array(np.array(lpm))
        # computing of linear regression
        para, covmat = curve_fit(funct_mod, datax, dataz)

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


def getdatapump(path):
    """
    Cette fonction permet de charger les données de pompe présentes sur le
    fichier .txt désigné par le path. Ce fichiers .txt possède les
    caractéristiques des datasheets. Les données sont renvoyées sous
    forme de 6 tableaux (liste contenants des listes en vrai) :
    tension, lpm, tdh, courant, watts, efficacite.

    Fonctionne correctement. Attention cependant, cette fonction a été
    changé depuis la version en py2 et ne renvoie pas tout à fait la même
    chose (renvoie des listes de liste, à la place de tuple de tableaux).
    A priori l'appel aux valeurs contenues se fait de la même manière et
    ne devrait donc pas poser de problème, mais c'est à vérifier.
    Bref, rester prudent !

    (comes from former function 'data_pompe' in 'donnees_pompe')
    """

    #%% Importation des données
    data = np.loadtxt(path, dtype={'names': ('Tension', 'tdh', 'Courant',
                                             'lpm', 'watts','efficacite'),
                      'formats': (float, float, float, float, float, float)},
                        skiprows=1)

    #%% Tri des données
    volt = np.zeros(data.size)                 #Crée un tableau de zéros de la taille indiqué
    for i in range (0,data.size):
        volt[i] = data[i][0]
    counter = collections.Counter(volt)    # crée un dictionnaire avec pour clés les valeurs de volt et en valeur le nombre d'occurence
    keys_sorted= sorted(list(counter.keys()))    # liste des tensions par ordre croissant
#    val_sorted= sorted(list(y.values()))  # liste des nombres d'occurences de chaque tensions dans le document

    #%% Création et remplissage des listes des données
    tension = keys_sorted
    lpm = {}        # création d'une liste principale, qui contiendra les sous-listes
    tdh = {}
    courant = {}
    watts = {}
    efficacite = {}

    k = 0
    for V in tension:
        tdh_temp=[]
        courant_temp=[]
        lpm_temp=[]
        watts_temp=[]
        efficacite_temp=[]
        for j in range (0,counter[V]):
            tdh_temp.append(data[k][1])
            courant_temp.append(data[k][2])
            lpm_temp.append(data[k][3])
            watts_temp.append(data[k][4])
            efficacite_temp.append(data[k][5])
            k = k+1
        tdh[V]=tdh_temp
        courant[V]=courant_temp
        lpm[V]=lpm_temp
        watts[V]=watts_temp
        efficacite[V]=efficacite_temp

    return tension, lpm, tdh, courant, watts, efficacite

if __name__=="__main__":
#%% creation de la variable pompe
    pump1=Pump(path="fichiers_pompes/SCB_10_150_120_BL.txt",
                       model='SCB_10')
#    plt.figure()
#    for h in np.arange(0, 80, 10):
#        res=pump1.IVcurvedata(h)
#        plt.plot(res['V'], res['I'])
#    IV = pump1.IVcurvedata(28)

#    pump1.curves_coeffs()
#    print(pump1.opti_zone(0.1))


#%% exploration tension et puissance de démarrage :
#    print(pump1.startingVPI(15))
#    print(pump1.startingVPI(30))
#
#    tdhrange=np.arange(10,100)
#    Vmin=np.zeros(len(tdhrange))
#    Pmin=np.zeros(len(tdhrange))
#    Imin=np.zeros(len(tdhrange))
#
#    for i,td in enumerate(tdhrange):
#            try :
#                res=pump1.startingVPI(td)
#                Vmin[i]=res['V']
#                Pmin[i]=res['P']
#                Imin[i]=res['I']
#
#            except ValueError:
#                Vmin[i]=0
#    plt.figure(2)
#    plt.plot(tdhrange,Vmin,label='Vmin')
#    plt.plot(tdhrange,Pmin,label='Pmin')
#    plt.plot(tdhrange,Imin*100,label='Imin*100')
#    plt.legend()

#%% plot 3d test :
#    X=np.arange(0,10)
#    Xmat=np.matrix(X)
#    Xmat=np.transpose(Xmat)
#    Y=np.arange(0,10)
#    Ymat=np.matrix(Y)
#    Zmat=Xmat*Ymat
#    Zmat[5,5]=15
#    fig = plt.figure(3)
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_surface(Xmat, Ymat, Zmat)
#
#    tup=pump1.functVforIH()
#
#    fct=tup[0]
#    res=fct(X,Y)
#
#    fig = plt.figure(3)
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(resmat[:,2],resmat[:,1],resmat[:,0])


#%% selection du modèle de fonction pour curve_fit multi-dimensionnel

#    def funct1(inp,c1,h1):
#        x,y = inp[0],inp[1]
#        return c1*x + h1*y
#    def funct2(inp,c1,h1,t1):
#        x,y = inp[0],inp[1]
#        return c1*x + h1*y + t1*x*y
#    def funct3(inp,a,c1,h1,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + h1*y + t1*x*y
#    def funct41(inp,a,c1,c2,h1,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + c2*x**2 + h1*y + t1*x*y
#    def funct42(inp,a,c1,h1,h2,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + h1*y + h2*y**2 + t1*x*y
#    def funct5(inp,a,c1,c2,h1,h2,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + c2*x**2 + h1*y + h2*y**2 + t1*x*y
#    def funct6(inp,a,c1,c2,c3,h1,h2,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + c2*x**2 +c3*x**3 + h1*y + h2*y**2 + t1*x*y
#    def funct71(inp,a,c1,c2,c3,h1,h2,h3,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + c2*x**2 +c3*x**3 + h1*y + h2*y**2+ h3*y**3 + t1*x*y
#    def funct72(inp,a,c1,c2,c3,c4,h1,h2,t1):
#        x,y = inp[0],inp[1]
#        return a + c1*x + c2*x**2 +c3*x**3 + c4*x**4 + h1*y + h2*y**2+ t1*x*y
#
#    fct_list=[funct1,funct2,funct3,funct4,funct5,funct6]
#
#    res=pump1.functVforIH() # NE MARCHE PLUS
#    datax = [np.array(res['I']),np.array(res['H'])]
#    dataz = np.array(np.array(res['V']))
#
#    funct=funct1
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct2
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct3
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct41
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct42
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct5
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4],para[5])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct6
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4],para[5],
#                    para[6])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct71
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4],para[5],
#                    para[6],para[7])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)
#
#    funct=funct72
#    para, covmat = curve_fit(funct, datax, dataz)
#    stddev = np.sqrt(np.diag(covmat))
#    datacheck=funct(datax,para[0],para[1],para[2],para[3],para[4],para[5],
#                    para[6],para[7])
#    ectyp=np.sqrt(sum((dataz-datacheck)**2)/len(dataz))
#    print(funct,'\nectyp:',ectyp)


#%% plot of functVforIH

    vol=[]
    tdh=[]
    cur=[]
    for V in pump1.tension:
        for i, I in enumerate(pump1.courant[V]):
            vol.append(V)
            tdh.append(pump1.tdh[V][i])
            cur.append(I)
    f1, stddev, intervals=pump1.functVforIH()
    vol_check=[]
    for i,I in enumerate(cur):
        vol_check.append(f1(I, tdh[i], error_raising=False))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',title='Voltage as a function of'
                         ' current (A) and static head (m)')
    ax.scatter(cur,tdh,vol,label='from data')
    ax.scatter(cur,tdh,vol_check,label='from curve fitting')
    ax.set_xlabel('current')
    ax.set_ylabel('head')
    ax.set_zlabel('voltage V')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on V:', stddev)
    print('V for IH=(4,25): {0:.2f}'.format(f1(4, 25)))

#%% plot of functIforVH

    vol=[]
    tdh=[]
    cur=[]
    for V in pump1.tension:
        for i, I in enumerate(pump1.courant[V]):
            vol.append(V)
            tdh.append(pump1.tdh[V][i])
            cur.append(I)
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

    f2,stddev=pump1.functQforVH()
    vol=[]
    tdh=[]
    lpm=[]
    for V in pump1.tension:
        for i, Q in enumerate(pump1.lpm[V]):
            vol.append(V)
            tdh.append(pump1.tdh[V][i])
            lpm.append(Q)
#    # alternative way (faster ?): convert in array and flatten it
#    volflat=np.array(list(vol.values())).flat[:]
#    tdhflat=np.array(list(tdh.values())).flat[:]
#    lpmflat=np.array(list(lpm.values())).flat[:]

    lpm_check=[]
    for i,v in enumerate(vol):
        try:
            Q=f2(v,tdh[i])
        except (errors.VoltageError, errors.HeadError):
            Q=0
        lpm_check.append(Q)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',title='Q (lpm) as a function of'
                         ' voltage (V) and static head (m)')
    ax.scatter(vol,tdh,lpm,label='from data')
    ax.scatter(vol,tdh,lpm_check,label='from curve fitting')
    ax.set_xlabel('voltage')
    ax.set_ylabel('head')
    ax.set_zlabel('discharge Q')
    ax.legend(loc='lower left')
    plt.show()
    print('std dev on Q calculated from V:',stddev)
    print('Q for VH=(74,25): {0:.2f}'.format(f2(74,25)))


#%% plot of functQforPH

    f2, stddev = pump1.functQforPH()
    power = []
    tdh = []
    lpm = []
    for V in pump1.tension:
        for i, Q in enumerate(pump1.lpm[V]):
            power.append(pump1.watts[V][i])
            tdh.append(pump1.tdh[V][i])
            lpm.append(Q)
#    # alternative way (faster ?): convert in array and flatten it
#    powerflat=np.array(list(power.values())).flat[:]
#    tdhflat=np.array(list(tdh.values())).flat[:]
#    lpmflat=np.array(list(lpm.values())).flat[:]

    lpm_check=[]
    for i,po in enumerate(power):
        try:
            Q=f2(po, tdh[i])
        except (errors.PowerError, errors.HeadError):
            Q=0
        lpm_check.append(Q)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d',title='Q (lpm) as a function of'
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

#
