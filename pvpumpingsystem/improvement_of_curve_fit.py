# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:47:23 2019

@author: AP78430
"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

#    pump1 = Pump(lpm={12: [212, 204, 197, 189, 186, 178, 174, 166, 163, 155,
#                           136],
#                      24: [443, 432, 413, 401, 390, 382, 375, 371, 352, 345,
#                           310]},
#                 tdh={12: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
#                           54.9, 61.0, 70.1],
#                      24: [6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
#                           54.9, 61.0, 70.1]},
#                 current={24: [1.5, 1.7, 2.1, 2.4, 2.6, 2.8, 3.1, 3.3, 3.6,
#                               3.8, 4.1],
#                          12: [1.2, 1.5, 1.8, 2.0, 2.1, 2.4, 2.7, 3.0, 3.3,
#                               3.4, 3.9]}, model='Shurflo_9325')

def func_model4(x, a, b, c, d, e):
    return a + b*x + c*x**2 + d*x**3 + e*x**4


def func_model3(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3


def func_model2(x, a, b, c):
    return a + b*x + c*x**2


bounds4 = ([0, -np.inf, -np.inf, -np.inf, -np.inf],
           [np.inf, 0, 0, 0, 0])
bounds3 = ([0, -np.inf, -np.inf, -np.inf],
           [np.inf, 0, 0, 0])
bounds2 = ([0, -np.inf, -np.inf],
           [np.inf, 0, 0])

lpm_x = np.array([212, 204, 197, 189, 186, 178, 174, 166, 163, 155, 136],
                 dtype=float)
lpm_x_extended = np.array([250, 212, 204, 197, 189, 186, 178,
                           174, 163, 155, 140, 90, 0], dtype=float)
tdh_expected = np.array([6.1, 12.2, 18.3, 24.4, 30.5, 36.6, 42.7, 48.8,
                         54.9, 61.0, 70.1])

coeff4, matcov4 = opt.curve_fit(func_model4,
                                lpm_x,
                                tdh_expected,
                                p0=[10, -1, -1, 0, 0],
                                bounds=bounds4)
tdh4 = func_model4(lpm_x, *coeff4)
tdh4_extended = func_model4(lpm_x_extended, *coeff4)

coeff3, matcov = opt.curve_fit(func_model3,
                               lpm_x,
                               tdh_expected,
                               p0=[10, -1, -1, 0],
                               bounds=bounds3)
tdh3 = func_model3(lpm_x, *coeff3)
tdh3_extended = func_model3(lpm_x_extended, *coeff3)

coeff2, matcov = opt.curve_fit(func_model2,
                               lpm_x,
                               tdh_expected,
                               p0=[10, -1, -1],
                               bounds=bounds2)
tdh2 = func_model2(lpm_x, *coeff2)
tdh2_extended = func_model2(lpm_x_extended, *coeff2)

res2 = tdh2 - tdh_expected
stddev2 = np.sqrt(sum(res2**2))
print('stddev2: ', stddev2)

res3 = tdh3 - tdh_expected
stddev3 = np.sqrt(sum(res3**2))
print('stddev3: ', stddev3)

res4 = tdh4 - tdh_expected
stddev4 = np.sqrt(sum(res4**2))
print('stddev4: ', stddev4)

plt.figure()
plt.plot(tdh_expected, lpm_x, label='true data')
plt.plot(tdh4_extended, lpm_x_extended, label='order 4')
plt.plot(tdh3_extended, lpm_x_extended, label='order 3')
plt.plot(tdh2_extended, lpm_x_extended, label='order 2')
plt.legend()