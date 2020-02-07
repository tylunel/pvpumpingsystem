# -*- coding: utf-8 -*-
"""
Module for defining MPPT characteristic. Still at embryonic stage.

@author: Tanguy Lunel

Defines a MPPT
"""

# TODO: develop voltage_rating attribute to insure that it fits with
# the input voltage of the pump, and with the ouput voltage from PV array.

# TODO: add way to have an efficiency depending on the input power


class MPPT(object):
    """
    Class defining a MPPT.

    Attributes
    ----------
        efficiency: float
            Mean efficiency if float.
            Efficiency according to power if array.

        price: float,
            Price of the MPPT

        voltage_rating: float
            Voltage rating
    """

    def __init__(self,
                 efficiency=0.96,
                 price=float('nan'),
                 voltage_rating=None):

        self.efficiency = efficiency
        self.price = price
        self.voltage_rating = voltage_rating

    def __repr__(self):
        return str(self.__dict__)
