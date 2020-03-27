# -*- coding: utf-8 -*-
"""
Module for defining MPPT characteristic. Still at embryonic stage.
For controllers corresponding to the sunpump specs (i.e. with output voltage
up to 120V), check controller_files

@author: Tanguy Lunel

Defines a MPPT
"""

# TODO: develop input_voltage_range attribute to ensure that it fits with
# the ouput voltage from PV array. (Will allow to reduce the computation time
# as well as pvpumpingsystem will not look for too big number of modules
# in the PVPS anymore)

# TODO: develop output_voltage_available attribute to ensure that it fits with
# the input voltage of the pump

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

        idname: str,
            Name of the MPPT

        output_voltage_available: list
            Correspond to the list of keys of 'input_voltage_range'

        input_voltage_range: dict
            Input voltage range given as value (tuple) for each output voltage
            available given as key (float).
    """

    def __init__(self,
                 efficiency=0.96,
                 price=float('nan'),
                 idname='default',
                 output_voltage_available=None,
                 input_voltage_range=None):

        self.idname = idname
        self.efficiency = efficiency
        self.price = price
        self.output_voltage_available = output_voltage_available
        self.input_voltage_range = input_voltage_range

    def __repr__(self):
        return str(self.__dict__)
