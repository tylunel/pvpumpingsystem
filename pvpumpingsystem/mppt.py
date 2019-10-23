# -*- coding: utf-8 -*-
"""

@author: Tanguy

Defines a MPPT
"""


class MPPT(object):
    """Class defining a MPPT made of :
            - voltage_rating: float
                Voltage rating
            - efficiency: numeric or array-like
                Mean efficiency if float.
                Efficiency according to power if array.

    """
    def __init__(self, voltage_rating, efficiency):
        self.voltage_rating = voltage_rating
        self.efficiency = efficiency

    def __repr__(self):
        return str(self.__dict__)
