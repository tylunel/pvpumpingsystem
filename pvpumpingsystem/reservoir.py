# -*- coding: utf-8 -*-
"""
Module for reservoir modeling.

@author: Tanguy Lunel
"""

import numpy as np


# TODO: replace water_volume by state of charge SOC

# TODO: change default value of size by 0. Put +inf here make the LLP false
# later. Need to adapt the rest so as it works well with 0

class Reservoir(object):
    """Class defining a water tank with its main characteristics.

    Attributes
    ----------
    size: float, default is 0
        Volume of reservoir [L]. '0' means no reservoir is used
    water_volume: float, default is 0
        Volume of water in the reservoir [L]. 0 = empty
    material: str, default is None
        Material of the reservoir
    price: float, default is 0
        Price of the reservoir and of the pipes [USD] (to be separated
        ultimately)
    """

    def __init__(self, size=0,
                 water_volume=0,
                 price=0,
                 material=None):
        self.size = size
        self.water_volume = water_volume
        self.price = price
        self.material = material

    def __repr__(self):
        return self.__dict__

    def change_water_volume(self, quantity, verbose=False):
        """Function for adding or removing water in the reservoir.

        Parameters
        ----------
        quantity: float
            amount of water too add or remove (in liters)

        Returns
        -------
        tuple
            (water_volume, extra (+) or lacking water(-))
        """

        self.water_volume = np.nansum([self.water_volume, quantity])

        if self.water_volume > self.size:
            extra_water = self.water_volume-self.size
            self.water_volume = self.size
            if verbose:
                print('Warning: The water volume exceeds size of reservoir')
            return (self.water_volume, extra_water)

        if self.water_volume < 0:
            lacking_water = self.water_volume
            self.water_volume = 0
            if verbose:
                print('Warning: The reservoir is empty, cannot ' +
                      'supply more water')
            return (0, lacking_water)

        return (self.water_volume, 0)
