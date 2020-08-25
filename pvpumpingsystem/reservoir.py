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
    """Class defining a reservoir.

    Attributes
    ----------
    size: float, default is 0, i.e. like if there was no reservoir
        Volume of reservoir [L]
    water_volume: float, default is 0
        Volume of water in the reservoir [L]. 0 = empty
    material: str, default is None
        Material of the reservoir
    price: float, default is 0
        Price of the reservoir [USD]
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
        quantity: numeric
            amount of water too add or remove (in liters)

        Returns
        -------
        tuple with:
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
