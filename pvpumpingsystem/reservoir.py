# -*- coding: utf-8 -*-
"""
Module for reservoir modeling.

@author: Tanguy Lunel
"""

import numpy as np
import warnings


# TODO: replace water_volume by state of charge SOC

class Reservoir(object):
    """Class defining a reservoir.

        Attributes:
        --------
        size: float, default is inf, i.e. like if there was no reservoir
            Volume of reservoir [L]
        water_volume: float, default is 0
            Volume of water in the reservoir [L]. 0 = empty
        material: str, default is None
            Material of the reservoir
    """

    def __init__(self, size=float("inf"), water_volume=0, material=None):
        self.size = size
        self.water_volume = water_volume
        self.material = material

    def __repr__(self):
        return self.__dict__

    def change_water_volume(self, quantity, verbose=False):
        """Function for adding or removing water in the reservoir.

        Parameters:
        -----------
        quantity: numeric
            amount of water too add or remove (in liters)

        returns:
            - tuple with:
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

#    def reservoir_level(self,Q=None,consumption=None):
#        """Function giving the volume of water contained in the reservoir
#        at anytime.
#
#        parameters:
#            - Q (numeric or array-like): volume of water added(in m3)
#            - consumption (numeric or array-like): volume of water removed
#                (in m3)
#
#        returns:
#            volume of water in the reservoir (numeric or array-like) (in m3)
#
#        """
#        warnings.simplefilter('error')
#        warnings.warn("\nFunction not really useful anymore, use "
#                      "change_water_volume instead. Or switch 'error' by "
#                      "'default' in the first line of the function",
#                      DeprecationWarning)
#
#        if Q is None and consumption is None:
#            return self.water_volume
#
#        Q=np.array(Q)
#        consumption=np.array(consumption)
#
#        if Q.size != consumption.size:
#            raise ValueError('Q and consumption must have same size')
#
#        vol_init=self.water_volume
#        tab_volume=[] # the volume stored here is the volume at the end of hour
#
#        if Q is not None and consumption is not None:
#            for i, flowin in Q:
#                vol_new=vol_init + flowin - consumption[i]
#                tab_volume.append(vol_new)


if __name__ == '__main__':
    reserv1 = Reservoir(3, 1)
    q = [0.5,0.2,2,-1,4,-8]
    tabres = []
    for qi in q:
        tabres.append(reserv1.change_water_volume(qi))

    print(tabres)
    arrres = np.array(tabres)
    print(arrres)
