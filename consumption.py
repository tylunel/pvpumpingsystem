# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:47:29 2019

@author: AP78430

Class Consumption
"""

import pandas as pd
import datetime


class Consumption(object):
    """
    The Consumption class defines a consumption schedule, typically through
    a year.

    Parameters
    ----------
    flow: pd.DataFrame
        The consumption schedule in itself. It is given in liter per minute.

    constant_flow: numeric
        Parameter allowing to build consumption data with constant consumption
        through the flows DataFrame.

    constant_flow: 1D array-like
        Parameter allowing to build consumption data with a repeated
        consumption through the time.
    """

    def __init__(self, flow=None, constant_flow=None, repeated_flow=None):
        if flow is None:
            index = pd.date_range(datetime.datetime(2005, 1, 1, 0),
                                  datetime.datetime(2005, 12, 31, 23),
                                  freq='H')
            self.flow = pd.DataFrame(index=index, columns=('consumption',))
        else:
            self.flow = flow

        if constant_flow is not None:
            self.flow = self.flow.fillna(constant_flow)
        elif repeated_flow is not None:
            length = len(repeated_flow)
            for i, index in enumerate(self.flow.index):
                self.flow.loc[index] = repeated_flow[i % length]
        else:
            self.flow = self.flow.fillna(0)

    def __repr__(self):
        return str(self.flow)


if __name__ == '__main__':
    consum = Consumption(repeated_flow=[0,0,0,0,0,0,15,20,15,10,10,10,
                                        10,10,10,30,30,30,0,0,0,0,0,0])
    print(consum)
