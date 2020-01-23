# -*- coding: utf-8 -*-
"""
Defines Consumption class.

@author: Tanguy Lunel

"""
# TODO: Implement models where it is easily possible to make the
# repeated_flow change from a month to the other.

# TODO: Implement dynamic models, where the water demand can be function of
# the weather (irradiance (evapotranpiration) and precipitations)

import pandas as pd
import datetime


class Consumption(object):
    """
    The Consumption class defines a consumption schedule, typically through
    a year.

    Parameters
    ----------
    flow_rate: pd.DataFrame
        The consumption schedule in itself. It is given in liter per minute.

    constant_flow: numeric
        Parameter allowing to build consumption data with constant consumption
        through the flow_rates DataFrame.

    repeated_flow: 1D array-like
        Parameter allowing to build consumption data with a repeated
        consumption through the time.
    """

    def __init__(self, flow_rate=None, constant_flow=None, repeated_flow=None,
                 length=8760):
        if flow_rate is None:
            index = pd.date_range(datetime.datetime(2005, 1, 1, 0),
                                  periods=length,
                                  freq='H')
            self.flow_rate = pd.DataFrame(index=index, columns=('Qlpm',))
        else:
            self.flow_rate = flow_rate

        if constant_flow is not None:
            self.flow_rate = self.flow_rate.fillna(constant_flow)
        elif repeated_flow is not None:
            length = len(repeated_flow)
            for i, index in enumerate(self.flow_rate.index):
                self.flow_rate.loc[index] = repeated_flow[i % length]
        else:
            self.flow_rate = self.flow_rate.fillna(0)

    def __repr__(self):
        return str(self.flow_rate)


if __name__ == '__main__':
    consum = Consumption(repeated_flow=[0,0,0,0,0,0,15,20,15,10,10,10,
                                        10,10,10,30,30,30,0,0,0,0,0,0])
    print(consum)
