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
        The consumption schedule in itself [L/min]

    constant_flow: numeric
        Parameter allowing to build consumption data with constant consumption
        through the flow_rates DataFrame.

    repeated_flow: 1D array-like
        Parameter allowing to build consumption data with a repeated
        consumption through the time.
    """

    def __init__(self, flow_rate=None, constant_flow=None, repeated_flow=None,
                 length=8760, year=2005, safety_factor=1):
        if flow_rate is None:
            index = pd.date_range(datetime.datetime(year, 1, 1, 0),
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

        self.flow_rate *= safety_factor
        self.safety_factor = safety_factor

    def __repr__(self):
        return str(self.flow_rate)


def adapt_to_flow_pumped(Q_consumption, Q_pumped):
    """
    Method for shrinking the consumption flow_rate attribute
    at the same size than the corresponding pumped flow rate data.

    Parameters
    ----------
    Q_consumption: pd.DataFrame,
        Dataframe with pandas timestamp as index. Typically comes from
        PVPumpSystem.consumption.flow_rate

    Q_pumped: pd.DataFrame,
        Dataframe with pandas timestamp as index. Typically comes from
        PVPumpSystem.flow.Qlpm

    Return
    ------
    pandas.DataFrame
        Consumption data modified.
    """

    timezone = Q_pumped.index.tz

    # test if Q_consumption.index is timezone naive:
    if Q_consumption.index.tzinfo is None or \
            Q_consumption.index.tzinfo.utcoffset(Q_consumption.index) \
            is None:
        Q_consumption.index = Q_consumption.index.tz_localize(timezone)

    # get intersection of index
    intersect = Q_pumped.index.intersection(Q_consumption.index)
    if intersect.empty is True:
        raise ValueError('The consumption data and the water pumped data '
                         'are not relying on the same dates.')
    Q_consumption = Q_consumption.loc[intersect]

    return Q_consumption
