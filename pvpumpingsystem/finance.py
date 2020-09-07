# -*- coding: utf-8 -*-
"""
Module aimed at gathering the functions for financial analysis.
Ultimately, it should include:
    initial investment assessment, net present value, loan repayment

@author: tylunel

Before adding new function, check numpy_finance for equivalent financial
functions.

"""

import numpy as np
import numpy_financial as npf


def initial_investment(pvps, labour_price_coefficient=0.2, **kwargs):
    """
    Function computing the initial investment cost.

    Parameters
    ----------
    pvps: pvpumpingsystem.PVPumpSystem,
        The Photovoltaic pumping system whose cost is to be analyzed.

    labour_price_coefficient: float, default is 0.2
            Ratio of the price of labour and secondary costs (wires,
            racks (can be expensive!), transport of materials, etc) on initial
            investment. It is considered at 0.2 in Gualteros (2017),
            but is more around 0.40 in Tarpuy(Peru) case.

    Returns
    -------
    float
        Initial investment for the whole pumping system.
    """
    try:
        pv_modules_price = (pvps.pvgeneration.system.modules_per_string
                            * pvps.pvgeneration.system.strings_per_inverter
                            * pvps.pvgeneration.price_per_module)

        if pvps.coupling == 'mppt' or pvps.coupling == 'mppt_no_iteration':
            result = (pvps.motorpump.price
                      + pv_modules_price
                      + pvps.reservoir.price
                      + pvps.mppt.price) \
                      * (1 + labour_price_coefficient)
        elif pvps.coupling == 'direct':
            result = (pvps.motorpump.price
                      + pv_modules_price
                      + pvps.reservoir.price) \
                      * (1 + labour_price_coefficient)
    except AttributeError as e:
        if "has no attribute 'price'" in e.args[0]:  # check the message
            return np.nan
        else:
            raise e

    return result


# TODO: put lifespan of each object in the correponding object itself
def net_present_value(pvps, discount_rate=0.02,
                      labour_price_coefficient=0.2, opex=0,
                      lifespan_pv=30, lifespan_mppt=14, lifespan_pump=12):
    """
    Function computing the net present value of a PVPS

    Parameters
    ----------
    pvps: pvpumpingsystem.PVPumpSystem,
        The photovoltaic pumping system to evaluate.

    discount_rate: float, default is 0.02
        Dicsount rate.

    labour_price_coefficient: float, default is 0.2
        Ratio of the price of labour on capital cost.
            Example: If labour_price_coefficient = 0.2 (20%), it is
            considered that a 1000 USD PV array will cost 200 USD more
            to be installed on site.

    opex: float, default is 0
        Yearly operational expenditure of the pvps.

    lifespan_pv: float, default is 30
        Lifespan of the photovoltaic modules in years. It is also
        considered as the lifespan of the whole system.

    lifespan_mppt: float, default is 14
        Lifespan of the mppt in years.

    lifespan_pump: float, default is 12
        Lifespan of the pump in years.

    Returns
    -------
    float:
        The net present value of the PVPS
    """
    # creation of list of opex cost with length of lifespan_pv
    cashflow_list = [opex]
    cashflow_list *= lifespan_pv

    try:
        pv_modules_price = (pvps.pvgeneration.system.modules_per_string
                            * pvps.pvgeneration.system.strings_per_inverter
                            * pvps.pvgeneration.price_per_module)

        cashflow_list[0] += ((pvps.reservoir.price + pv_modules_price)
                             * (1 + labour_price_coefficient))

        # add cost of pump on each year it is expected to be replaced/bought
        for i in range(int(np.ceil(lifespan_pv/lifespan_pump))):
            year = i * lifespan_pump
            cashflow_list[year] += (pvps.motorpump.price
                                    * (1 + labour_price_coefficient))

        if pvps.coupling == 'mppt':
            for i in range(int(np.ceil(lifespan_pv/lifespan_mppt))):
                year = i * lifespan_mppt
                cashflow_list[year] += (pvps.mppt.price
                                        * (1 + labour_price_coefficient))

    except AttributeError as e:
        if "has no attribute 'price'" in e.args[0]:  # check the message
            return np.nan
        else:
            raise e

    # calculation of net present value
    npv = npf.npv(discount_rate, cashflow_list)

    return npv
