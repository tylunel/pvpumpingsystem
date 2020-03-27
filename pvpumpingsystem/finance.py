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


def initial_investment(pvps, labour_price_coefficient=None):
    """
    Function computing the initial investment cost.

    Parameters
    ----------
    pvps: pvpumpingsystem.PVPumpSystem,
        The Photovoltaic pumping system whose cost is to be analyzed.

    labour_price_coefficient: float, default is None
            Ratio of the price of labour and secondary costs on initial
            investment.
            None value will look for value of pvps object. In pvps, default
            is 1.2.

    Returns
    -------
    float: Initial investment for the whole pumping system.
    """
    try:
        pv_modules_price = (pvps.pvgeneration.system.modules_per_string
                            * pvps.pvgeneration.system.strings_per_inverter
                            * pvps.pvgeneration.price_per_module)

        if labour_price_coefficient is None:
            labour_price_coefficient = pvps.labour_price_coefficient

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
def net_present_value(pvps, opex, discount_rate=0.02,
                      lifespan_pv=30, lifespan_mppt=10, lifespan_pump=10):
    """
    Function computing the net present value of a PVPS

    Parameters
    ----------
    pvps: pvpumpingsystem.PVPumpSystem,
        The photovoltaic pumping system to evaluate.

    opex: float,
        Yearly operational expenditure of the pvps.

    discount_rate: float, default is 0.02
        Dicsount rate.

    lifespan_pv: float
        Lifespan of the photovoltaic modules. It is also considered as the
        lifespan of the whole system.

    lifespan_mppt: float
        Lifespan of the mppt

    lifespan_pump: float
        Lifespan of the pump

    Returns
    -------
    float: The net present value of the PVPS
    """
    # creation of list of opex cost with length of lifespan_pv
    cashflow_list = [opex]
    cashflow_list *= lifespan_pv

    try:
        pv_modules_price = (pvps.pvgeneration.system.modules_per_string
                            * pvps.pvgeneration.system.strings_per_inverter
                            * pvps.pvgeneration.price_per_module)

        cashflow_list[0] += ((pvps.reservoir.price + pv_modules_price)
                             * (1 + pvps.labour_price_coefficient))

        # add cost of pump on each year it is expected to be replaced/bought
        for i in range(int(np.ceil(lifespan_pv/lifespan_pump))):
            year = i * lifespan_pump
            cashflow_list[year] += (pvps.motorpump.price
                                    * (1 + pvps.labour_price_coefficient))

        if pvps.coupling == 'mppt':
            for i in range(int(np.ceil(lifespan_pv/lifespan_mppt))):
                year = i * lifespan_mppt
                cashflow_list[year] += (pvps.mppt.price
                                        * (1 + pvps.labour_price_coefficient))

    except AttributeError as e:
        if "has no attribute 'price'" in e.args[0]:  # check the message
            return np.nan
        else:
            raise e

    # calculation of net present value
    npv = npf.npv(discount_rate, cashflow_list)

    return npv


if __name__ == '__main__':
    pass
