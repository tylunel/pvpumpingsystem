# -*- coding: utf-8 -*-
"""
Module aimed at gathering the functions for financial analysis.
Ultimately, it should inlude:
    initial investment assessment (capex), net present value, loan repayment

@author: tylunel


In Numpy, Some Simple financial functions already exist:

fv(rate, nper, pmt, pv[, when])
    Compute the future value.
pv(rate, nper, pmt[, fv, when])
    Compute the present value.
npv(rate, values)
    Returns the NPV (Net Present Value) of a cash flow series.
pmt(rate, nper, pv[, fv, when])
    Compute the payment against loan principal plus interest.
ppmt(rate, per, nper, pv[, fv, when])
    Compute the payment against loan principal.
ipmt(rate, per, nper, pv[, fv, when])
    Compute the interest portion of a payment.
irr(values)
    Return the Internal Rate of Return (IRR).
mirr(values, finance_rate, reinvest_rate)
    Modified internal rate of return.
nper(rate, pmt, pv[, fv, when])
    Compute the number of periodic payments.
rate(nper, pmt, pv, fv[, when, guess, tol, ...])
    Compute the rate of interest per period.
"""
# TODO: Check for terminology. CAPEX may not be the equivalent of initial
# investment

import numpy as np


def initial_investment(pvps, labour_price_coefficient=None):
    """
    Function computing the initial investment cost.

    Parameters
    ----------
    pvps: pvpumpingsystem.PVPumpSystem,
        The Photovoltaic pumping system whose cost is to be analyzed.

    labour_price_coefficient: float, default value is taken from pvps object
            Ratio of the price of labour and secondary costs on initial
            investment.

    """
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

    return result


def net_present_value(pvps, opex, discount_rate,
                      lifespan_pv=30, lifespan_mppt=10, lifespan_pump=10):
    """
    Function computing the net present value of a PVPS

    Parameters
    ----------

    [To fill]

    lifespan_pv: float
        Lifespan of the photovoltaic modules. It is also considered as the
        lifespan of the whole system.
    """
    # creation of list of opex cost with length of lifespan_pv
    cashflow_list = [opex]
    cashflow_list *= lifespan_pv

    cashflow_list[0] += (pvps.reservoir.price
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

    # calculation of net present value
    npv = np.npv(discount_rate, cashflow_list)

    return npv


if __name__ == '__main__':
    pass

