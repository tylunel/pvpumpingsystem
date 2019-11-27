# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:22:25 2019

@author: AP78430
"""

import pytest
import numpy as np
import pvlib
import pandas as pd

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps


@pytest.fixture
def pvps_set_up():

    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')

    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
    pvsys1 = pvlib.pvsystem.PVSystem(
            surface_tilt=45, surface_azimuth=180,
            albedo=0, surface_type=None,
            module=CECMOD.Kyocera_Solar_KU270_6MCA,
            module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
                               **glass_params},
            modules_per_string=2, strings_per_inverter=2,
            inverter=None, inverter_parameters={'pdc0': 700},
            racking_model='open_rack_cell_glassback',
            losses_parameters=None, name=None
            )
    weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(
        '../weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC_truncated.epw',
        coerce_year=2005)
    locat1 = pvlib.location.Location.from_epw(metadata1)

    chain1 = pvlib.modelchain.ModelChain(
             system=pvsys1, location=locat1,
             orientation_strategy=None,
             clearsky_model='ineichen',
             transposition_model='haydavies',
             solar_position_method='nrel_numpy',
             airmass_model='kastenyoung1989',
             dc_model='desoto', ac_model='pvwatts', aoi_model='physical',
             spectral_model='first_solar', temp_model='sapm',
             losses_model='pvwatts', name=None)

    chain1.run_model(times=weatherdata1.index, weather=weatherdata1)

    pump1 = pp.Pump(path="../pumps_files/SCB_10_150_120_BL.txt",
                    modeling_method='arab')
    pipes1 = pn.PipeNetwork(h_stat=10, l_tot=100, diam=0.08,
                            material='plastic', optimism=True)
    reserv1 = rv.Reservoir(1000000, 0)
    consum1 = cs.Consumption(constant_flow=1)

    pvps1 = pvps.PVPumpSystem(chain1, pump1, coupling='mppt',
                              pipes=pipes1, consumption=consum1,
                              reservoir=reserv1)
    return pvps1


def test_calc_flow(pvps_set_up):
    """Test the computing of flows in the case coupled with mppt.
    """
    pvps_set_up.calc_flow(atol=0.1, stop=24)
    Q = pvps_set_up.flow.Qlpm.values
    Q_expected = np.array([0., 0., 0., 0., 0., 0., 0., 0.,
                           32.67, 51.86, 58.54, 61.04,
                           44.00, 41.31, 34.25,  0.,
                           0., 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(Q, Q_expected, rtol=1e-3)


def test_functioning_point_noiteration(pvps_set_up):
    """Test the ability of code to find the functioning point between
    pump and pv array when directly-coupled.
    """
    df_iv = pvps_set_up.functioning_point_noiteration()
    arr_iv = np.array(df_iv[11:19], dtype=float)
    arr_iv_expected = np.array([[2.12604482, 75.76427699],
                                [2.12241983, 75.18362267],
                                [2.12135466, 75.01300389],
                                [2.11539094, 74.05773065],
                                [np.nan, np.nan],
                                [np.nan, np.nan],
                                [0., 0.],
                                [0., 0.]])
    np.testing.assert_allclose(arr_iv, arr_iv_expected, rtol=0.1)


if __name__ == '__main__':
    # test all the tests in the module
    pytest.main(['test_pvpumpsystem.py'])
