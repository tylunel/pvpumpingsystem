# -*- coding: utf-8 -*-
"""
@author: Tanguy Lunel
"""

import pytest
import numpy as np
import pvlib
import os
import inspect

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.pvgeneration as pvgen

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def pvps_set_up():

#    CECMOD = pvlib.pvsystem.retrieve_sam('cecmod')
#
#    glass_params = {'K': 4, 'L': 0.002, 'n': 1.526}
#    pvsys1 = pvlib.pvsystem.PVSystem(
#            surface_tilt=45, surface_azimuth=180,
#            albedo=0, surface_type=None,
#            module=CECMOD.Kyocera_Solar_KU270_6MCA,
#            module_parameters={**dict(CECMOD.Kyocera_Solar_KU270_6MCA),
#                               **glass_params},
#            modules_per_string=2, strings_per_inverter=2,
#            inverter=None, inverter_parameters={'pdc0': 700},
#            racking_model='open_rack',
#            losses_parameters=None, name=None
#            )
    pvgen1 = pvgen.PVGeneration(
            # Weather data
            path_weather_data=('../data/weather_files/CAN_PQ_Montreal.Intl.'
                               'AP.716270_CWEC_truncated.epw'),  # to adapt:

            # PV array parameters
            pv_module_name='kyocera solar KU270 6MCA',
            price_per_module=200,  # in US dollars
            surface_tilt=45,  # 0 = horizontal, 90 = vertical
            surface_azimuth=180,  # 180 = South, 90 = East
            albedo=0,  # between 0 and 1
            modules_per_string=2,
            strings_in_parallel=2,
            # PV module glazing parameters (not always given in specs)
            glass_params={'K': 4,  # extinction coefficient [1/m]
                          'L': 0.002,  # thickness [m]
                          'n': 1.526},  # refractive index
            racking_model='open_rack',  # or'close_mount' or 'insulated_back'

            # Models used (check pvlib.modelchain for all available models)
            orientation_strategy=None,  # or 'flat' or 'south_at_latitude_tilt'
            clearsky_model='ineichen',
            transposition_model='haydavies',
            solar_position_method='nrel_numpy',
            airmass_model='kastenyoung1989',
            dc_model='desoto',  # 'desoto' or 'cec' only
            ac_model='pvwatts',
            aoi_model='physical',
            spectral_model='first_solar',
            temperature_model='sapm',
            losses_model='pvwatts'
            )
    pvgen1.run_model()

#    weather_testfile = os.path.join(
#        test_dir,
#        '../data/weather_files/'
#        'CAN_PQ_Montreal.Intl.AP.716270_CWEC_truncated.epw')
#    weatherdata1, metadata1 = pvlib.iotools.epw.read_epw(weather_testfile,
#                                                         coerce_year=2005)
#    locat1 = pvlib.location.Location.from_epw(metadata1)

#    chain1 = pvlib.modelchain.ModelChain(
#             system=pvsys1, location=locat1,
#             orientation_strategy=None,
#             clearsky_model='ineichen',
#             transposition_model='haydavies',
#             solar_position_method='nrel_numpy',
#             airmass_model='kastenyoung1989',
#             dc_model='desoto', ac_model='pvwatts', aoi_model='physical',
#             spectral_model='first_solar', temperature_model='sapm',
#             losses_model='pvwatts', name=None)

#    chain1.run_model(times=weatherdata1.index, weather=weatherdata1)

    pump_testfile = os.path.join(test_dir,
                                 '../data/pump_files/SCB_10_150_120_BL.txt')
    pump1 = pp.Pump(path=pump_testfile,
                    modeling_method='arab')
    pipes1 = pn.PipeNetwork(h_stat=10, l_tot=100, diam=0.08,
                            material='plastic', optimism=True)
    reserv1 = rv.Reservoir(1000000, 0)
    consum1 = cs.Consumption(constant_flow=1)

    pvps1 = pvps.PVPumpSystem(pvgen1,
                              pump1,
                              coupling='mppt',
                              pipes=pipes1,
                              consumption=consum1,
                              reservoir=reserv1)
    return pvps1


def test_calc_flow_mppt(pvps_set_up):
    """Test the computing of flows in the case coupled with mppt.
    """
    pvps_set_up.coupling = 'mppt'
    pvps_set_up.calc_flow(atol=0.1, stop=24)
    Q = pvps_set_up.flow.Qlpm.values
    Q_expected = np.array([0., 0., 0., 0., 0., 0., 0., 0.,
                           32.77, 52.11, 58.80, 61.24,
                           44.18, 41.47, 34.35, 0.,
                           0., 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(Q, Q_expected, rtol=1)


def test_calc_flow_direct(pvps_set_up):
    """Test the computing of flows when pump and pv are directly coupled.
    """
    pvps_set_up.coupling = 'direct'
    pvps_set_up.calc_flow(atol=0.1, stop=24)
    Q = pvps_set_up.flow.Qlpm.values
    Q_expected = np.array([0., 0., 0., 0., 0., 0., 0., 0.,
                           26.7413, 28.6602, 29.1504, 29.5732,
                           28.6143, 28.3941, 27.3603, np.nan,
                           np.nan, 0., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(Q, Q_expected, rtol=1)


def test_functioning_point_noiteration(pvps_set_up):
    """Test the ability of code to find the functioning point between
    pump and pv array when directly-coupled.
    """
    df_iv = pvps_set_up.functioning_point_noiteration()
    arr_iv = np.array(df_iv[11:19], dtype=float)
    arr_iv_expected = np.array([[3.1552, 75.1672],
                                [3.0930, 74.2068],
                                [3.0753, 73.9332],
                                [2.9869, 72.5665],
                                [np.nan, np.nan],
                                [np.nan, np.nan],
                                [0., 0.],
                                [0., 0.]])
    np.testing.assert_allclose(arr_iv, arr_iv_expected, rtol=0.1)


if __name__ == '__main__':
    # runs all the tests in this module
    pytest.main(['test_pvpumpsystem.py'])
