# -*- coding: utf-8 -*-
"""
@author: tylunel
"""
# flake8: noqa

import pytest
import pvlib
import os
import inspect

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.reservoir as res
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.sizing as siz
# import pvpumpingsystem.pvgeneration as pvgen

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def databases():
    # pump database
    pump_sunpump = pp.Pump(name='SCB_10_150_120_BL')
    pump_shurflo = pp.Pump(name='Shurflo_9325', price=700)
    pump_database = [pump_shurflo, pump_sunpump]

    # pv database
    pv_database = ['Canadian_Solar_Inc__CS1U_430MS',
                   'Canadian_Solar_Inc__CS5C_80M'
                   ]

    # MPPT
    mppt1 = mppt.MPPT(efficiency=0.96,
                      price=1000)

    return {'pumps': pump_database,
            'pv_modules': pv_database,
            'mppt': mppt1}

    return None


def test_shrink_weather_representative():
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather_representative(weather_data)
    expected_shape = (48, 35)
    assert expected_shape == weather_shrunk.shape


def test_shrink_weather_worst_month():
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather_worst_month(weather_data)
    worst_month = int(weather_shrunk.month.drop_duplicates().iloc[0])
    assert worst_month == 12  # 12 = december


@pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
def test_sizing_minimize_npv_mppt(databases):
    """
    Goes through following functions:
        sizing.subset_respecting_llp_mppt()
        sizing.sizing_minimize_npv()

    Note that the OptimizeWarning ignored through pytest.mark above is a
    warning that comes from the relatively small amount of data on shurflo
    pump. It warns that scipy and pvpumpingsystem could not provide
    statistical figures on the quality of the model fitting for this pump.
    """

    pump_database = databases['pumps']
    pv_database = databases['pv_modules']
    mppt1 = databases['mppt']
    reservoir1 = res.Reservoir(size=5000)

    # weather data
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather_representative(weather_data)

    # rest of pumping system
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)
    consum = cs.Consumption(constant_flow=1)

    pvps_fixture = pvps.PVPumpSystem(None,
                                      None,
                                      coupling='mppt',
                                      mppt=mppt1,
                                      consumption=consum,
                                      reservoir=reservoir1,
                                      pipes=pipes)

    selection, _ = siz.sizing_minimize_npv(
            pv_database, pump_database,
            weather_shrunk, weather_metadata,
            pvps_fixture,
            llp_accepted=0.01,
            M_s_guess=1)
    assert ('shurflo_9325' in selection.pump.values and
            'Canadian_Solar_Inc__CS5C_80M' in selection.pv_module.values)


@pytest.mark.filterwarnings("ignore::scipy.optimize.OptimizeWarning")
def test_sizing_minimize_npv_direct(databases):
    """
    Goes through following functions:
        sizing.subset_respecting_llp_direct()
        sizing.sizing_minimize_npv()

    Note that the OptimizeWarning ignored through pytest.mark above is a
    warning that comes from the relatively small amount of data on shurflo
    pump. It warns that scipy and pvpumpingsystem could not provide
    statistical figures on the quality of the model fitting for this pump.
    """

    pump_database = databases['pumps']
    pv_database = databases['pv_modules']
    mppt1 = databases['mppt']
    reservoir1 = res.Reservoir(size=5000)

    # weather data
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather_representative(weather_data)

    # rest of pumping system
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)
    consum = cs.Consumption(constant_flow=1)

    pvps_fixture = pvps.PVPumpSystem(None,
                                     None,
                                     coupling='direct',
                                     mppt=mppt1,
                                     consumption=consum,
                                     reservoir=reservoir1,
                                     pipes=pipes)
    with pytest.warns(UserWarning) as record:
        selection, _ = siz.sizing_minimize_npv(
                pv_database, pump_database,
                weather_shrunk, weather_metadata,
                pvps_fixture,
                llp_accepted=0.01,
                M_s_guess=1)
    # Check that the warning is the one expected
    assert "do not match" in record[0].message.args[0]

    assert ('shurflo_9325' in selection.pump.values and
            'Canadian_Solar_Inc__CS5C_80M' in selection.pv_module.values)


if __name__ == '__main__':
    # test all the tests in the module
    pytest.main(['test_sizing.py'])
