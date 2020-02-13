# -*- coding: utf-8 -*-
"""
@author: tylunel
"""

import pytest
import pvlib
import os
import inspect

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.mppt as mppt
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.sizing as siz

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def databases():
    # pump database
    pump_sunpump = pp.Pump(
        os.path.join(test_dir, "../data/pump_files/SCB_10_150_120_BL.txt"),
        price=1100,
        idname='SCB_10')
    pump_shurflo = pp.Pump(
        os.path.join(test_dir, "../data/pump_files/Shurflo_9325.txt"),
        price=700,
        idname='Shurflo_9325')
    pump_database = [pump_sunpump, pump_shurflo]

    # pv database
    pv_database = ['Canadian_Solar_Inc__CS5C_80M',
                   'Canadian_Solar_Inc__CS1U_430MS']

    # MPPT
    mppt1 = mppt.MPPT(efficiency=0.96,
                      price=1000)

    return {'pumps': pump_database,
            'pv_modules': pv_database,
            'mppt': mppt1}

    return None


def test_shrink_weather():
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather(weather_data)
    expected_shape = (48, 35)
    assert expected_shape == weather_shrunk.shape


def test_sizing_minimize_npv(databases):

    pump_database = databases['pumps']
    pv_database = databases['pv_modules']
    mppt1 = databases['mppt']

    # weather data
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather(weather_data)

    # rest of pumping system
    pipes = pn.PipeNetwork(h_stat=20, l_tot=100, diam=0.08,
                           material='plastic', optimism=True)
    consum = cs.Consumption(constant_flow=1,
                            length=len(weather_shrunk))

    pvps_fixture = pvps.PVPumpSystem(None, None,
                                     coupling='mppt',
                                     mppt=mppt1,
                                     consumption=consum,
                                     pipes=pipes)

    selection, _ = siz.sizing_minimize_npv(pv_database, pump_database,
                                           weather_shrunk, weather_metadata,
                                           pvps_fixture,
                                           llp_accepted=0.01,
                                           M_s_guess=1)
    assert ('Shurflo_9325' in selection.pump.values and
            'Canadian_Solar_Inc__CS1U_430MS' in selection.pv_module.values)


if __name__ == '__main__':
    # test all the tests in the module
    pytest.main(['test_sizing.py'])
