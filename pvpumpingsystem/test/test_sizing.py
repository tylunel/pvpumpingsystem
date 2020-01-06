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
import pvpumpingsystem.pvpumpsystem as pvps
import pvpumpingsystem.sizing as siz

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def pvps_set_up():
    return None


def test_shrink_pv_database():
    pv_database_shrunk = siz.shrink_pv_database('Canadian_Solar',
                                                nb_elt_kept=3)
    expected_list = ['Canadian_Solar_Inc__CS5C_80M',
                     'Canadian_Solar_Inc__CS6K_275P_AG',
                     'Canadian_Solar_Inc__CS1U_430MS']
    assert list(pv_database_shrunk) == expected_list


def test_shrink_weather():
    weather_path = os.path.join(
        test_dir,
        '../data/weather_files/CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw')
    weather_data, weather_metadata = pvlib.iotools.epw.read_epw(
            weather_path, coerce_year=2005)
    weather_shrunk = siz.shrink_weather(weather_data)
    expected_shape = (48, 35)
    assert expected_shape == weather_shrunk.shape


def test_sizing_maximise_flow():
    # pump databse
    pump_sunpump = pp.Pump(
        os.path.join(test_dir, "../data/pump_files/SCB_10_150_120_BL.txt"),
        model='SCB_10')
    pump_shurflo = pp.Pump(
        os.path.join(test_dir, "../data/pump_files/Shurflo_9325.txt"),
        model='Shurflo_9325')
    pump_database = [pump_sunpump, pump_shurflo]

    # pv database
    provider = "Canadian_Solar"
    nb_elt_kept = 3
    pv_database = siz.shrink_pv_database(provider, nb_elt_kept)

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
    pvps_fixture = pvps.PVPumpSystem(None, None, coupling='mppt',
                                     pipes=pipes)

    selection, _ = siz.sizing_maximize_flow(pv_database, pump_database,
                                            weather_shrunk, weather_metadata,
                                            pvps_fixture)
    assert 'Canadian_Solar_Inc__CS1U_430MS' in selection.pv_module.values


if __name__ == '__main__':
    # test all the tests in the module
    pytest.main(['test_sizing.py'])
