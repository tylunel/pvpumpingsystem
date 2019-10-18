# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:22:25 2019

@author: AP78430
"""
import unittest
import pytest
import numpy as np
import pvlib

import pvpumpingsystem.pump as pp
import pvpumpingsystem.pipenetwork as pn
import pvpumpingsystem.reservoir as rv
import pvpumpingsystem.consumption as cs
import pvpumpingsystem.pvpumpsystem as pvps


class PVPumpSystemTest(unittest.TestCase):
    """ Class for testing PVpumpSystem methods.
    """
    def setUp(self):

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
            'https://energyplus.net/weather-download/' +
            'north_and_central_america_wmo_region_4/USA/CO/' +
            'USA_CO_Denver.Intl.AP.725650_TMY3/' +
            'USA_CO_Denver.Intl.AP.725650_TMY3.epw',
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

        pump1 = pp.Pump(path="../pumps_files/SCB_10_150_120_BL.txt")
        pipes1 = pn.PipeNetwork(h_stat=10, l_tot=100, diam=0.08,
                                material='plastic', optimism=True)
        reserv1 = rv.Reservoir(1000000, 0)
        consum1 = cs.Consumption(constant_flow=1)

        self.pvps1 = pvps.PVPumpSystem(chain1, pump1, coupling='mppt',
                                       pipes=pipes1, consumption=consum1,
                                       reservoir=reserv1)

    def test_calc_flow(self):
        """Test the computing of flows.
        """
        self.pvps1.calc_flow(atol=0.1, stop=24)
        Q = self.pvps1.flow.Qlpm.values
        Q_expected = np.array([0., 0., 0., 0., 0., 0., 0., 0., 19.72, 53.20,
                               58.03, 60.46, 56.21, 58.64, 48.92, 41.79,
                               0., 0., 0., 0., 0., 0., 0., 0.])
        np.testing.assert_allclose(Q, Q_expected, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
#    pytest.main()  # test all the test modules in the directory