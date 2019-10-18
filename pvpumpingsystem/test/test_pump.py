# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:21:16 2019

@author: Tanguy
"""

import unittest
import numpy as np

import pvpumpingsystem.pump as pp
from pvpumpingsystem import errors


class PumpTest(unittest.TestCase):
    """ Class for testing Pump methods.
    """
    def setUp(self):
        self.pump1 = pp.Pump(path="../pumps_files/SCB_10_150_120_BL.txt",
                             model='SCB_10')

    def test_functVforIH(self):
        """Test whether the standard deviation between real data and
        simulated data is acceptable (<3%), if the output functV works
        well, and if this function is able to correctly raise errors.
        """
        # check standard deviation
        functV, stddev, intervals = self.pump1.functVforIH()
        np.testing.assert_array_less(stddev, 3)

        # check computing through functV
        res = functV(5, 20)
        res_expected = 101.816
        np.testing.assert_allclose(res, res_expected,
                                   rtol=1e-3)

        # check the raising of errors
        with self.assertRaises(errors.CurrentError):
            functV(1, 20)
        with self.assertRaises(errors.HeadError):
            functV(6, 100)

        # check desactivation of error raising (if not working, raises errors)
        functV(7, 120, error_raising=False)

    def test_functIforVH(self):
        """Test whether the standard deviation between real data and
        simulated data is acceptable (<3%), if the output functV works
        well, and if this function is able to correctly raise errors.
        """
        # check standard deviation
        functI, stddev, intervals = self.pump1.functIforVH()
        np.testing.assert_array_less(stddev, 3)

        # check computing through functV
        res = functI(80, 20)
        res_expected = 3.2861
        np.testing.assert_allclose(res, res_expected,
                                   rtol=1e-3)

        # check the raising of errors
        with self.assertRaises(errors.VoltageError):
            functI(50, 20)

        # check desactivation of error raising (if not working, raises errors)
        functI(7, 120, error_raising=False)

    def test_functQforVH(self):
        """Test whether the standard deviation between real data and
        simulated data is acceptable (<3%), if the output functV works
        well, and if this function is able to correctly raise errors.
        """
        # check standard deviation
        functQ, stddev = self.pump1.functQforVH()
        np.testing.assert_array_less(stddev, 3)

        # check computing through functV
        res = functQ(120, 20)
        res_expected = 57.12
        np.testing.assert_allclose(res, res_expected,
                                   rtol=1e-3)

        # check the raising of errors
        with self.assertRaises(errors.VoltageError):
            functQ(20, 20)
        with self.assertRaises(errors.HeadError):
            functQ(75, 120)

    def test_IVcurvedata(self):
        """Test if the IV curve is as expected.
        """
        IV = self.pump1.IVcurvedata(28)
        IV_expected = ([2.18428526, 2.29238051, 2.40047576, 2.50857101,
                        2.61666626, 2.72476151, 2.83285676, 2.94095201,
                        3.04904726, 3.15714251, 3.26523776, 3.37333301,
                        3.48142825, 3.5895235, 3.69761875, 3.805714,
                        3.91380925, 4.0219045, 4.12999975, 4.238095,
                        4.34619025, 4.4542855, 4.56238075, 4.670476,
                        4.77857125, 4.8866665, 4.99476175, 5.102857,
                        5.21095225, 5.3190475, 5.42714275, 5.535238,
                        5.64333325, 5.7514285, 5.85952375, 5.967619,
                        6.07571425, 6.1838095, 6.29190475, 6.4],
                       [-1., 71.25751693, 72.38928364, 73.52953971,
                        74.67828513, 75.8355199, 77.00124402, 78.17545749,
                        79.35816031,  80.54935247,  81.74903399,  82.95720486,
                        84.17386508,  85.39901465,  86.63265357,  87.87478183,
                        89.12539945,  90.38450642,  91.65210274,  92.92818841,
                        94.21276342,  95.50582779,  96.80738151,  98.11742458,
                        99.43595699, 100.76297876, 102.09848988, 103.44249035,
                        104.79498017, 106.15595933, 107.52542785, 108.90338572,
                        110.28983294, 111.6847695, 113.08819542, 114.50011069,
                        115.9205153, 117.34940927, 118.78679259, 120.23266526])
        np.testing.assert_allclose((IV['I'], IV['V']), IV_expected,
                                   rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
