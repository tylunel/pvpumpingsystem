# -*- coding: utf-8 -*-
"""
@author: Tanguy Lunel
"""

import numpy as np
import pandas as pd
import pytest
import os
import inspect

import pvpumpingsystem.pump as pp
from pvpumpingsystem import errors

# ignore all OptimizeWarning in the module
pytestmark = pytest.mark.filterwarnings(
    "ignore::scipy.optimize.OptimizeWarning")

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


@pytest.fixture
def pumpset():
    pump_testfile = os.path.join(test_dir,
                                 '../data/pump_files/SCB_10_150_120_BL.txt')
    return pp.Pump(path=pump_testfile,
                   idname='SCB_10',
                   modeling_method='arab',
                   motor_electrical_architecture='permanent_magnet')


def test_init(pumpset):
    assert type(pumpset.specs) is pd.DataFrame
    assert type(pumpset.coeffs['coeffs_f1']) is np.ndarray


def test_limited_pump_data_1():
    """Tests 'theoretical_basic' model.

    This pump data has only one data point"""

    pump_testfile = os.path.join(
            test_dir, '../data/pump_files/min_specs.txt')
    # The pump is modeled with extremely basic model
    pumpset = pp.Pump(path=pump_testfile,
                      modeling_method='theoretical',
                      motor_electrical_architecture='permanent_magnet')

    functQ, intervals = pumpset.functQforPH()
    res = functQ(540, 50)['Q']
    res_expected = 2.1
    np.testing.assert_allclose(res, res_expected, rtol=0.05)

    res = functQ(540, 1)['Q']
    res_expected = 105  # not physically consistent! Work on the model needed
    np.testing.assert_allclose(res, res_expected, rtol=0.05)


def test_limited_pump_data_3():
    """Tests 'theoretical_constant_efficiency' model and recalculation of
    specs when some are lacking.

    This pump data has 8 points given at one same voltage, and
    with no information on efficiency.
    """

    pump_testfile = os.path.join(
            test_dir, '../data/pump_files/rosen_SC33-158-D380-9200.txt')
    # The initialization recalculates the current used based on a constant
    # efficiency
    pumpset = pp.Pump(path=pump_testfile,
                      modeling_method='theoretical',
                      motor_electrical_architecture='permanent_magnet')
    # check that the current values are not all the same
    assert not (pumpset.specs.current[0] == pumpset.specs.current[1:]).all()

    functI, intervals = pumpset.functIforVH()
    res = functI(540, 50)
    res_expected = 17.94
    np.testing.assert_allclose(res, res_expected, rtol=0.05)

    functQ, intervals = pumpset.functQforVH()
    res = functQ(540, 50)['Q']
    res_expected = 487
    np.testing.assert_allclose(res, res_expected, rtol=0.05)


def test_all_models_coeffs(pumpset):
    """
    Assert that all models manage to have a r_squared value of minimum 0.8
    after the curve_fit.
    'theoretical' here corresponds to 'theoretical_variable_efficiency'
    """

    for model in ['arab', 'kou', 'theoretical', 'hamidat']:
        # sets the model to use and redo the curve-fit
        pumpset.modeling_method = model
        assert pumpset.coeffs['r_squared_f2'] > 0.8


def test_functIforVH(pumpset):
    """
    Test if the output functV works well,
    and if this function is able to correctly raise errors.
    """
    for model in ['arab', 'kou', 'theoretical']:
        pumpset.modeling_method = model
        # check standard deviation
        functI, intervals = pumpset.functIforVH()

        # check computing through functV
        res = functI(80, 20)
        res_expected = 3.3
        np.testing.assert_allclose(
                res,
                res_expected,
                rtol=0.05)  # 5% relative error accepted between the models

        # check the raising of errors
        with pytest.raises(errors.VoltageError):
            functI(50, 20)

        # check desactivation of error raising (if not working, raises errors)
        functI(7, 120, error_raising=False)


def test_functQforPH(pumpset):
    """
    Test whether the output functV works well,
    and if this function is able to correctly raise errors.
    """
    for model in ['arab', 'kou', 'theoretical', 'hamidat']:
        # sets the model to use and redo the curve-fit
        pumpset.modeling_method = model
        # check standard deviation
        functQ, stddev = pumpset.functQforPH()

        # check computing through functV
        res = functQ(400, 20)['Q']
        res_expected = 36.27
        np.testing.assert_allclose(res, res_expected,
                                   rtol=0.05)

        # check the processing of unused power when head is too high
        res = functQ(560, 81)['P_unused']
        res_expected = 560
        np.testing.assert_allclose(res, res_expected,
                                   rtol=0.05)


def test_iv_curve_data(pumpset):
    """Test if the IV curve is as expected.
    """
    IV = pumpset.iv_curve_data(28)
    IV_expected = (
        [2.60514703, 2.70275489, 2.80036276, 2.89797062, 2.99557848,
         3.09318635, 3.19079421, 3.28840207, 3.38600994, 3.4836178,
         3.58122566, 3.67883353, 3.77644139, 3.87404925, 3.97165712,
         4.06926498, 4.16687284, 4.26448071, 4.36208857, 4.45969643,
         4.55730430, 4.65491216, 4.75252002, 4.85012789, 4.94773575,
         5.04534361, 5.14295148, 5.24055934, 5.33816720, 5.43577507,
         5.53338293, 5.63099079, 5.72859866, 5.82620652, 5.92381438,
         6.02142225, 6.11903011, 6.21663797, 6.31424584, 6.4118537],
        [73.11753597,  74.31965043,  75.52176489,  76.72387936,
         77.92599382,  79.12810828,  80.33022274,  81.53233721,
         82.73445167,  83.93656613,  85.13868059,  86.34079505,
         87.54290952,  88.74502398,  89.94713844,  91.14925290,
         92.35136737,  93.55348183,  94.75559629,  95.95771075,
         97.15982522,  98.36193968,  99.56405414, 100.76616860,
         101.9682830, 103.17039753, 104.37251199, 105.57462645,
         106.7767409, 107.97885538, 109.18096984, 110.38308430,
         111.5851987, 112.78731323, 113.98942769, 115.19154215,
         116.3936566, 117.59577108, 118.79788554, 120.])
    np.testing.assert_allclose((IV['I'], IV['V']), IV_expected,
                               rtol=1e-3)


if __name__ == '__main__':
    pytest.main(['test_pump.py'])
