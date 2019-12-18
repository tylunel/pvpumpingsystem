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
                   model='SCB_10',
                   modeling_method='arab',
                   motor_electrical_architecture='permanent_magnet')


def test_init(pumpset):
    assert type(pumpset.specs) is pd.DataFrame
    assert type(pumpset.coeffs['coeffs_f1']) is np.ndarray


def test_all_models_coeffs(pumpset):
    """
    Assert that all models manage to have a r_squared value of minimum 0.8
    """
    # Arab model
    assert pumpset.coeffs['r_squared_f1'] > 0.8
    # Kou model
    pumpset.modeling_method = 'kou'
    assert pumpset.coeffs['r_squared_f1'] > 0.8
    # theoretical model
    pumpset.modeling_method = 'theoretical'
    assert pumpset.coeffs['r_squared_f1'] > 0.8
    # Hamidat model
    pumpset.modeling_method = 'hamidat'
    assert pumpset.coeffs['r_squared_f2'] > 0.8


def test_functIforVH(pumpset):
    """
    Test if the output functV works
    well, and if this function is able to correctly raise errors.
    """
    # check standard deviation
    functI, intervals = pumpset.functIforVH()

    # check computing through functV
    res = functI(80, 20)
    res_expected = 3.2861
    np.testing.assert_allclose(res, res_expected,
                               rtol=1)

    # check the raising of errors
    with pytest.raises(errors.VoltageError):
        functI(50, 20)

    # check desactivation of error raising (if not working, raises errors)
    functI(7, 120, error_raising=False)


def test_functQforPH(pumpset):
    """
    Test whether the output functV works
    well, and if this function is able to correctly raise errors.
    """
    # check standard deviation
    functQ, stddev = pumpset.functQforPH()

    # check computing through functV
    res = functQ(400, 20)['Q']
    res_expected = 37.09
    np.testing.assert_allclose(res, res_expected,
                               rtol=1e-2)

    # check the processing of unused power when head is too high
    res = functQ(560, 80)['P_unused']
    res_expected = 560
    np.testing.assert_allclose(res, res_expected,
                               rtol=1e-2)


def test_iv_curve_data(pumpset):
    """Test if the IV curve is as expected.
    """
    IV = pumpset.iv_curve_data(28)
    IV_expected = (
        [2.60514703, 2.70275489, 2.80036276, 2.89797062, 2.99557848,
       3.09318635, 3.19079421, 3.28840207, 3.38600994, 3.4836178 ,
       3.58122566, 3.67883353, 3.77644139, 3.87404925, 3.97165712,
       4.06926498, 4.16687284, 4.26448071, 4.36208857, 4.45969643,
       4.5573043 , 4.65491216, 4.75252002, 4.85012789, 4.94773575,
       5.04534361, 5.14295148, 5.24055934, 5.3381672 , 5.43577507,
       5.53338293, 5.63099079, 5.72859866, 5.82620652, 5.92381438,
       6.02142225, 6.11903011, 6.21663797, 6.31424584, 6.4118537 ],
        [73.11753597,  74.31965043,  75.52176489,  76.72387936,
        77.92599382,  79.12810828,  80.33022274,  81.53233721,
        82.73445167,  83.93656613,  85.13868059,  86.34079505,
        87.54290952,  88.74502398,  89.94713844,  91.1492529 ,
        92.35136737,  93.55348183,  94.75559629,  95.95771075,
        97.15982522,  98.36193968,  99.56405414, 100.7661686 ,
       101.96828307, 103.17039753, 104.37251199, 105.57462645,
       106.77674091, 107.97885538, 109.18096984, 110.3830843 ,
       111.58519876, 112.78731323, 113.98942769, 115.19154215,
       116.39365661, 117.59577108, 118.79788554, 120.        ])
    np.testing.assert_allclose((IV['I'], IV['V']), IV_expected,
                               rtol=1e-3)


if __name__ == '__main__':
    pytest.main(['test_pump.py'])
