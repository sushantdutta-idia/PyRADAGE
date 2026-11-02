"""Physical constants and unit conversions used by the KDA model.

Values are CODATA 2022/SI where exact definitions exist.  They are kept in
one module so result files can record the numerical convention explicitly.
"""

from __future__ import annotations

import math


C_LIGHT = 299_792_458.0  # m s^-1, exact
E_CHARGE = 1.602_176_634e-19  # C, exact
M_ELECTRON = 9.109_383_7139e-31  # kg, CODATA 2022
MU_0 = 1.256_637_061_27e-6  # N A^-2, CODATA 2022
SIGMA_T = 6.652_458_7051e-29  # m^2, CODATA 2022
STEFAN_BOLTZMANN = 5.670_374_419e-8  # W m^-2 K^-4, exact-derived
RADIATION_CONSTANT = 4.0 * STEFAN_BOLTZMANN / C_LIGHT  # J m^-3 K^-4

KPC_M = 3.085_677_581_491_367e19
MPC_M = 1.0e3 * KPC_M
JULIAN_YEAR_S = 365.25 * 86_400.0
MYR_S = 1.0e6 * JULIAN_YEAR_S
MHZ_HZ = 1.0e6

FOUR_PI = 4.0 * math.pi

CONSTANT_SET = "SI/CODATA-2022; T_CMB configurable (default 2.7255 K)"

