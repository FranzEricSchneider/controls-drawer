import numpy as np
import pytest

from perception.free_parameter_eqs import HTFromParameters
from utils.geometry_tools import checkOrthonormal


class TestHTFromParameters():
    def testTranslation(self):
        omega = 0.0;  phi = 0.0; kappa = 0.0
        s_14  = 1.0; s_24 = 2.0;  s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [-s_14, -s_24, -s_34]))

        omega = 0.0;  phi = 0.0; kappa = np.pi
        s_14  = 1.0; s_24 = 2.0;  s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [s_14, s_24, -s_34]))

        omega = np.pi; phi = 0.0; kappa = 0.0
        s_14  = 1.0;  s_24 = 2.0;  s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [-s_14, s_24, s_34]))

        omega = 0.0; phi = np.pi; kappa = 0.0
        s_14  = 1.0; s_24 = 2.0;   s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [s_14, -s_24, s_34]))

        omega = np.pi / 2; phi = np.pi / 2; kappa = 0.0
        s_14  = 1.0;      s_24 = 2.0;        s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [-s_34, -s_14, -s_24]))

        omega = -np.pi / 2; phi = np.pi / 2; kappa = 0.0
        s_14  = 1.0;       s_24 = 2.0;        s_34 = 3.0
        HT = HTFromParameters((omega, phi, kappa, s_14, s_24, s_34))
        checkOrthonormal(HT)
        assert all(np.isclose(HT[0:3, 3], [-s_34, s_14, s_24]))
