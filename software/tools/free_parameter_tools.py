import numpy as np
from numpy import eye, cos, sin


def HT_from_parameters(parameters):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    HT = eye(4)
    # These are the translational parameters
    HT[0, 3] = s_14
    HT[1, 3] = s_24
    HT[2, 3] = s_34
    # These relationships are laid out in the cse.usf.edu PDF linked in the
    # Sharelatex document
    HT[0, 0] = cos(phi) * cos(kappa)
    HT[0, 1] = sin (omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa)
    HT[0, 2] = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa)
    HT[1, 0] = -cos(phi) * sin(kappa)
    HT[1, 1] = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa)
    HT[1, 2] = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa)
    HT[2, 0] = sin(phi)
    HT[2, 1] = -sin(omega) * cos(phi)
    HT[2, 2] = cos(omega) * cos(phi)
    return HT


def matrix_row(parameters, x_1, x_2, x_3, f):
    return np.vstack([
        f1_row(parameters, x_1, x_2, x_3, f),
        f2_row(parameters, x_1, x_2, x_3, f),
    ])


def f1_row(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    return np.array([
        f * df1_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df1_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df1_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df1_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df1_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df1_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
    ])


def f2_row(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    return np.array([
        f * df2_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df2_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df2_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df2_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df2_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        f * df2_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
    ])


def function1(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    return f *\
            (
                (cos(phi) * cos(kappa)) * x_1 +
                (sin (omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa)) * x_2 +
                (-cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa)) * x_3 +
                s_14
            ) /\
            (
                (sin(phi)) * x_1 +
                (-sin(omega) * cos(phi)) * x_2 +
                (cos(omega) * cos(phi)) * x_3 +
                s_34
            )

def function2(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    return f *\
            (
                (-cos(phi) * sin(kappa)) * x_1 +
                (-sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa)) * x_2 +
                (cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa)) * x_3 +
                s_24
            ) /\
            (
                (sin(phi)) * x_1 +
                (-sin(omega) * cos(phi)) * x_2 +
                (cos(omega) * cos(phi)) * x_3 +
                s_34
            )


# See the Sharelatex for what all these partial derivatives mean
def df1_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (-x_1*cos(phi) - x_2*sin(omega)*sin(phi) + x_3*sin(phi)*cos(omega)) *\
           (s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2 +\
           (-x_1*sin(phi)*cos(kappa) + x_2*sin(omega)*cos(kappa)*cos(phi) - x_3*cos(kappa)*cos(omega)*cos(phi)) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df1_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (x_2*(-sin(kappa)*sin(omega) + sin(phi)*cos(kappa)*cos(omega)) + x_3*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi)) +\
           (x_2*cos(omega)*cos(phi) + x_3*sin(omega)*cos(phi)) *\
           (s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2


def df1_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (-x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df1_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return 1 / (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df1_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return 0


def df1_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return -(s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
            (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2


def df2_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (-x_1*cos(phi) - x_2*sin(omega)*sin(phi) + x_3*sin(phi)*cos(omega)) *\
           (s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2 +\
           (x_1*sin(kappa)*sin(phi) - x_2*sin(kappa)*sin(omega)*cos(phi) + x_3*sin(kappa)*cos(omega)*cos(phi)) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df2_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (x_2*(-sin(kappa)*sin(phi)*cos(omega) - sin(omega)*cos(kappa)) + x_3*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi)) +\
           (x_2*cos(omega)*cos(phi) + x_3*sin(omega)*cos(phi)) *\
           (s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2


def df2_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return (-x_1*cos(kappa)*cos(phi) + x_2*(-sin(kappa)*cos(omega) - sin(omega)*sin(phi)*cos(kappa)) + x_3*(-sin(kappa)*sin(omega) + sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df2_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return 0


def df2_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return 1 / (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))


def df2_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3):
    return -(s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
            (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2
