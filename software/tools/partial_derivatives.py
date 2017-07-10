import numpy as np
from numpy import cos, sin


def matrix_row(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return np.vstack([
        f1_row(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        f2_row(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
    ])

def f1_row(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return np.array([
        df1_dphi(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df1_domega(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df1_dkappa(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df1_ds_14(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df1_ds_24(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df1_ds_34(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),  
    ])

def f2_row(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return np.array([
        df2_dphi(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df2_domega(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df2_dkappa(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df2_ds_14(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df2_ds_24(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),
        df2_ds_34(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34),  
    ])


# In addition to the derivatives we also need to calculate the functions themselves

def function1(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (
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

def function2(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (
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

def df1_dphi(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (-x_1*cos(phi) - x_2*sin(omega)*sin(phi) + x_3*sin(phi)*cos(omega)) *\
           (s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2 +\
           (-x_1*sin(phi)*cos(kappa) + x_2*sin(omega)*cos(kappa)*cos(phi) - x_3*cos(kappa)*cos(omega)*cos(phi)) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df1_domega(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (x_2*(-sin(kappa)*sin(omega) + sin(phi)*cos(kappa)*cos(omega)) + x_3*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi)) +\
           (x_2*cos(omega)*cos(phi) + x_3*sin(omega)*cos(phi)) *\
           (s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2

def df1_dkappa(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (-x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df1_ds_14(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return 1 / (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df1_ds_24(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return 0

def df1_ds_34(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return -(s_14 + x_1*cos(kappa)*cos(phi) + x_2*(sin(kappa)*cos(omega) + sin(omega)*sin(phi)*cos(kappa)) + x_3*(sin(kappa)*sin(omega) - sin(phi)*cos(kappa)*cos(omega))) /\
            (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2


def df2_dphi(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (-x_1*cos(phi) - x_2*sin(omega)*sin(phi) + x_3*sin(phi)*cos(omega)) *\
           (s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2 +\
           (x_1*sin(kappa)*sin(phi) - x_2*sin(kappa)*sin(omega)*cos(phi) + x_3*sin(kappa)*cos(omega)*cos(phi)) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df2_domega(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (x_2*(-sin(kappa)*sin(phi)*cos(omega) - sin(omega)*cos(kappa)) + x_3*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi)) +\
           (x_2*cos(omega)*cos(phi) + x_3*sin(omega)*cos(phi)) *\
           (s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2

def df2_dkappa(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return (-x_1*cos(kappa)*cos(phi) + x_2*(-sin(kappa)*cos(omega) - sin(omega)*sin(phi)*cos(kappa)) + x_3*(-sin(kappa)*sin(omega) + sin(phi)*cos(kappa)*cos(omega))) /\
           (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df2_ds_14(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return 0

def df2_ds_24(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return 1 / (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))

def df2_ds_34(phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34):
    return -(s_24 - x_1*sin(kappa)*cos(phi) + x_2*(-sin(kappa)*sin(omega)*sin(phi) + cos(kappa)*cos(omega)) + x_3*(sin(kappa)*sin(phi)*cos(omega) + sin(omega)*cos(kappa))) /\
            (s_34 + x_1*sin(phi) - x_2*sin(omega)*cos(phi) + x_3*cos(omega)*cos(phi))**2
