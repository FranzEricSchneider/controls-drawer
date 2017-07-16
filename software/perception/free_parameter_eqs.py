import numpy as np
from numpy import eye, cos, sin


def nonLinearLeastSquares(f, vertices, exteriorPts, plotValues=False):
    # Method and many of the more meaningless names taken from here:
    # http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html

    # Choose initial values for the free parameters
    phi = np.pi
    omega = 0.0
    kappa = -1.0
    s_14 = 0.01
    s_24 = 0.02
    s_34 = 0.07
    # Track the parameters over time in a matrix, use the latest values to
    # calculate each consecutive step
    freeParameters = np.array([phi, omega, kappa, s_14, s_24, s_34]).reshape(6, 1)

    # TODO: Make this for loop a combination of delta resolution and maximum
    #       iterations
    for i in range(50):
        # Loop through every measurement point
        residuals = None
        AMatrix = None
        for i in xrange(len(vertices)):
            x_1 = exteriorPts[i][0]
            x_2 = exteriorPts[i][1]
            x_3 = exteriorPts[i][2]

            # y1 and y2 are the "measured output" variables, the (x,y) values
            # in the image frame
            y1 = vertices[i][0]
            y2 = vertices[i][1]

            # Calculate current residuals
            newResiduals = np.array([
                [y1 - function1(freeParameters[:, -1], x_1, x_2, x_3)],
                [y2 - function2(freeParameters[:, -1], x_1, x_2, x_3)],
            ])
            if residuals is None:
                residuals = newResiduals
            else:
                residuals = np.vstack((residuals, newResiduals))

            newAMatrix = matrix_row(freeParameters[:, -1], x_1, x_2, x_3)
            if AMatrix is None:
                AMatrix = newAMatrix
            else:
                AMatrix = np.vstack((AMatrix, newAMatrix))

        # I know the names don't mean anything, see Wolfram link
        aMatrix = AMatrix.T.dot(AMatrix)
        bMatrix = AMatrix.T.dot(residuals)
        deltaFreeParameters = np.linalg.solve(aMatrix, bMatrix)
        freeParameters = np.hstack((freeParameters,
                                    freeParameters[:, -1].reshape(6, 1) + deltaFreeParameters))

    if plotValues:
        import matplotlib.pyplot as plt
        titles = ["phi", "omega", "kappa", "s_14", "s_24", "s_34"]
        for i in range(6):
            plt.subplot(3, 2, i + 1)
            plt.plot(freeParameters[i, :], "o-")
            plt.title(titles[i])
            plt.xlabel("Iterations")
        plt.show()

    # Return the best guess (most settled) values for the parameters
    return freeParameters[:, -1]


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
        df1_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df1_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df1_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df1_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df1_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df1_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
    ])


def f2_row(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
    return np.array([
        df2_dphi(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df2_domega(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df2_dkappa(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df2_ds_14(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df2_ds_24(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
        df2_ds_34(phi, omega, kappa, s_14, s_24, s_34, x_1, x_2, x_3),
    ])


def function1(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
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

def function2(parameters, x_1, x_2, x_3, f):
    phi, omega, kappa, s_14, s_24, s_34 = parameters
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
