import control as cnt
import numpy as np
import matplotlib.pyplot as plt


gamma = -0.15   # radians (equilibrium)
v = 50.8691     # m/s (equilibrium)
n = 0.9887      # L/mg (where L is lift factor) (equilibrium)
k1 = 61.6594    # glider constant
k2 = 4.8747e-5  # glider constant
g = 9.81        # gravity (m/s^2)

# The equilibrium values of x1, x2, and input u
xe1 = gamma
xe2 = v
ue = n

A11 = np.sin(xe1) * xe2 / g  # Upper left
A12 = (-np.cos(xe1) / g) - (ue * g / np.power(xe2, 2))  # Upper right
A21 = -np.cos(xe1) / g  # Lower left
A22 = (2 * k1 * np.power(ue, 2) * g / np.power(xe2, 3)) - (2 * k2 * xe2 * g)

B1 = g / xe2
B2 = -2 * k1 * ue * g / np.power(xe2, 2)


X1_over_U_numerator = [B1, (A12 * B2 - A22 * B1)]
X1_over_U_denominator = [1, -(A22 + A11), A22 * A11 - A12 * A21]
H1 = cnt.tf(X1_over_U_numerator, X1_over_U_denominator)

X2_over_U_numerator = [B2, (A21 * B1 - A11 * B2)]
X2_over_U_denominator = [1, -(A22 + A11), A22 * A11 - A12 * A21]
H2 = cnt.tf(X2_over_U_numerator, X2_over_U_denominator)

A = 1
B = -(A22 + A11)
C = A22 * A11 - A12 * A21
print((-B + np.sqrt(B**2 - 4*A*C)) / 2*A)
print((-B - np.sqrt(B**2 - 4*A*C)) / 2*A)

cnt.rlocus(H1 / (1 + H1));
plt.show()

cnt.rlocus(H2 / (1 + H2));
plt.show()