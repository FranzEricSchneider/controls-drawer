from sympy import symbols, sin, cos, diff, latex


# Calculus partial derivatives using this tutorial: http://docs.sympy.org/latest/tutorial/calculus.html
# To see why we need the partial derivatives see the LaTeX

p_1, p_2, f, phi, omega, kappa, x_1, x_2, x_3, s_14, s_24, s_34 = \
    symbols('p_1 p_2 f phi omega kappa x_1 x_2 x_3 s_14 s_24 s_34')

# Here's the first equation:
# \begin{align*}
#     F_1 = \frac{p_1}{f}
#     &=
#     \frac{
#         [cos(\phi) cos(\kappa)] x_1 +
#         [sin (\omega) sin(\phi) cos(\kappa) + cos(\omega) sin(\kappa)] x_2 +
#         [-cos(\omega) sin(\phi) cos(\kappa) + sin(\omega) sin(\kappa)] x_3 +
#         s_{14}
#     }
#     {
#         [sin(\phi)] x_1 +
#         [-sin(\omega) cos(\phi)] x_2 +
#         [cos(\omega) cos(\phi)] x_3 +
#         s_{34}
#     }
# \end{align*}

function1 = (
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

df1_dphi = diff(function1, phi)
df1_domega = diff(function1, omega)
df1_dkappa = diff(function1, kappa)
df1_ds_14 = diff(function1, s_14)
df1_ds_24 = diff(function1, s_24)
df1_ds_34 = diff(function1, s_34)

print("df1_dphi: \n{}\n".format(df1_dphi))
print("df1_domega: \n{}\n".format(df1_domega))
print("df1_dkappa: \n{}\n".format(df1_dkappa))
print("df1_ds_14: \n{}\n".format(df1_ds_14))
print("df1_ds_24: \n{}\n".format(df1_ds_24))
print("df1_ds_34: \n{}\n".format(df1_ds_34))
# print("df1_dphi: \n{}\n".format(latex(df1_dphi)))
# print("df1_domega: \n{}\n".format(latex(df1_domega)))
# print("df1_dkappa: \n{}\n".format(latex(df1_dkappa)))
# print("df1_ds_14: \n{}\n".format(latex(df1_ds_14)))
# print("df1_ds_24: \n{}\n".format(latex(df1_ds_24)))
# print("df1_ds_34: \n{}\n".format(latex(df1_ds_34)))

# Here's the second equation:
# \begin{align*}
#     F_2 = \frac{p_2}{f}
#     &=
#     \frac{
#         [-cos(\phi) sin(\kappa)] x_1 +
#         [-sin(\omega) sin(\phi) sin(\kappa) + cos(\omega) cos(\kappa)] x_2 +
#         [cos(\omega) sin(\phi) sin(\kappa) + sin(\omega) cos(\kappa)] x_3 +
#         s_{24}
#     }
#     {
#         [sin(\phi)] x_1 +
#         [-sin(\omega) cos(\phi)] x_2 +
#         [cos(\omega) cos(\phi)] x_3 +
#         s_{34}
#     }
# \end{align*}

function2 = (
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

df2_dphi = diff(function2, phi)
df2_domega = diff(function2, omega)
df2_dkappa = diff(function2, kappa)
df2_ds_14 = diff(function2, s_14)
df2_ds_24 = diff(function2, s_24)
df2_ds_34 = diff(function2, s_34)

print("df2_dphi: \n{}\n".format(df2_dphi))
print("df2_domega: \n{}\n".format(df2_domega))
print("df2_dkappa: \n{}\n".format(df2_dkappa))
print("df2_ds_14: \n{}\n".format(df2_ds_14))
print("df2_ds_24: \n{}\n".format(df2_ds_24))
print("df2_ds_34: \n{}\n".format(df2_ds_34))
# print("df2_dphi: \n{}\n".format(latex(df2_dphi)))
# print("df2_domega: \n{}\n".format(latex(df2_domega)))
# print("df2_dkappa: \n{}\n".format(latex(df2_dkappa)))
# print("df2_ds_14: \n{}\n".format(latex(df2_ds_14)))
# print("df2_ds_24: \n{}\n".format(latex(df2_ds_24)))
# print("df2_ds_34: \n{}\n".format(latex(df2_ds_34)))
