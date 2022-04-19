import sympy
from sympy import Interval, S
from sympy.plotting import plot
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt


gamma_M, gamma_Mtot, c_L, gamma_ML, gamma_MLM, K_Dd, K_Ds = sympy.symbols(r"\Gamma_\textrm{M} \Gamma_\textrm{M\,\ tot} c_L \Gamma_\textrm{ML} \Gamma_\textrm{MLM} K_\textrm{D\,\ d} K_\textrm{D\,\ s}", real=True, positive=True)

M_balance = gamma_Mtot - gamma_M - gamma_ML - 2*gamma_MLM

M_ML_equil = (gamma_M * c_L)/gamma_ML - K_Ds

ML_MLM_equil = (gamma_M * gamma_ML)/gamma_MLM - K_Dd

gamma_M_sol = sympy.solveset(M_balance, gamma_M, domain=S.Reals)

gamma_M_sol = gamma_M_sol.args[0]

gamma_ML_sol = sympy.solveset(M_ML_equil.subs(gamma_M, gamma_M_sol), gamma_ML, domain=sympy.S.Reals)

gamma_ML_sol = gamma_ML_sol.args[0].args[0]

gamma_MLM_sol = sympy.solveset(ML_MLM_equil.subs([(gamma_M, gamma_M_sol), (gamma_ML, gamma_ML_sol)]), gamma_MLM, domain=S.Reals)

gamma_MLM_sol = gamma_MLM_sol.args[0].args[0]

expression = sympy.simplify(gamma_MLM_sol)

#%%
f=plt.figure(figsize=[10,6])
f.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.75)
f.clear()
ax=f.add_subplot(111)  

nCurves_Gamma = 10
nCurves_KDs = 10
nCurves_KDd = 10

#calculate in terms of molecules per m3, m2

colors_Gamma = plt.cm.Greens(np.linspace(0,1,nCurves_Gamma))
colors_KDd = plt.cm.Blues(np.linspace(0,1,nCurves_KDd))
colors_KDs = plt.cm.Oranges(np.linspace(0,1,nCurves_KDs))


expression = lambdify([K_Dd,K_Ds, gamma_Mtot, c_L], gamma_MLM_sol, 'numpy')

input_c_L = np.linspace(1e-20, 3e-6, 100) #in particles per um3 ?

input_gamma_Mtot = np.linspace(1e-6, 1e-5, nCurves_Gamma)
input_KDs = np.linspace(1e-10, 1e-7, nCurves_KDs)
input_KDd = np.linspace(1e-10, 1e-5, nCurves_KDd)

for idx, value_gamma_Mtot in enumerate(input_gamma_Mtot):
    result = expression(1e-6, 1e-7, value_gamma_Mtot, input_c_L)
    ax.plot(input_c_L, result,  color=colors_Gamma[idx])


for idx, value_K_Ds in enumerate(input_KDs):
    result = expression(1e-6, value_K_Ds, 1e-5, input_c_L)
    ax.plot(input_c_L, result,  color=colors_KDs[idx])

for idx, value_K_Dd in enumerate(input_KDd):
    result = expression(value_K_Dd, 1e-7, 1e-5, input_c_L)
    ax.plot(input_c_L, result,  color=colors_KDd[idx])

ax.set_ylim(bottom = 0)
ax.set_xscale('symlog')
