import sympy
from sympy import Interval, S
from sympy.plotting import plot
from sympy.parsing import sympy_parser
from sympy import lambdify
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
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

### Evaluate expression
expression = lambdify([K_Dd,K_Ds, gamma_Mtot, c_L], gamma_MLM_sol, 'numpy')

input_c_L_uM = np.logspace(-15, 1, 100) #in micromolar   

input_c_L = input_c_L_uM  * 0.6022 #in particles per um3

value_gamma_Mtot = 0.08 #in particles per um2

value_K_Dd= 0.01
value_K_Ds = 0.00000001 

result = expression(value_K_Dd,value_K_Ds, value_gamma_Mtot, input_c_L)

### Plot
f=plt.figure(figsize=[10,6])
f.subplots_adjust(left=0.15,right=0.85,bottom=0.2,top=0.75)
f.clear()
ax=f.add_subplot(111) 
ax.plot(input_c_L, result)

ax.set_ylim(bottom = 0)
ax.set_xlabel('Ligand [nM]')

ytick = ax.get_yticks()
ax.set_yticks(ytick)
ax.set_yticklabels((ytick*5329).astype(int))
ax.set_ylabel('Number of dimers per FOV')
ax.set_xscale('log')


