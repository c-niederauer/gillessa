import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %% Fitting expression to data
def dimer_func(val_c_L, val_K_Dd, val_K_Ds, val_gamma_Mtot):
    inner_func = (val_K_Dd * (val_K_Ds**2) + 2*val_K_Dd * val_K_Ds * val_c_L +
                  val_K_Dd * (val_c_L**2)) + (8*val_K_Ds*val_gamma_Mtot * val_c_L)
    val_gamma_MLM = (-(val_K_Ds+val_c_L)*(val_K_Dd**0.5) * (inner_func**0.5) +
                     inner_func-(4*val_K_Ds*val_gamma_Mtot*val_c_L))/(8*val_K_Ds*val_c_L)
    return val_gamma_MLM


def fit_dimers(xdata, ydata, d_mono=None):
    """
    Fit dimer surface density.
    xdata: Ligand concentration in uM
    ydata: Dimer number per field of view
    """
    # Init start values
    p0 = [1e-6, 1e-7, 1e-2]
    xdata = xdata*0.6022  # convert: uM/L to #/um^3
    ydata = ydata/5329  # convert: #/FOV to #/um^2

    if not d_mono == None:
        eps = 1e-9
        popt, pcov = curve_fit(dimer_func,
                               xdata, ydata,
                               p0=[*p0[0:2], d_mono],
                               bounds=((0, 0, d_mono-eps),
                                       (np.inf, np.inf, d_mono+eps)),
                               maxfev=10000)
    else:
        popt, pcov = curve_fit(dimer_func,
                               xdata, ydata,
                               p0=p0,
                               bounds=((0, 0, 0), (np.inf, np.inf, np.inf)),
                               maxfev=10000)
    return popt, pcov


# %%
# Data
xdata = np.array([1e-25, 1e-10, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])  # ligand conc. uM

# ydata = dimer_func(xdata, 1e-2, 1e-8, 5e-2) #simulate dimer numbers
ydata = 3*(np.array([5, 30, 35, 25, 15, 10, 5, 5, 5])-4)  # divide by FOV size in um2

# Fit
popt, pcov = fit_dimers(xdata, ydata, d_mono=0.04)
perr = np.sqrt(np.diag(pcov))
# Evaluate fit for plotting
xfit = np.logspace(-15, 1, 100)
yfit = dimer_func(xfit, *popt)*5329  # convert #/um^2 to #/FOV

# Plot
f = plt.figure(figsize=[4, 3])
f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
f.clear()
ax = f.add_subplot(111)

ax.plot(xdata, ydata, 'x', c='k')
ax.plot(xfit, yfit, c='darkorange')
ax.set_ylabel('Number of dimers per FOV')
ax.set_xlabel('Ligand concentration [uM]')
ax.set_xscale('log')
ax.set_ylim(bottom=0)
# ax.set_xlim(left=0)
# ax.set_xscale('symlog',linthresh=1e-15)
# xtick = ax.get_xticks()
# xtick_new = np.log10(xtick).astype(int)
# xtick_new[0]=0
# ax.set_xticks(xtick)
# ax.set_xticklabels(xtick_new)
plt.tight_layout()
plt.savefig(path + '_fit.png', dpi=200)
print(
    f'K_D,d = {popt[0]:.2e} +/- {perr[0]:.2e} \n K_D,s = {popt[1]:.2e} +/- {perr[1]:.2e} \n ')

# %% Residuals
# Plot
f = plt.figure(figsize=[10, 6])
f.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.75)
f.clear()
ax = f.add_subplot(111)
yfit_data = dimer_func(xdata, *popt)*5329  # convert #/um^2 to #/FOV
ax.plot(xdata, abs(yfit_data-ydata), '-.')
ytick = ax.get_yticks()
ax.set_yticks(ytick)
ax.set_yticklabels((ytick).astype(int))
ax.set_ylabel('Number of dimers per FOV')
ax.set_xlabel('Ligand concentration [uM]')
ax.set_xlim(left=0)
ax.set_xscale('symlog', linthresh=1e-15)
