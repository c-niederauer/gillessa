import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import pandas as pd
from tqdm import tqdm
from IPython import display

#%% SSA degradation (A -(k)> B)
# initialize system
n_0 = 20
k = 0.1
dt = 0.0005
tmax = 100000
n_t = [n_0]

# simulate
for t in range(1, tmax):
    r = np.random.uniform()
    a_t = n_t[t-1]
    if r < (a_t * k * dt):
        n_t.append(a_t - 1)
    else:
        n_t.append(a_t)

# plot results
plt.figure()
plt.plot(n_t)
plt.show()

#%% Dimerization A+A <-(kon, koff)-> AA

# initialize params
n0 = 100
k1 = 0.001
k2 = 0.005
maxstep = 200
At = [n0]
tlist = [0]

def tauTimeExp(a0):
    '''returns tau given r1 in (0,1)'''
    # tau = (1/a0)*np.log(1/r1)
    tau = np.random.exponential(1/a0)
    return tau

def sim(n0, k1, k2, maxstep):
    At = [n0]
    tlist = [0]

    for step in range(1, maxstep):
        '''simulate'''
        # particle numbers
        a_t = At[step-1]
        aa_t = n0 - a_t

        # sample r
        r2 = np.random.uniform()

        # calculate alphas and get tau
        a1 = a_t * (a_t - 1) * k1
        a2 = aa_t * k2
        a0 = a1 + a2
        tau = tauTimeExp(a0)

        # add t+tau to list
        t = tlist[step-1]
        tlist.append(t+tau)

        # choose reaction
        if r2 < (a1/a0):
            At.append(a_t - 2)
        else:
            At.append(a_t + 2)
        
    return At, tlist

particles, times = sim(n0, k1, k2, maxstep)

# plot results
plt.figure()
plt.plot(times, particles, label='monomer')
plt.plot(times, [n0-i for i in particles], label='dimer')
plt.title('Dimerization')
plt.legend()
plt.show()
#%%
def tauTime(avec):
    '''returns tau given r1 in (0,1)'''
    r1 = np.random.uniform()
    a0 = sum(avec)
    tau = (1/a0)*np.log(1/r1)
    return tau


def get_alphas(pvec, kvec):
    '''Returns probability of reaction events: alpha_i
    pvec : [a_t, f_t, fa_t, faf_t]
    kvec : [k1, k2, k3, k4]
    '''
    a_t, f_t, fa_t, faf_t = pvec
    k1, k2, k3, k4, k5 = kvec

    # propensity functions
    a1 = f_t * a_t * k1
    a2 = fa_t * k2
    a3 = fa_t * f_t * k3
    a4 = faf_t * k4
    a5 = faf_t * k5

    return [a1, a2, a3, a4, a5]


def chooseR(pvec, avec):
    r2 = np.random.uniform()
    a_t, f_t, fa_t, faf_t = pvec
    a1, a2, a3, a4, a5 = avec
    a0 = sum(avec)

    # a1 conditions
    if r2 < a1/a0:
        a_t -= 1
        f_t -= 1
        fa_t += 1
    
    # a2 conditions
    if a1/a0 <= r2 < (a1+a2)/a0:
        a_t += 1
        f_t += 1
        fa_t -= 1
    
    # a3 conditions:
    if (a1+a2)/a0 <= r2 < (a1+a2+a3)/a0:
        f_t -= 1
        fa_t -= 1
        faf_t += 1
    
    # a4 conditions:
    if (a1+a2+a3)/a0 <= r2 < (a1+a2+a3+a4)/a0:
        f_t += 1
        fa_t += 1
        faf_t -= 1
    
    if (a1+a2+a3+a4)/a0 <= r2 < a0:
        f_t += 2
        a_t += 1
        faf_t -= 1
    
    return [a_t, f_t, fa_t, faf_t]


def sim(pvec_initial, kvec, maxstep):
    '''Returns list of lists for particle number evolution: [At, Ft, FAt, FAFt]'''
    plist = [pvec_initial]
    tlist = [0]

    for step in range(1, maxstep):
        '''simulate'''
        # vector of particle numbers at step: [a_t, f_t, fa_t, faf_t]
        pvec = plist[step-1]

        # get vector of reaction probabilities avec
        avec = get_alphas(pvec=pvec, kvec=kvec)

        # add tau time to list
        tau = tauTime(avec)
        t = tlist[step-1]
        tlist.append(t+tau)

        # choose reaction and update pvec
        pvec_update = chooseR(pvec, avec)
        plist.append(pvec_update)
        
    return plist, tlist

#%%
particles, times = sim(pvec_initial=[250,175,0,0], kvec=[.02,.0001,.002,.001,.001], maxstep=1000)

plt.figure(figsize=[10,6])
plt.plot(times, [i[0] for i in particles], label='A(t)')
plt.plot(times, [i[1] for i in particles], label='F(t)')
plt.plot(times, [i[2] for i in particles], label='FA(t)')
plt.plot(times, [i[3] for i in particles], label='FAF(t)')
plt.legend()
plt.show()

#%%
plt.figure(figsize=[12,7])

for k in tqdm(np.linspace(.05, 1, 1)):
    # initialize
    steady_faf = []
    steady_fa = []
    fmax = 200
    amax = 1000

    for x in range(1, amax, 5):
        particles, times = sim([x,fmax,0,0], kvec=[.05, .0001, k, .001, .001], maxstep=10000)
        faf = [i[3] for i in particles][1000:]
        fa = [i[2] for i in particles][1000:]
        ss_faf = np.divide(sum(faf), 10000)
        ss_fa = np.divide(sum(fa), 10000)
        steady_faf.append(ss_faf)
        steady_fa.append(ss_fa)

    plt.plot(steady_fa, label=f'FA_{k:.2f}')
    plt.plot(steady_faf, label=f'FAF_{k:.2f}')

# config plot
plt.xlabel('conc. AP20187')
plt.ylabel('particles')
plt.xscale('log')
plt.legend()
plt.show()
#%%
