# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:18:09 2022

@author: zgl12
"""

# %%

import numpy as np
from scipy.integrate import quad
from splineSearch import sply as spl
import statistics as stat
import time

def TS_chi(deltaMu, Q):
    # Pantheon+ subsample
    omega = 0.5*(1.-Q)*(2.+Q)
    dist = (61.7/66.7) * np.hstack([tempdL(omega) for tempdL in spl.ts])
    dist = np.transpose(dist)
    # print(len(dist))
    
    data = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_33.txt')
    zcmb = data[:,0]
    zhel = data[:,-3]
    # print(len(zhel))
    
    PP = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_PanthData_33.txt')
    
    model = PP[:,10]
    
    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25

    sig = (mu-model-deltaMu)
    cov_inv = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_COVd_33.txt')
    mul = np.matmul(cov_inv, sig)
    chi2 = np.matmul(np.transpose(sig), mul)
    chi02 = 1200
    chi_corr = chi2 - chi02
    integral = np.exp(-0.5*chi_corr)
    return integral

def LCDM_chi(deltaMu, Q):
    # Pantheon+ subsample
    omega_lambda = 1.0 - Q
    H0 = 66.7 # km/s/Mpc
    c = 299792.458
    dist = c/H0 * np.hstack([tempdL(Q,omega_lambda) for tempdL in spl.lcdm])      
    dist = np.transpose(dist)
    
    data = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_33.txt')
    zcmb = data[:,0]
    zhel = data[:,-3]    
    
    PP = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_PanthData_33.txt')
    
    model = PP[:,10]

    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
    mu = mu[:,0]
    
    sig = (mu-model-deltaMu)
    cov_inv = np.genfromtxt('zero_case_PPLUS_MU_SAMPLE_COVd_33.txt')
    mul = np.matmul(cov_inv, sig)
    chi2 = np.matmul(np.transpose(sig), mul)
    chi02 = 1200
    chi_corr = chi2 - chi02
    integral = np.exp(-0.5*chi_corr)
    return integral

start = time.time()

# lcdm_chi = np.genfromtxt('FULL_profile_LCDM.txt')
# ts_chi = np.genfromtxt('FULL_profile_TS.txt')
lcdm_chi = np.genfromtxt('ZER0JLA_profile_LCDM.txt')
ts_chi = np.genfromtxt('ZER0JLA_profile_TS.txt')

omega_m = np.linspace(0,0.65,101)
omega_m[0] = 0.001

# fv0 = 0.5*(np.sqrt((9-8*omega_m)) - 1)
fBound = [0.5*(np.sqrt((9-8*min(omega_m))) - 1), 0.5*(np.sqrt((9-8*max(omega_m))) - 1)]
fv0 = np.linspace(fBound[0], fBound[1], 101)


FULL_LCDM = np.genfromtxt('ZER0JLA_profile_LCDM_LL.txt')
FULL_ts = np.genfromtxt('ZER0JLA_profile_TS_LL.txt')

FULL_LCDM = np.exp(-0.5*(FULL_LCDM))
FULL_ts = np.exp(-0.5*(FULL_ts))

FULL_LCDM = FULL_LCDM.tolist()
mxLCDM_FULL_ind = FULL_LCDM.index(max(FULL_LCDM))
mxLCDM_FULL = omega_m[mxLCDM_FULL_ind]

FULL_ts = FULL_ts.tolist()
mxTS_FULL_ind = FULL_ts.index(max(FULL_ts))
mxTS_FULL = fv0[mxTS_FULL_ind]

print('fv0', mxTS_FULL)
print('Om', mxLCDM_FULL)

delta_mu_ts = ts_chi[:,1][mxTS_FULL_ind]
delta_mu_lcdm = lcdm_chi[:,1][mxLCDM_FULL_ind]



delta_mu_ts_std = stat.stdev(ts_chi[:,1])
bounds_ts = 3*delta_mu_ts_std
delta_mu_lcdm_std = stat.stdev(lcdm_chi[:,1])
bounds_lcdm = 3*delta_mu_lcdm_std

print(delta_mu_ts, '+/-', delta_mu_ts_std)
print(delta_mu_lcdm, '+/-', delta_mu_lcdm_std)

P_TS, ts_error_I = quad(TS_chi, delta_mu_ts - bounds_ts, delta_mu_ts + bounds_ts, args = (mxTS_FULL,))
P_LCDM, lcdm_error_I = quad(LCDM_chi, delta_mu_lcdm - bounds_lcdm, delta_mu_lcdm + bounds_lcdm, args = (mxLCDM_FULL,))

print(P_TS)
print(P_LCDM)
print()
print(P_TS/P_LCDM)
print()
print(np.log(P_TS/P_LCDM))

end = time.time()

print((end-start)/3600)

