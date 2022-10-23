# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:53:14 2022

@author: zgl12
"""

import numpy as np
from splineSearch import sply as spl
from scipy import optimize
import time

start = time.time()

omega = np.linspace(0,0.65,101)
omega[0]= 0.001

ts_chi = []
tsLL = []
ts_fun = []

lcdm_chi = []
lcdmLL = []
lcdm_fun = []

#fv0 = 0.5*(np.sqrt(9-8*om)-1)
tolerance = 1e-10
prefix = 'CORR_'

for om in omega:

    def LCDM_chi(Q):
        # Pantheon+ subsample
        omega_lambda = 1.0 - Q[0]
        H0 = 66.7 # km/s/Mpc
        c = 299792.458
        dist = c/H0 * np.hstack([tempdL(Q[0],omega_lambda) for tempdL in spl.lcdm])      
        dist = np.transpose(dist)
        
        data = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_33.txt')
        zcmb = data[:,0]
        zhel = data[:,-3]
        # print(len(zhel))
        
        PP = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_PanthData_33.txt')
        model = PP[:,10]
        cov_inv = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_COVd_33.txt')
    
        mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
        mu = mu[:,0]
        
        sig = (mu-model-Q[1])
        mul = np.matmul(cov_inv, sig)
        chi2 = np.matmul(np.transpose(sig), mul)
        return chi2
    
    def TS_chi(Q):
        # Pantheon+ subsample
        #omega = 0.5*(1.-Q)*(2.+Q)
        dist = (61.7/66.7) * np.hstack([tempdL(Q[0]) for tempdL in spl.ts])
        dist = np.transpose(dist)
        
        data = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_33.txt')
        zcmb = data[:,0]
        zhel = data[:,-3]
        # print(len(zhel))
        
        PP = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_PanthData_33.txt')
        model = PP[:,10]
        cov_inv = np.genfromtxt('CORR_PPLUS_MU_SAMPLE_COVd_33.txt')
        
        mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
        sig = (mu-model-Q[1])
        mul = np.matmul(cov_inv, sig)
        chi2 = np.matmul(np.transpose(sig), mul)
        #chi02 = 2700
        return chi2
    
    def m2CONS( pars , omega = om):  # Constraint empty universe
        return pars[0] - omega


    Q = [om, 0.14]
    
    timescape = optimize.minimize(TS_chi, Q, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONS}, ), tol= tolerance) 
    ts_chi.append(timescape.x)
    tsLL.append(timescape.fun)
    # tsLL.append(np.exp(-0.5*timescape.fun))
    
    lambdacdm = optimize.minimize(LCDM_chi, Q, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONS}, ), tol= tolerance) 
    lcdm_chi.append(lambdacdm.x)
    lcdmLL.append(lambdacdm.fun)
    # lcdmLL.append(np.exp(-0.5*lambdacdm.fun))

end = time.time()

np.savetxt(prefix+'profile_TS.txt', ts_chi, delimiter = '\t')
np.savetxt(prefix+'profile_LCDM.txt', lcdm_chi, delimiter = '\t')
np.savetxt(prefix+'profile_TS_LL.txt', tsLL, delimiter = '\t')
np.savetxt(prefix+'profile_LCDM_LL.txt', lcdmLL, delimiter = '\t')

total_time = (end-start)/3600
print(total_time)