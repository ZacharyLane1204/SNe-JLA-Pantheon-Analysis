# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 12:38:26 2022

@author: zgl12
"""
# %%
import numpy as np
from getsplines import spl
import time
from scipy.integrate import quad


def TS_p(Q):
    # Pantheon+ subsample
    omega = 0.5*(1.-Q)*(2.+Q)
    dist = (61.7/66.7) * np.hstack([tempdL(omega) for tempdL in spl.ts])
    dist = np.transpose(dist)
    
    data = np.genfromtxt('PantheonPlus.txt')
    zcmb = data[:,0]
    zhel = data[:,6]    
    
    PP = np.genfromtxt('PantheonPlusData.txt')
    
    model = PP[1:,10]
    variance = PP[1:,11]
    
    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
    sig = (mu-model)**2
    chi2_1 = sig/variance**2
    chi2 = np.sum(chi2_1)
    
    chi02 = 1800
    
    integral = np.exp(-(chi2-1900)/2)
    return integral

def LCDM_p(Q):
    # Pantheon+ subsample
    omega_lambda = 1.0 - Q
    H0 = 66.7 # km/s/Mpc
    c = 299792.458
    dist = c/H0 * np.hstack([tempdL(Q,omega_lambda) for tempdL in spl.lcdm])      
    dist = np.transpose(dist)
    
    data = np.genfromtxt('PantheonPlus.txt')
    zcmb = data[:,0]
    zhel = data[:,6]    
    
    PP = np.genfromtxt('PantheonPlusData.txt')
    
    model = PP[1:,10]
    variance = PP[1:,11]
    
    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
    mu = mu[:,0]
    sig = (mu-model)**2
    chi2_1 = sig/variance**2
    chi2 = np.sum(chi2_1)
    
    chi02 = 1800
    
    integral = np.exp(-(chi2-1900)/2)
    return integral

def TS_chi(Q):
    # Pantheon+ subsample
    #omega = 0.5*(1.-Q)*(2.+Q)
    dist = (61.7/66.7) * np.hstack([tempdL(Q) for tempdL in spl.ts])
    dist = np.transpose(dist)
    # print(len(dist))
    
    data = np.genfromtxt('PantheonPlusFull.txt')
    zcmb = data[:,0]
    zhel = data[:,-3]
    # print(len(zhel))
    
    PP = np.genfromtxt('PantheonPlusData.txt')
    
    model = PP[1:,10]
    variance = PP[1:,11]
    
    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
    sig = (mu-model)**2
    chi2_1 = sig/variance**2
    chi2 = np.sum(chi2_1)
    return chi2

def LCDM_chi(Q):
    # Pantheon+ subsample
    omega_lambda = 1.0 - Q
    H0 = 66.7 # km/s/Mpc
    c = 299792.458
    dist = c/H0 * np.hstack([tempdL(Q,omega_lambda) for tempdL in spl.lcdm])      
    dist = np.transpose(dist)
    
    data = np.genfromtxt('PantheonPlusFull.txt')
    zcmb = data[:,0]
    zhel = data[:,-3]    
    
    PP = np.genfromtxt('PantheonPlusData.txt')
    
    model = PP[1:,10]
    variance = PP[1:,11]
    
    mu = 5*np.log10(dist * (1+zhel)/(1+zcmb)) + 25
    mu = mu[:,0]
    sig = (mu-model)**2
    chi2_1 = sig/variance**2
    chi2 = np.sum(chi2_1)
    return chi2


# fv0 = 0.774545
# omega_matter = 0.369875



fv0_prior = [0.5,0.799]
om_prior = [0.143,0.487]

omegA = np.linspace(0,1,51)
omegA = omegA[1:]
ome = omegA[38]


ts = []
lcdm = []
for om in omegA:
    
    #fv0 = 0.5*(np.sqrt(9-8*om)-1)
    
    timescape_chi2 = TS_chi(om)
    lcdm_chi2 = LCDM_chi(om)
    ts.append(timescape_chi2)
    lcdm.append(lcdm_chi2)



    
# ts = TS_chi(0.405)
# lcdm = LCDM_chi(0.405)

SNe = 1701
constraints = 1
dof = SNe - constraints


ts = np.array(ts)
lcdm = np.array(lcdm)

timescape_chi2_dof = ts/dof
lambdacdm_chi2_dof = lcdm/dof


# P_TS, ts_error_I = quad(TS_p, fv0_prior[0], fv0_prior[1])
# P_LCDM, lcdm_error_I = quad(LCDM_p, om_prior[0], om_prior[1])

# B = P_TS/P_LCDM
# lnB = np.log(B)

# print(timescape_chi2)
# print(timescape_chi2_dof)
# # print(P_TS)
# print()
# print(lcdm_chi2)
# print(lambdacdm_chi2_dof)
# # print(P_LCDM)
# print()
# print(B)
# print()
# print(lnB)





# %%
