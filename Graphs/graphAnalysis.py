# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 17:42:31 2022

@author: zgl12
"""

# %% 

import numpy as np
import matplotlib.pyplot as plt
from parameter_MLE import Parameter_Strip as ParS
import freq_code_analysis as freq
#from freq_code_analysis import Timescape, LCDM, Milne


# def Plotting():
#     plt.figure()
#     plt.tight_layout()
#     plt.plot(lin,ts_x1, color = 'k', label = 'Timescape')
#     plt.plot(lin,lcdm_x1, color = 'C1', label = '$\Lambda$CDM')
#     #plt.plot(lin,milne_x1, color = 'C0', label = 'Milne')
#     plt.axvline(0.033, linestyle = 'dotted', color = 'k', alpha = 0.3)
#     plt.axvline(0.024, linestyle = 'dotted', color = 'k', alpha = 0.3)
#     plt.xlim(0,0.1)
#     plt.ylabel(r'$x_{1}$', fontsize = 13)
#     plt.xlabel('$z_{min}$', fontsize = 13)
#     plt.legend()
#     #plt.savefig("Hmmm_Zac_colour.pdf", format="pdf", bbox_inches="tight", dpi=1200)
#     plt.show()



timescape_JLA = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_TS.txt')
lamCDM_JLA = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_LCDM.txt')
milne_JLA = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_Milne.txt')


TS_omega_JLA, TS_alpha_JLA, TS_beta_JLA, TS_x_JLA,TS_c_JLA, TS_M_JLA = freq.Timescape(timescape_JLA)
lamCDM_omega_JLA, lamCDM_alpha_JLA, lamCDM_beta_JLA, lamCDM_x_JLA, lamCDM_c_JLA, lamCDM_M_JLA = freq.LCDM(lamCDM_JLA)

model_ts = 1
model_lcdm = 2
model_milne = 3

evidence = 0.00001
mle = 13
standard = 2
sample = 'JLA/'
file_folder = 'LinearSpace/'
prefix = 'JLA_'

bayes_lin = np.linspace(0,0.1,21)
freq_lin = np.linspace(0,0.1,41)


Q, logZ, imp_logZ, alpha, beta, colour, x1, omega_uncert = ParS(prefix,model_ts,evidence,sample,file_folder,mle)

fv0 = np.array(Q, dtype=float)
ts_logZ = np.array(logZ, dtype=float)
ts_imp_logZ = np.array(imp_logZ, dtype=float)
ts_alpha = np.array(alpha, dtype=float)
ts_beta = np.array(beta, dtype=float)
ts_colour = np.array(colour, dtype=float)
ts_x1 = np.array(x1, dtype=float)
ts_omega_uncert = np.array(omega_uncert, dtype = str)
t = [] #1    0.743223535447498973E+00    0.359111920475504373E-01
for ts in ts_omega_uncert:
    ts = ts.split(' ')
    ts = [x for x in ts if x]
    ts = ts[2]
    t.append(ts)
ts_omega_uncert = np.array(t, dtype = float)

Q, logZ, imp_logZ, alpha, beta, colour, x1, omega_uncert = ParS(prefix,model_lcdm,evidence,sample,file_folder,mle)

Omega_M = np.array(Q, dtype=float)
lcdm_logZ = np.array(logZ, dtype=float)
lcdm_imp_logZ = np.array(imp_logZ, dtype=float)
lcdm_alpha = np.array(alpha, dtype=float)
lcdm_beta = np.array(beta, dtype=float)
lcdm_colour = np.array(colour, dtype=float)
lcdm_x1 = np.array(x1, dtype=float)
lcdm_omega_uncert = np.array(omega_uncert, dtype = str)
l = [] #1    0.743223535447498973E+00    0.359111920475504373E-01
for lc in lcdm_omega_uncert:
    lc = lc.split(' ')
    lc = [x for x in lc if x]
    lc = lc[2]
    l.append(lc)
lc_omega_uncert = np.array(l, dtype = float)

plt.rcParams['font.size']= 15

# plt.figure(figsize = (9,5.5))
# plt.tight_layout()
# plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.plot(bayes_lin, ts_alpha, color='k', label = 'Timescape')
# plt.plot(freq_lin, TS_alpha_JLA, color='k', linestyle = '--')
# plt.plot(bayes_lin, lcdm_alpha, linestyle = '-', color='C1', label = '$\Lambda$CDM')
# plt.plot(freq_lin, lamCDM_alpha_JLA, color='C1', linestyle = '--')
# plt.xlabel('$z_{min}$')
# plt.ylabel(r'$\alpha$')
# plt.legend()
# plt.xlim(0,0.1)
# # plt.savefig('panthP_bayes_freq_alpha.pdf', format = 'pdf', dpi = 1200)
# plt.show()


# plt.figure(figsize = (9,5.5))
# plt.tight_layout()
# plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.plot(bayes_lin, ts_beta, color='k', label = 'Timescape')
# plt.plot(freq_lin, TS_beta_JLA, color='k', linestyle = '--')
# plt.plot(bayes_lin, lcdm_beta, linestyle = '-', color='C1', label = '$\Lambda$CDM')
# plt.plot(freq_lin, lamCDM_beta_JLA, color='C1', linestyle = '--')
# plt.xlabel('$z_{min}$')
# plt.ylabel(r'$\beta$')
# plt.legend()
# plt.xlim(0,0.1)
# # plt.savefig('panthP_bayes_freq_beta.pdf', format = 'pdf', dpi = 1200)
# plt.show()


# plt.figure(figsize = (9,5.5))
# plt.tight_layout()
# plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.plot(bayes_lin, ts_colour, color='k', label = 'Timescape')
# plt.plot(freq_lin, TS_c_JLA, color='k', linestyle = '--')
# plt.plot(bayes_lin, lcdm_colour, linestyle = '-', color='C1', label = '$\Lambda$CDM')
# plt.plot(freq_lin, lamCDM_c_JLA, color='C1', linestyle = '--')
# plt.xlabel('$z_{min}$')
# plt.ylabel(r'$c$')
# plt.legend()
# plt.xlim(0,0.1)
# # plt.savefig('panthP_bayes_freq_colour.pdf', format = 'pdf', dpi = 1200)
# plt.show()


# plt.figure(figsize = (9,5.5))
# plt.tight_layout()
# plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
# plt.plot(freq_lin, TS_x_JLA, color='k', linestyle = '--')
# plt.plot(bayes_lin, ts_x1, color='k', label = 'Timescape')
# plt.plot(bayes_lin, lcdm_x1, linestyle = '-', color='C1', label = '$\Lambda$CDM')
# plt.plot(freq_lin, lamCDM_x_JLA, color='C1', linestyle = '--')
# plt.xlabel('$z_{min}$')
# plt.ylabel(r'$x_{1}$')
# plt.legend()
# plt.xlim(0,0.1)
# # plt.savefig('panthP_bayes_freq_x1.pdf', format = 'pdf', dpi = 1200)
# plt.show()

# plt.figure(figsize = (9,5.5))
# ax = plt.axes()
# plt.axvline(0.033, linestyle = 'dotted', color = 'k', alpha = 0.3)
# plt.axvline(0.024, linestyle = 'dotted', color = 'k', alpha = 0.3)
# lns1 = ax.plot(bayes_lin,Omega_M, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
# plt.xlim(0,0.1)
# plt.xlabel("$z_{min}$")
# ax.set_ylabel("$\Omega_{M0}$")
# ax1 = ax.twinx()
# ax1.fill_between(bayes_lin, fv0 + ts_omega_uncert, fv0 - ts_omega_uncert, alpha = 0.2, color = 'C0')
# ax.fill_between(bayes_lin, Omega_M + lc_omega_uncert, Omega_M - lc_omega_uncert, alpha = 0.2, color = 'C1')
# lns2 = ax1.plot(bayes_lin,fv0, color = 'k', label = 'Timescape')
# ax1.set_ylabel("$f_{v0}$")
# ax1.set_ylim(max(fv0)+0.048,min(fv0)-0.048,)
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=4)
# # plt.savefig("JLA_bayes_fv0_omega.pdf", dpi = 1200)
# plt.show()


# %%