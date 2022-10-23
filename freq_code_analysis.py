# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 12:54:47 2022

@author: zgl12
"""
# %% ------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import statistics


def Timescape(timescape):
    
    TS_omega = timescape[0]
    TS_alpha = timescape[1]
    TS_x = timescape[2]
    TS_beta = timescape[4]
    TS_c = timescape[5]
    TS_M = timescape[7]
    TS_omega = np.transpose(TS_omega)
    TS_alpha = np.transpose(TS_alpha)
    TS_x = np.transpose(TS_x)
    TS_beta = np.transpose(TS_beta)
    TS_c = np.transpose(TS_c)
    return TS_omega, TS_alpha, TS_beta, TS_x, TS_c, TS_M

def LCDM(lamCDM):
    
    lamCDM_omega = lamCDM[0]
    lamCDM_alpha = lamCDM[2]
    lamCDM_x = lamCDM[3]
    lamCDM_beta = lamCDM[5]
    lamCDM_c = lamCDM[6]
    lamCDM_M = lamCDM[8]
    lamCDM_omega = np.transpose(lamCDM_omega)
    lamCDM_x = np.transpose(lamCDM_x)
    lamCDM_alpha = np.transpose(lamCDM_alpha)
    lamCDM_beta = np.transpose(lamCDM_beta)
    lamCDM_c = np.transpose(lamCDM_c)
    return lamCDM_omega, lamCDM_alpha, lamCDM_beta, lamCDM_x, lamCDM_c, lamCDM_M

def Milne(milne):
    
    milne_omega = milne[0]
    milne_alpha = milne[2]
    milne_x = milne[3]
    milne_beta = milne[5]
    milne_c = milne[6]
    milne_M = milne[8]
    milne_omega = np.transpose(milne_omega)
    milne_x = np.transpose(milne_x)
    milne_alpha = np.transpose(milne_alpha)
    milne_beta = np.transpose(milne_beta)
    milne_c = np.transpose(milne_c)
    return milne_omega, milne_alpha, milne_beta, milne_x, milne_c, milne_M


timescape_JLA = np.loadtxt(r'ExtraAnalysisFiles/JLA_TS.txt')
lamCDM_JLA = np.loadtxt(r'ExtraAnalysisFiles/JLA_LCDM.txt')
milne_JLA = np.loadtxt(r'ExtraAnalysisFiles/JLA_Milne.txt')

timescape_Panth = np.loadtxt(r'ExtraAnalysisFiles/Panth_TS.txt')
lamCDM_Panth = np.loadtxt(r'ExtraAnalysisFiles/Panth_LCDM.txt')
milne_Panth = np.loadtxt(r'ExtraAnalysisFiles/Panth_Milne.txt')

timescape_PanthP = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_TS.txt')
lamCDM_PanthP = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_LCDM.txt')
milne_PanthP = np.loadtxt(r'ExtraAnalysisFiles/PanthPlus_Milne.txt')


TS_omega_JLA, TS_alpha_JLA, TS_beta_JLA, TS_x_JLA,TS_c_JLA, TS_M_JLA = Timescape(timescape_JLA)
lamCDM_omega_JLA, lamCDM_alpha_JLA, lamCDM_beta_JLA, lamCDM_x_JLA, lamCDM_c_JLA, lamCDM_M_JLA = LCDM(lamCDM_JLA)
milne_omega_JLA, milne_alpha_JLA, milne_beta_JLA, milne_x_JLA, milne_c_JLA, milne_M_JLA = Milne(milne_JLA)

TS_omega_Panth, TS_alpha_Panth, TS_beta_Panth, TS_x_Panth, TS_c_Panth, TS_M_Panth = Timescape(timescape_Panth)
lamCDM_omega_Panth, lamCDM_alpha_Panth, lamCDM_beta_Panth, lamCDM_x_Panth, lamCDM_c_Panth, lamCDM_M_Panth = LCDM(lamCDM_Panth)
milne_omega_Panth, milne_alpha_Panth, milne_beta_Panth, milne_x_Panth, milne_c_Panth, milne_M_Panth = Milne(milne_Panth)

TS_omega_PanthP, TS_alpha_PanthP, TS_beta_PanthP, TS_x_PanthP, TS_c_PanthP, TS_M_PanthP = Timescape(timescape_PanthP)
lamCDM_omega_PanthP, lamCDM_alpha_PanthP, lamCDM_beta_PanthP, lamCDM_x_PanthP, lamCDM_c_PanthP, lamCDM_M_PanthP = LCDM(lamCDM_PanthP)
milne_omega_PanthP, milne_alpha_PanthP, milne_beta_PanthP, milne_x_PanthP, milne_c_PanthP, milne_M_PanthP = Milne(milne_PanthP)



zmin = np.linspace(0,0.1,41)


# Plot JLA
# Plot JLA with Pantheon
# Plot JLA with Pantheon Plus
# Plot all model specific ones
z = zmin[14:]

alpha_percent_pv = abs(TS_alpha_JLA - TS_alpha_PanthP)/TS_alpha_JLA*100
alpha_percent_ave_pv = np.mean(alpha_percent_pv)
alpha_percent_std_pv = statistics.stdev(alpha_percent_pv)

c_percent_pv = abs(TS_c_JLA[:14] - TS_c_PanthP[:14])/abs(TS_c_JLA[:14])*100
c_percent_ave_pv = np.mean(c_percent_pv)
c_percent_std_pv = statistics.stdev(c_percent_pv)

c_percent_hf = abs(TS_c_JLA[14:] - TS_c_PanthP[14:])/abs(TS_c_JLA[14:])*100
c_percent_ave_hf = np.mean(c_percent_hf)
c_percent_std_hf = statistics.stdev(c_percent_hf)

Dark = False


if Dark == False: # Plot Light

    plt.rcParams['font.size'] = '16'
    
    c = ['k', 'C0', 'C1', 'purple']

    plt.figure(figsize = (9,5.5))
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.plot(zmin, TS_alpha_JLA, color= c[0], linestyle = '-', label = 'Timescape')
    # plt.plot(zmin, TS_alpha_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    # plt.plot(zmin, TS_alpha_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_alpha_JLA, linestyle = '-', color= c[1], label = 'Milne')
    # plt.plot(zmin, milne_alpha_Panth, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_alpha_PanthP, linestyle = '-', color= c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_alpha_JLA, color= c[2], linestyle = '-', label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_alpha_Panth, linestyle = '-', color= c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_alpha_PanthP, linestyle = '-', color= c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$', fontsize = 17)
    plt.ylabel(r'$\alpha$', fontsize = 17)
    plt.legend()
    plt.xlim(0,0.1)
    # plt.ylim(0.127,0.141)
    plt.tight_layout()
    # plt.savefig('jla_zac_alpha.pdf', format = 'pdf',dpi = 1200)
    plt.show()


    plt.figure(figsize = (9,5.5))
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.plot(zmin, TS_beta_JLA, color=c[0], linestyle = '-', label = 'Timescape')
    # plt.plot(zmin, TS_beta_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    # plt.plot(zmin, TS_beta_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_beta_JLA, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_beta_Panth, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_beta_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_beta_JLA, color=c[2], linestyle = '-', label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_beta_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_beta_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$', fontsize = 17)
    plt.ylabel(r'$\beta$', fontsize = 17)
    plt.legend()
    plt.xlim(0,0.1)
    # plt.ylim(3.03,3.21)
    plt.tight_layout()
    # plt.savefig('jla_zac_beta.pdf', format = 'pdf', dpi = 1200)
    plt.show()


    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.plot(zmin, TS_x_JLA, color=c[1], linestyle = '-', label = 'JLA')
    plt.plot(zmin, TS_x_Panth, linestyle = '-', color=c[2], label = 'P646')
    plt.plot(zmin, TS_x_PanthP, linestyle = '-', color=c[0], label = 'P$+$584')
    # plt.plot(zmin, milne_x_JLA, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_x_Panth, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_x_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, lamCDM_x_JLA, color= c[2], linestyle = '-', label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_x_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_x_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.xlabel('$z_{min}$', fontsize = 17)
    plt.ylabel(r'$x_{1}$', fontsize = 17)
    plt.xlabel('$z_{min}$', fontsize = 17)
    plt.legend()
    plt.xlim(0,0.1)
    # plt.ylim(0.01,0.17)
    plt.tight_layout()
    # plt.savefig('ts_zac_x.pdf', format = 'pdf', dpi = 1200)
    plt.show()



    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'k', linewidth = 1.5)
    plt.plot(zmin, TS_c_JLA, color=c[1], linestyle = '-', label = 'JLA')
    plt.plot(zmin, TS_c_Panth, linestyle = '-', color=c[2], label = 'P646')
    plt.plot(zmin, TS_c_PanthP, linestyle = '-', color=c[0], label = 'P$+$584')
    # plt.plot(zmin, milne_c_JLA, color=c[1], linestyle = '-', label = 'Milne')
    # plt.plot(zmin, milne_c_Panth, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, milne_c_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, lamCDM_c_JLA, color=c[2], linestyle = '-', label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_c_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.plot(zmin, lamCDM_c_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$', fontsize = 17)
    plt.ylabel(r'$c$', fontsize = 17)
    plt.legend()
    plt.xlim(0,0.1)
    # plt.ylim(-0.027, -0.011)
    plt.tight_layout()
    # plt.savefig('ts_zac_c.pdf', format = 'pdf', dpi = 1200)
    plt.show()

else: # Dark == True:

    plt.rcParams['font.size']= 15
    plt.style.use('dark_background')

    c = ['yellow', 'darkturquoise', 'deeppink']

    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.plot(zmin, TS_alpha_JLA, color= c[0], linestyle = '--')
    # plt.plot(zmin, TS_alpha_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, TS_alpha_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_alpha_JLA, linestyle = '--', color= c[1])
    # plt.plot(zmin, milne_alpha_Panth, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, milne_alpha_PanthP, linestyle = '-', color= c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_alpha_JLA, color= c[2], linestyle = '--')
    # plt.plot(zmin, lamCDM_alpha_Panth, linestyle = '-', color= c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.plot(zmin, lamCDM_alpha_PanthP, linestyle = '-', color= c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$')
    plt.ylabel(r'$\alpha$')
    plt.legend()
    plt.xlim(0,0.1)
    #plt.ylim(0.127,0.141)
    # plt.savefig('dark_panthP_zac_alpha.pdf', format = 'pdf',dpi = 1200)
    plt.show()


    # plt.figure(figsize = (8.5,5.5))
    # ax = plt.axes()
    # plt.tight_layout()
    # plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    # plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    # plt.plot(zmin, TS_omega_JLA, color=c[0], linestyle = 'dotted')
    # # plt.plot(zmin, TS_omega_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    # # plt.plot(zmin, TS_omega_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    # plt.plot(zmin, milne_omega_JLA, linestyle = 'dotted', color=c[1])
    # # plt.plot(zmin, milne_alpha_Panth, linestyle = '-', color=c[1], label = 'Milne')
    # # plt.plot(zmin, milne_alpha_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    # plt.plot(zmin, lamCDM_omega_JLA, color=c[2], linestyle = 'dotted')
    # # plt.plot(zmin, lamCDM_omega_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # # plt.plot(zmin, lamCDM_omega_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    # plt.xlabel('$z_{min}$')
    # plt.ylabel(r'$\Omega_m$')
    # plt.legend()
    # plt.xlim(0,0.1)
    # plt.ylim(0.127,0.141)
    # fv = 0.5*np.sqrt(9-8*TS_omega_JLA) -1
    # ax1 = ax.twinx()
    # ax1.plot(zmin, fv, color = 'k', linestyle = 'dotted',)
    # ax1.set_ylabel(r"$f_{v0}$")
    # #plt.savefig('dark_panth_zac_alpha.pdf', format = 'pdf',dpi = 1200)
    # plt.show()

    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.plot(zmin, TS_beta_JLA, color=c[0], linestyle = '--')
    # plt.plot(zmin, TS_beta_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, TS_beta_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_beta_JLA, linestyle = '--', color=c[1])
    # plt.plot(zmin, milne_beta_Panth, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, milne_beta_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_beta_JLA, color=c[2], linestyle = '--')
    # plt.plot(zmin, lamCDM_beta_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.plot(zmin, lamCDM_beta_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$')
    plt.ylabel(r'$\beta$')
    plt.legend()
    plt.xlim(0,0.1)
    #plt.ylim(3.03,3.21)
    # plt.savefig('dark_panthP_zac_beta.pdf', format = 'pdf', dpi = 1200)
    plt.show()


    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.plot(zmin, TS_x_JLA, color=c[0], linestyle = '--')
    # plt.plot(zmin, TS_x_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, TS_x_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_x_JLA, linestyle = '--', color=c[1])
    # plt.plot(zmin, milne_x_Panth, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, milne_x_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_x_JLA, color= c[2], linestyle = '--')
    # plt.plot(zmin, lamCDM_x_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.plot(zmin, lamCDM_x_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$')
    plt.ylabel(r'$x_{1}$')
    plt.legend()
    plt.xlim(0,0.1)
    #plt.ylim(0.01,0.17)
    # plt.savefig('dark_panthP_zac_x.pdf', format = 'pdf', dpi = 1200)
    plt.show()



    plt.figure(figsize = (9,5.5))
    plt.tight_layout()
    plt.axvline(0.024, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.axvline(0.033, linestyle = 'dotted', alpha = 0.7, color = 'white', linewidth = 1.5)
    plt.plot(zmin, TS_c_JLA, color=c[0], linestyle = '--')
    # plt.plot(zmin, TS_c_Panth, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, TS_c_PanthP, linestyle = '-', color=c[0], label = 'Timescape')
    plt.plot(zmin, milne_c_JLA, color=c[1], linestyle = '--')
    # plt.plot(zmin, milne_c_Panth, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, milne_c_PanthP, linestyle = '-', color=c[1], label = 'Milne')
    plt.plot(zmin, lamCDM_c_JLA, color=c[2], linestyle = '--')
    # plt.plot(zmin, lamCDM_c_Panth, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.plot(zmin, lamCDM_c_PanthP, linestyle = '-', color=c[2], label = 'Spatially Flat $\Lambda$CDM')
    plt.xlabel('$z_{min}$')
    plt.ylabel(r'$c$')
    plt.legend()
    plt.xlim(0,0.1)
    #plt.ylim(-0.027, -0.008)
    # plt.savefig('dark_panthP_zac_c.pdf', format = 'pdf', dpi = 1200)
    plt.show()
# %%
