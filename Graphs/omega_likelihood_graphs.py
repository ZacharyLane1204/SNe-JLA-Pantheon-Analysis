# %% 

import numpy as np
import matplotlib.pyplot as plt


# omega_cut = np.linspace(0,0.65,101)
# omega_cut = omega_cut[1:]

JLA_lcdm = np.genfromtxt(r'S:/Documents/Zac_Final_Build/JLA_profile_logl_lcdm.txt')
JLA_ts = np.genfromtxt(r'S:/Documents/Zac_Final_Build/JLA_profile_logl_ts.txt')

P_ts = np.genfromtxt('Panth_profile_logl_ts.txt')
P_lcdm = np.genfromtxt('Panth_profile_logl_lcdm.txt')

PP_lcdm = np.genfromtxt('PanthPlus_profile_logl_lcdm.txt')
PP_ts = np.genfromtxt('PanthPlus_profile_logl_ts.txt')

omega_m = np.linspace(0,0.65,101)
omega_m[0] = 0.001

# fv0 = 0.5*(np.sqrt((9-8*omega_m)) - 1)
fBound = [0.5*(np.sqrt((9-8*min(omega_m))) - 1), 0.5*(np.sqrt((9-8*max(omega_m))) - 1)]
fv0 = np.linspace(fBound[0], fBound[1], 101)

plt.rcParams['font.size']= 16

om1 = omega_m[:51].tolist()
om2 = omega_m[52:].tolist()

om = om1 + om2
j1 = JLA_lcdm[:51].tolist()
j2 = JLA_lcdm[52:].tolist()

om = om1+om2
j = j1 + j2

plt.figure(figsize = (9,5.5))
ax = plt.axes()
lns1 = ax.plot(om, j, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
plt.ylabel(r'$\mathcal{L}(\Omega_m)$')
plt.xlim(0,0.65)
ax.set_xlabel("$\Omega_{M0}$")
ax1 = ax.twiny()
lns2 = ax1.plot(fv0,JLA_ts, color = 'k', label = 'Timescape')
ax1.set_xlabel("$f_{v0}$")
ax1.set_xlim(max(fv0),min(fv0))
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.savefig("JLA_fv0_omega_like.pdf", dpi = 1200)
plt.show()

# P_lcdm = P_lcdm.tolist()
# mxLCDM = P_lcdm.index(max(P_lcdm))
# mxLCDM = omega_m[mxLCDM]

# P_ts = P_ts.tolist()
# mxTS = P_ts.index(max(P_ts))
# mxTS = fv0[mxTS]
# print(mxTS)
# print(mxLCDM)



# plt.figure(figsize = (9,5.5))
# ax = plt.axes()
# lns1 = ax.plot(omega_m, PP_lcdm, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
# plt.ylabel(r'$\mathcal{L}(\Omega_m)$')
# plt.xlim(0,0.65)
# ax.set_xlabel("$\Omega_{M0}$")
# ax1 = ax.twiny()
# lns2 = ax1.plot(fv0,PP_ts, color = 'k', label = 'Timescape')
# ax1.set_xlabel("$f_{v0}$")
# ax1.set_xlim(max(fv0),min(fv0))
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# # plt.savefig("PanthPlus_profile_like.pdf", dpi = 1200)
# plt.show()

# JLA_lcdm = np.log(max(JLA_lcdm))
# print(JLA_lcdm)
# JLA_lcdm = JLA_lcdm.tolist()
# mxLCDM_PP = JLA_lcdm.index(max(JLA_lcdm))
# mxLCDM_PP = omega_m[mxLCDM_PP]

# JLA_ts = JLA_ts.tolist()
# mxTS_PP = JLA_ts.index(max(JLA_ts))
# mxTS_PP = fv0[mxTS_PP]

# print(mxTS_PP)
# print(mxLCDM_PP)




# plt.figure(figsize = (9,5.5))
# ax = plt.axes()
# lns1 = ax.plot(omega_m, P_lcdm, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
# plt.ylabel(r'$\mathcal{L}(\Omega_m)$')
# plt.xlim(0,0.65)
# ax.set_xlabel("$\Omega_{M0}$")
# ax1 = ax.twiny()
# lns2 = ax1.plot(fv0,P_ts, color = 'k', label = 'Timescape')
# ax1.set_xlabel("$f_{v0}$")
# ax1.set_xlim(max(fv0),min(fv0))
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=2)
# # plt.savefig("Panth_profile_like.pdf", dpi = 1200)
# plt.show()

# P_lcdm = P_lcdm.tolist()
# mxLCDM_P = P_lcdm.index(max(P_lcdm))
# mxLCDM_P = omega_m[mxLCDM_P]

# P_ts = P_ts.tolist()
# mxTS_P = P_ts.index(max(P_ts))
# mxTS_P = fv0[mxTS_P]

# %%
