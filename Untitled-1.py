# %%

import numpy as np
import matplotlib.pyplot as plt


# omega_cut = np.linspace(0,0.65,101)
# omega_cut = omega_cut[1:]

# JLA_lcdm = np.genfromtxt(r'S:/Documents/Zac_Final_Build/JLA_profile_logl_lcdm.txt')
# JLA_ts = np.genfromtxt(r'S:/Documents/Zac_Final_Build/JLA_profile_logl_ts.txt')

# P_ts = np.genfromtxt('Panth_profile_logl_ts.txt')
# P_lcdm = np.genfromtxt('Panth_profile_logl_lcdm.txt')

# PP_lcdm = np.genfromtxt('PanthPlus_profile_logl_lcdm.txt')
# PP_ts = np.genfromtxt('PanthPlus_profile_logl_ts.txt')

FULL_lcdm = np.genfromtxt('FULL_profile_LCDM_LL.txt')
FULL_ts = np.genfromtxt('FULL_profile_TS_LL.txt')

# FULL_lcdm = FULL_lcdm - 2.9979e+12
# FULL_ts = FULL_ts 

min_lc = min(FULL_lcdm)
min_ts = min(FULL_ts)

print('Min LCDM:', min_lc)
print('Min TS:', min_ts)

FULL_lcdm = np.exp(-0.5*FULL_lcdm)
FULL_ts = np.exp(-0.5*FULL_ts)


omega_m = np.linspace(0,0.65,101)
omega_m[0] = 0.001

# fv0 = 0.5*(np.sqrt((9-8*omega_m)) - 1)
fBound = [0.5*(np.sqrt((9-8*min(omega_m))) - 1), 0.5*(np.sqrt((9-8*max(omega_m))) - 1)]
fv0 = np.linspace(fBound[0], fBound[1], 101)

plt.rcParams['font.size']= 16


plt.figure(figsize = (9,5.5))
ax = plt.axes()
lns1 = ax.plot(omega_m, FULL_lcdm, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
plt.ylabel(r'$\mathcal{L}(\Omega_m)$')
plt.xlim(0,0.65)
ax.set_xlabel("$\Omega_{M0}$")
ax1 = ax.twiny()
lns2 = ax1.plot(fv0,FULL_ts, color = 'k', label = 'Timescape')
ax1.set_xlabel("$f_{v0}$")
ax1.set_xlim(max(fv0),min(fv0))
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
# plt.savefig("profile_mu.pdf", dpi = 1200)
plt.show()

FULL_lcdm = FULL_lcdm.tolist()
mxLCDM_FULL = FULL_lcdm.index(max(FULL_lcdm))
mxLCDM_FULL = omega_m[mxLCDM_FULL]

FULL_ts = FULL_ts.tolist()
mxTS_FULL = FULL_ts.index(max(FULL_ts))
mxTS_FULL = fv0[mxTS_FULL]

print('fv0', mxTS_FULL)
print('Om', mxLCDM_FULL)

# plt.figure(figsize = (9,5.5))
# ax = plt.axes()
# lns1 = ax.plot(omega_m, JLA_lcdm, color = 'C1', label = 'Spatially Flat $\Lambda$CDM')
# plt.ylabel(r'$\mathcal{L}(\Omega_m)$')
# plt.xlim(0,0.65)
# ax.set_xlabel("$\Omega_{M0}$")
# ax1 = ax.twiny()
# lns2 = ax1.plot(fv0,JLA_ts, color = 'k', label = 'Timescape')
# ax1.set_xlabel("$f_{v0}$")
# ax1.set_xlim(max(fv0),min(fv0))
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# # plt.savefig("JLA_bayes_fv0_omega.pdf", dpi = 1200)
# plt.show()

# # P_lcdm = P_lcdm.tolist()
# # mxLCDM = P_lcdm.index(max(P_lcdm))
# # mxLCDM = omega_m[mxLCDM]

# # P_ts = P_ts.tolist()
# # mxTS = P_ts.index(max(P_ts))
# # mxTS = fv0[mxTS]



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

# PP_lcdm = PP_lcdm.tolist()
# mxLCDM_PP = PP_lcdm.index(max(PP_lcdm))
# mxLCDM_PP = omega_m[mxLCDM_PP]

# PP_ts = PP_ts.tolist()
# mxTS_PP = PP_ts.index(max(PP_ts))
# mxTS_PP = fv0[mxTS_PP]




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
