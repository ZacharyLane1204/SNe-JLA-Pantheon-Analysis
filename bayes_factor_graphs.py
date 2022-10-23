# %% 

import numpy as np
import matplotlib.pyplot as plt
from parameter_MLE import Parameter_Strip as ParS


model_ts = 1
model_lcdm = 2
model_milne = 3

evidence = 0.00001
mle = 13
standard = 2
sample = 'JLA_PanthPlus/'
file_folder = 'LinearSpace/'
prefix = 'PantheonPlus_'

bayes_lin = np.linspace(0,0.1,21)
freq_lin = np.linspace(0,0.1,41)


Q, logZ, imp_logZ, alpha, beta, colour, x1, sig = ParS(prefix,model_ts,evidence,sample,file_folder,mle)

fv0 = np.array(Q, dtype=float)
ts_logZ = np.array(logZ, dtype=float)
ts_imp_logZ = np.array(imp_logZ, dtype=float)
ts_alpha = np.array(alpha, dtype=float)
ts_beta = np.array(beta, dtype=float)
ts_colour = np.array(colour, dtype=float)
ts_x1 = np.array(x1, dtype=float)
sig = np.array(sig, dtype=str)


Q, logZ, imp_logZ, alpha, beta, colour, x1, sig = ParS(prefix,model_lcdm,evidence,sample,file_folder,mle)

Omega_M = np.array(Q, dtype=float)
lcdm_logZ = np.array(logZ, dtype=float)
lcdm_imp_logZ = np.array(imp_logZ, dtype=float)
lcdm_alpha = np.array(alpha, dtype=float)
lcdm_beta = np.array(beta, dtype=float)
lcdm_colour = np.array(colour, dtype=float)
lcdm_x1 = np.array(x1, dtype=float)


imp_pp = (ts_imp_logZ - lcdm_imp_logZ)
log_pp = (ts_logZ - lcdm_logZ)



sample = 'JLA/'
file_folder = 'LinearSpace/'
prefix = 'JLA_'

Q, logZ, imp_logZ, alpha, beta, colour, x1, sig = ParS(prefix,model_ts,evidence,sample,file_folder,mle)

fv0 = np.array(Q, dtype=float)
ts_logZ = np.array(logZ, dtype=float)
ts_imp_logZ = np.array(imp_logZ, dtype=float)
ts_alpha = np.array(alpha, dtype=float)
ts_beta = np.array(beta, dtype=float)
ts_colour = np.array(colour, dtype=float)
ts_x1 = np.array(x1, dtype=float)
sig = np.array(sig, dtype=str)


Q, logZ, imp_logZ, alpha, beta, colour, x1, sig = ParS(prefix,model_lcdm,evidence,sample,file_folder,mle)

Omega_M = np.array(Q, dtype=float)
lcdm_logZ = np.array(logZ, dtype=float)
lcdm_imp_logZ = np.array(imp_logZ, dtype=float)
lcdm_alpha = np.array(alpha, dtype=float)
lcdm_beta = np.array(beta, dtype=float)
lcdm_colour = np.array(colour, dtype=float)
lcdm_x1 = np.array(x1, dtype=float)


imp_jla = (ts_imp_logZ - lcdm_imp_logZ)
log_jla = (ts_logZ - lcdm_logZ)


# plt.style.use('dark_background')
plt.rcParams['font.size']= 16
plt.figure(figsize = (9,5.5))
plt.plot(bayes_lin, imp_pp, linestyle = '-', color = 'k', label = 'P+584')
plt.plot(bayes_lin, imp_jla, color = 'k', linestyle = '--',label = 'JLA')
# plt.plot(bayes_lin, log_pp, color = 'k', linestyle = '--',label = 'Nested Sampling')
plt.ylabel(r'ln($\frac{Z_{ts}}{Z_{\Lambda CDM}}$)')
plt.xlabel(r'$z_{min}$')
plt.fill_between(bayes_lin,0,1,color = 'C0', alpha = 0.08)
plt.fill_between(bayes_lin,1,3,color = 'C0', alpha = 0.15)
plt.fill_between(bayes_lin,0,-1,color = 'C0', alpha = 0.08)
plt.fill_between(bayes_lin,-1,-3,color = 'C0', alpha = 0.15)
plt.xlim(0,0.1)
plt.ylim(-1.5,1.5)
plt.legend()
plt.tight_layout()
plt.savefig('bayes_factor_graph_PanthPlus.pdf', format = 'pdf', dpi = 1200)
plt.show()
# %%