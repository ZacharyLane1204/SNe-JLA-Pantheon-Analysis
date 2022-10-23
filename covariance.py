
# %%

import numpy as np
from scipy import linalg

cond = 0.05


cov = np.genfromtxt('PPLUS_MU_SAMPLE_COVD.txt')
# cov = cov[1:]

# vpec_cov = np.genfromtxt('PantheonSH0ES_122221_VPEC.cov.txt')
# vpec_cov = vpec_cov[1:]

# n = 1701

# cov = cov.reshape((n,n))
# vpec_cov = vpec_cov.reshape((n,n))

corr_cov = cov #- vpec_cov
prefix = 'zero_case_'

# vp = False

# if vp == True:
#     corr_cov = cov - vpec_cov
#     prefix = 'vpec_'

# cov_inv = linalg.inv(cov)
# np.savetxt('cov_inv.txt', cov_inv)

data = np.genfromtxt('PPLUS_MU_SAMPLE.txt')
full = np.genfromtxt('PPLUS_MU_SAMPLE_PanthData.txt')
# full = full[1:]

data1 = [item for item in data[:,0] if item >= cond]


panth_sim = []
p_ind = 0
pInd = []
for p in data[:,0]:
    if p in data1:
        panth_sim.append(p)
        pInd.append(p_ind)
    p_ind += 1

zz = data[pInd]

cov33 = corr_cov[pInd]
cov33 = cov33[:,pInd]
full = full[pInd]


# cov33 = np.tril(cov33) + np.triu(cov33.T, 1)
cov33_inv = linalg.inv(cov33)
# cov33_inv = np.tril(cov33_inv) + np.triu(cov33_inv.T, 1)


cov33_inv = linalg.inv(cov33)
np.savetxt(prefix + 'cov33.txt', cov33)
np.savetxt(prefix + 'PPLUS_MU_SAMPLE_33.txt', zz)
np.savetxt(prefix + 'PPLUS_MU_SAMPLE_COVd_33.txt', cov33_inv)
np.savetxt(prefix + 'PPLUS_MU_SAMPLE_PanthData_33.txt', full)


# %%
