# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:23:33 2022

@author: zgl12
"""

# %%

import numpy as np

def boostz(z,vel,RA0,DEC0,RAdeg,DECdeg):
    # Angular coords should be in degrees and velocity in km/s
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vel/C)*costheta*(1+z)

data = np.genfromtxt(r'S:/Documents/Zac_Final_Build/PantheonPlusData.txt', dtype=str) # loading the Pantheon + data

labels = data[0]

'''
Extra Information not needed for this analysis
'''
candidateID = data[1:,0] # Candidate ID
zHD = data[1:,2].astype(float) # Hubble Diagram Redshift (with CMB and peculiar velocity corrections)
zHD_err = data[1:,3].astype(float) # Hubble Diagram Redshift Uncertainty
mb_corr = data[1:,8].astype(float) # Tripp1998 corrected/standardized m_b magnitude
mb_corr_err_diag = data[1:,9].astype(float) # Tripp1998 corrected/standardized m_b magnitude uncertainty as determined from the diagonal of the covariance matrix. **WARNING, DO NOT FIT COSMOLOGICAL PARAMETERS WITH THESE UNCERTAINTIES. YOU MUST USE THE FULL COVARIANCE. THIS IS ONLY FOR PLOTTING/VISUAL PURPOSES**
mu_SHOES = data[1:,10].astype(float) # Tripp1998 corrected/standardized distance modulus where fiducial SNIa magnitude (M) has been determined from SH0ES 2021 Cepheid host distances.
mu_SHOES_err_diag = data[1:,11].astype(float) # Uncertainty on MU_SH0ES as determined from the diagonal of the covariance matrix. **WARNING, DO NOT FIT COSMOLOGICAL PARAMETERS WITH THESE UNCERTAINTIES. YOU MUST USE THE FULL COVARIANCE. THIS IS ONLY FOR PLOTTING/VISUAL PURPOSES**
cepheid_dist = data[1:,12].astype(float) # cepheid calculated absolute distance to host (uncertainty is incorporated in covariance matrix .cov)
is_calibrator = data[1:,13].astype(float) # binary to designate if this SN is in a host that has an associated cepheid distance
used_in_SHOES_HF = data[1:,14].astype(float) # 1 if used in SH0ES 2021 Hubble Flow dataset. 0 if not included.

host_RA = data[1:,28].astype(float) # Host Galaxy RA
host_DEC = data[1:,29].astype(float) # Host Galaxy DEC
host_angularSep = data[1:,30].astype(float) # Angular separation between SN and host (arcsec)
peculiar_velocity = data[1:,31].astype(float) # Peculiar velocity (km/s)
peculiar_velocity_err = data[1:,32].astype(float) # Peculiar velocity uncertainty (km/s)
milky_way_EBV = data[1:,33].astype(float) # Milky Way E(B-V)
peak_M_julianDate = data[1:,36].astype(float) # Fit Peak Date
peak_M_julianDate_err = data[1:,37].astype(float) # Fit Peak Date Uncertainty
probability_fit = data[1:,40].astype(float) # SNANA Fitprob
mb_corr_err_RAW = data[1:,41].astype(float) # statistical only error on the fitted m_B
mb_corr_err_peculiar_velocity = data[1:,42].astype(float) # peculiar_velocity_err propagated into a magnitude error assuming a fiducial LCDM cosmology
biasCorrection_mb = data[1:,43].astype(float) # Bias correction applied to brightness m_b
biasCorrection_err_mb = data[1:,44].astype(float) # Uncertainty on bias correction applied to brightness m_b
biasCorrection_mb_COVSCALE = data[1:,45].astype(float) #Reduction in uncertainty due to selection effects (multiplicative)
biasCorrection_mb_COVADD = data[1:,46].astype(float) # Uncertainty floor as given by the intrinsic scatter model (quadriture)

'''
SALT Information needed for Calculations
'''
c = data[1:,15].astype(float) # SALT2 color
c_err = data[1:,16].astype(float) # SALT2 color uncertainty
x1 = data[1:,17].astype(float) # SALT2 stretch
x1_err = data[1:,18].astype(float) # SALT2 stretch uncertainty
mb = data[1:,19].astype(float) # SALT2 uncorrected brightness
mb_err = data[1:,20].astype(float) # SALT2 uncorrected brightness uncertainty
x0 = data[1:,21].astype(float) # SALT2 light curve amplitude
x0_err = data[1:,22].astype(float) # SALT2 light curve amplitude uncertainty
COV_x1_c = data[1:,23].astype(float) # SALT2 fit covariance between x1 and c
COV_x1_x0 = data[1:,24].astype(float) # SALT2 fit covariance between x1 and x0
COV_c_x0 = data[1:,25].astype(float) # SALT2 fit covariance between c and x0
NumberDegreesFreedom = data[1:,38].astype(float) # Number of degrees of freedom in SALT2 fit
chi2_fit = data[1:,39].astype(float) # SALT2 fit chi squared

mb_ave = np.mean(mb)

'''
Other Information needed for Calculations
'''
host_logmass = data[1:,34].astype(float) # Host Galaxy Log Stellar Mass
host_logmass_err = data[1:,35].astype(float) # Host Galaxy Log Stellar Mass Uncertainty
RA = data[1:,26].astype(float) # Right Ascension
DEC = data[1:,27].astype(float) # Declination
zCMB = data[1:,4].astype(float) # CMB Corrected Redshift
zCMB_err = data[1:,5].astype(float) # CMB Corrected Redshift Uncertainty
zHel = data[1:,6].astype(float) # Heliocentric Redshift
zHel_err = data[1:,7].astype(float) # Heliocentric Redshift Uncertainty
IDSurvey = data[1:,1].astype(float)
"""
{1:'SDSS',
 4:'SNLS',
 5:'CSP',
 10:'DES',
 15:'PS1MD',
 18:’CNIa0.02’,
 50:'LOWZ/JRK07',
 51:'LOSS1',
 56:'SOUSA',
 57:’LOSS2’,
 61:'CFA1',
 62:'CFA2',
 63:'CFA3S',
 64:'CFA3K',
 65:'CFA4p2',
 66:'CFA4p3',
 100:'HST',
 101:'SNAP',
 106:'CANDELS',
 150:'FOUND'}
"""


dataLabelLength = len(labels)

#labels = np.array(np.hsplit(data, dataLabelLength))


C = 2.99792458e5 # km/s #Light

'''
Fixsen et al 1996 Values
Tully et al 2008 reference everything in the local group frame using Fisxsen et al 1996 values.
Because we are transforming to the local group frame, therefore we use this instead of PLANCK2013 data
'''
vcmb = 371.0 # km/s #Velocity boost of CMB 
# l_cmb = 264.14 # CMB multipole direction (degrees)
# b_cmb = 48.26 # CMB multipole direction (degrees)
# converts to
ra_cmb = 168.0118667 #Right Ascension of CMB
dec_cmb = -6.98303424 #Declination of CMB


zcmb1 = boostz(zHel, vcmb, ra_cmb, dec_cmb, RA, DEC)

PantheonPlus = np.column_stack((zcmb1, mb, x1, c, host_logmass, IDSurvey, zHel, RA, DEC))
np.savetxt(r'S:/ZacPantheonPlus/PantheonPlus.txt', PantheonPlus)

 # %%

#M0, x1, c0





