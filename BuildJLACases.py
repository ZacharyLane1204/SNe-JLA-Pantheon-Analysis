# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 21:57:44 2022

@author: zgl12
"""

import numpy as np

def boostz(z,vel,RA0,DEC0,RAdeg,DECdeg):
    # Angular coords should be in degrees and velocity in km/s
    C = 2.99792458e5 # km/s #Light
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vel/C)*costheta*(1+z)

def FinalOutput(data,p):

    '''
    Tully et al 2008. Tully et al reference everything in the local group frame
    and because we are transforming to the local group frame, therefore we use this instead of PLANCK2013 data
    '''
    vcmb = 371.0 # km/s #Velocity boost of CMB
    l_cmb = 264.14 # CMB multipole direction (degrees)
    b_cmb = 48.26 # CMB multipole direction (degrees)
    # converts to
    ra_cmb = 168.0118667 #Right Ascension of CMB
    dec_cmb = -6.98303424 #Declination of CMB
    zcmb1 = boostz(data[:,6], vcmb, ra_cmb, dec_cmb, data[:,7], data[:,8])

    #JLA = np.column_stack((zcmb,mb,x1,c,logMass,survey,zhel,ra,dec)) # The useful JLA data in an array
    Pantheon = np.column_stack((zcmb1, data[:,1], data[:,2], data[:,3], data[:,4], data[:,5], data[:,6], data[:,7], data[:,8]))
    if p == 0:
        file = 'JLA.tsv'
    elif p == 1:
        file = 'JLA_Pantheon.tsv'
    elif p == 2:
        file = 'JLA_PantheonPlus.tsv'
    np.savetxt(file, Pantheon, delimiter='\t', fmt=('%10.7f','%10.7f','%10.7f','%10.7f','%10.7f','%i','%9.7f','%11.7f','%11.7f'))
    return Pantheon

def Covariance():
    COVd = np.load('covmat/stat.npy')
    for i in ['cal', 'model', 'bias', 'dust', 'sigmaz', 'sigmalens', 'nonia']:
        COVd += np.load('covmat/'+i+'.npy')
    return COVd

def JLA():
    '''
    Loads JLA data
    '''
    ndtypes = [('SNIa','S12'), \
               ('zcmb',float), \
               ('zhel',float), \
               ('e_z',float), \
               ('mb',float), \
               ('e_mb',float), \
               ('x1',float), \
               ('e_x1',float), \
               ('c',float), \
               ('e_c',float), \
               ('logMst',float), \
               ('e_logMst',float), \
               ('tmax',float), \
               ('e_tmax',float), \
               ('cov(mb,s)',float), \
               ('cov(mb,c)',float), \
               ('cov(s,c)',float), \
               ('set',int), \
               ('RAdeg',float), \
               ('DEdeg',float), \
               ('bias',float)]

    # width of each column
    delim = (12, 9, 9, 1, 10, 9, 10, 9, 10, 9, 10, 10, 13, 9, 10, 10, 10, 1, 11, 11, 10)

    # load the data
    data = np.genfromtxt('tablef3.dat', delimiter=delim, dtype=ndtypes, autostrip=True) #Loading JLA data

    zcmb = data['zcmb']
    mb = data['mb']
    x1 = data['x1']
    c = data['c']
    logMass = data['logMst'] # log_10_ host stellar mass (in units=M_sun)
    survey = data['set']
    zhel = data['zhel']
    ra = data['RAdeg']
    dec = data['DEdeg']

    # Survey values key:
    #   1 = SNLS (Supernova Legacy Survey)
    #   2 = SDSS (Sloan Digital Sky Survey: SDSS-II SN Ia sample)
    #   3 = lowz (from CfA; Hicken et al. 2009, J/ApJ/700/331
    #   4 = Riess HST (2007ApJ...659...98R)

    JLA = np.column_stack((zcmb,mb,x1,c,logMass,survey,zhel,ra,dec)) # The useful JLA data in an array
    JLASurvey = JLA[:,5] # Survey ID's for the JLA

    return JLA, JLASurvey

def Save():
    '''
    Saving a .txt file of survey names
    '''
    ndtypes = [('SNIa','S12'), \
           ('zcmb',float), \
           ('zhel',float), \
           ('e_z',float), \
           ('mb',float), \
           ('e_mb',float), \
           ('x1',float), \
           ('e_x1',float), \
           ('c',float), \
           ('e_c',float), \
           ('logMst',float), \
           ('e_logMst',float), \
           ('tmax',float), \
           ('e_tmax',float), \
           ('cov(mb,s)',float), \
           ('cov(mb,c)',float), \
           ('cov(s,c)',float), \
           ('set',int), \
           ('RAdeg',float), \
           ('DEdeg',float), \
           ('bias',float)]

    # width of each column
    delim = (12, 9, 9, 1, 10, 9, 10, 9, 10, 9, 10, 10, 13, 9, 10, 10, 10, 1, 11, 11, 10)

    JLA = np.genfromtxt(r'tablef3.dat', delimiter=delim, dtype=ndtypes, autostrip=True)

    ch = ' '
    # Remove all characters after the character '-' from string
    ID = []
    for strValue in JLA:
        strValue = str(strValue)
        strValue = strValue.split(ch, 1)[0]
        ID.append(strValue)

    ID = np.array(ID, dtype = str)

    np.savetxt('jlaID.txt', ID, fmt='%s') # Saving Survey Name into a txt file, for some reason this was the easiest way I could find to do it

def PanthPlus():
    '''
    Loading Pantheon + Data
    '''
    panth = np.genfromtxt(r'PantheonPlusData.txt', dtype=str) # loading the Pantheon + data
    panthSurvey = panth[:,1] #SurveyID for Pantheon +
    #panth = panth.tolist()
    panthSN = panth[:,0] # SNe1a names

    return panthSN, panthSurvey, panth

def Panth():
    '''
    Loading Pantheon Data
    '''
    panth = np.genfromtxt(r'pantheonSNe.txt', dtype=str) # loading the Pantheon + data
    panthSurvey = panth[:,1] #SurveyID for Pantheon +
    #panth = panth.tolist()
    panthSN = panth[:,1] # SNe1a names

    return panthSN, panthSurvey, panth

def RemoveData(panthSN,JLAID):
    '''
    Removes and simplifies Survey ID names for both surveys
    '''
    totalJLA = []
    for j in JLAID:
        j = j[3:] #Removes prefixes and suffix punctuation
        j = j[:-2]
        j = j.lower()
        j = j.replace("sdss","") # Remove SDSS prefix
        j = j.replace(" ","") # Remove spaces
        j = j.replace("sn","") # Remove SN
        totalJLA.append(j)

    panthSN = panthSN.tolist()
    pan = []
    for p in panthSN:
        p = p.lower()
        p = p.replace("sdss","")
        p = p.replace("sn","")
        p = p.replace(" ","") # Removes spaces
        #p = p.replace("asas-","")
        pan.append(p)

    totalJLA = np.array(totalJLA)
    panthSN = np.array(pan)
    return panthSN, totalJLA

def Full_PanthPlus():
    data = np.genfromtxt(r'S:/Documents/Zac_Final_Build/PantheonPlusData.txt', dtype=str) # loading the Pantheon + data

    labels = data[0]



    '''
    SALT Information needed for Calculations
    '''
    c = data[1:,15].astype(float) # SALT2 color
    x1 = data[1:,17].astype(float) # SALT2 stretch
    mb = data[1:,19].astype(float) # SALT2 uncorrected brightness

    '''
    Other Information needed for Calculations
    '''
    host_logmass = data[1:,34].astype(float) # Host Galaxy Log Stellar Mass
    RA = data[1:,26].astype(float) # Right Ascension
    DEC = data[1:,27].astype(float) # Declination
    zHel = data[1:,6].astype(float) # Heliocentric Redshift
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

    '''
    Fixsen et al 1996 Values
    Tully et al 2008 reference everything in the local group frame using Fisxsen et al 1996 values.
    Because we are transforming to the local group frame, therefore we use this instead of PLANCK2013 data
    '''
    vcmb = 371.0 # km/s #Velocity boost of CMB 
    l_cmb = 264.14 # CMB multipole direction (degrees)
    b_cmb = 48.26 # CMB multipole direction (degrees)
    # converts to
    ra_cmb = 168.0118667 #Right Ascension of CMB
    dec_cmb = -6.98303424 #Declination of CMB


    zcmb1 = boostz(zHel, vcmb, ra_cmb, dec_cmb, RA, DEC)

    PantheonPlus = np.column_stack((zcmb1, mb, x1, c, host_logmass, IDSurvey, zHel, RA, DEC))
    np.savetxt(r'S:/Documents/Zac_Final_Build/PantheonPlus.txt', PantheonPlus)


def JLA_Process(panthSN, totalJLA, jla):
    '''
    Removes all the matching names outputs arrays
    '''
    panth_sim = []
    p_ind = 0
    pInd = []
    jInd = []
    for p in panthSN:
        if p in totalJLA:
            panth_sim.append(p)
            pInd.append(p_ind)
            j = totalJLA.tolist().index(p)
            # j_sim.append(j)
            jInd.append(j)
        p_ind += 1

    panth_sim = np.array(panth_sim)

    jInd = list(dict.fromkeys(jInd))
    jInd.sort()
    jInd = np.array(jInd) # JLA indices in common

    COVd = Covariance()

    jL = []
    cov = []
    for j in jInd:
        jL.append(jla[j])
        cov.append(COVd[3*j])
        cov.append(COVd[3*j + 1])
        cov.append(COVd[3*j + 2])

    COVd = np.array(cov)

    c = []
    for j in jInd:
        c.append(COVd[:, 3*j])
        c.append(COVd[:, 3*j + 1])
        c.append(COVd[:, 3*j + 2])

    COVd = np.array(c)
    J_data = np.array(jL)

    # Also manage the covariance matrix here (sort and cut method will work 3i, 3i + 1, 3i + 2)
    # Save covariance as needed in the proper format and put in processed
    # Then Send this to a new build file to do the local group transformation
    # Then use this to do a distance modulus calculation
    # Then send this to compute!!!

    return J_data, COVd

def Process(p):
    if p == 0: # JLA
        output, jlaSurvey = JLA()
        COVd = Covariance()
        np.savetxt('JLA_full_input.txt', output)
        np.savetxt('JLA_full_COVd.txt', COVd)
    elif p == 1: # Pantheon
        Save()
        jla, jlaSurvey = JLA()
        panthSN, panthSurvey, panth = Panth()
        JLAID = np.genfromtxt(r'jlaID.txt', dtype=str) # loading the JLA data
        panthSN, totalJLA = RemoveData(panthSN,JLAID)
        j_data, COVd = JLA_Process(panthSN, totalJLA, jla)
        # JLA data, JLA removed, JLA added
        output = FinalOutput(j_data,p)
        np.savetxt('pantheon_sub_input.txt', output)
        np.savetxt('pantheon_sub_COVd.txt', COVd)
    elif p == 2: # Pantheon Plus
        jla, jlaSurvey = JLA()
        panthSN, panthSurvey, panth = PanthPlus()
        JLAID = np.genfromtxt(r'S:/Documents/Rewrite/jlaID.txt', dtype=str) # loading the JLA data
        panthSN, totalJLA = RemoveData(panthSN,JLAID)
        j_data, COVd = JLA_Process(panthSN, totalJLA, jla)
        print(j_data[:,6])
        # JLA data, JLA removed, JLA added
        output = FinalOutput(j_data,p)
        np.savetxt('pantheonPlus_sub_input.txt', output)
        np.savetxt('pantheonPlus_sub_COVd.txt', COVd)
    elif p ==3: # Full Pantheon+
        output = Full_PanthPlus()
        COVd = 'NONE'
    return output, COVd

p_list = [0,1,2]
for p in p_list:
    Process(p)