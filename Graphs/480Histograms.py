# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:09:43 2022

@author: porri
"""

import pandas as pd
import numpy as np
#import pdflatex
import matplotlib
import matplotlib.pyplot as plt

def Histograms(p_zHD,p_x1,p_c,j_zHD,j_x1,j_c,pa_zHD,pa_x1,pa_c,u_zHD,u_x1,u_c):

    '''
    plt.figure(figsize=(12,8))
    plt.scatter(np.log10(p_zHD), p_x1, s= 30, color = 'black', alpha = 1, label = 'Pantheon$+$', edgecolor='none')
    plt.scatter(np.log10(pa_zHD), pa_x1, s= 30, marker = 'D', color = 'C1', alpha = 0.7, label = 'Pantheon', edgecolor='none')
    plt.scatter(np.log10(j_zHD), j_x1, s= 45, marker = '+', color = 'C0', alpha = 0.7, label = 'JLA', edgecolor='none')
    # plt.scatter(np.log10(u_zHD), u_x1, color = 'r', alpha = 0.3, label = 'Unique to JLA', edgecolor='black')
    plt.xlabel('log$_{10}(z_{Helio})$')
    plt.ylabel('$x_1$')
    plt.legend()
    plt.show()
    '''
    
    fig,ax = plt.subplots(2,1,figsize=(12,8),sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    plt.tight_layout()
    #plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    ax[0].scatter(np.log10(p_zHD), p_x1, s= 30, color = 'black', alpha = 1, label = 'Pantheon$+$', edgecolor='none')
    ax[0].scatter(np.log10(pa_zHD), pa_x1, s= 30, marker = 'D', color = 'C1', alpha = 0.7, label = 'Pantheon', edgecolor='none')
    ax[0].scatter(np.log10(j_zHD), j_x1, s= 45, marker = '+', color = 'C0', alpha = 0.7, label = 'JLA', edgecolor='none')
    ax[1].set_xlabel("log$_{10}(z_{CMB})$", fontsize=20)
    ax[1].set_ylabel("Counts", fontsize=20)
    ax[1].hist(np.log10(p_zHD), bins = 40, color = 'black', alpha = 0.7)
    ax[1].hist(np.log10(pa_zHD), bins = 40, color = 'C1', alpha = 0.7)
    ax[1].hist(np.log10(j_zHD), bins = 40, color = 'C0', alpha = 0.7)
    ax[0].set_ylabel("$x_1$", fontsize=20)
    ax[1].tick_params(axis='x', labelsize= 18)
    ax[0].tick_params(axis='y', labelsize= 18)
    ax[1].tick_params(axis='y', labelsize= 18)
    ax[0].legend(prop={'size': 18})
    plt.savefig("scatter_pplus_p_jla.pdf", format="pdf", bbox_inches="tight", dpi=1200)
    
    #plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
    p_bin_middles_c, p_bin_probability_c, p_bin_width_c = Bar(p_c)
    j_bin_middles_c, j_bin_probability_c, j_bin_width_c = Bar(j_c)
    pa_bin_middles_c, pa_bin_probability_c, pa_bin_width_c = Bar(pa_c)
    # u_bin_middles_c, u_bin_probability_c, u_bin_width_c = Bar(u_c)
    plt.figure()
    #plt.hist(c, bins = 20, color = 'limegreen')
    plt.step(p_bin_middles_c, p_bin_probability_c, color = 'black', label = 'Pantheon$+$')
    plt.step(pa_bin_middles_c, pa_bin_probability_c, color = 'C1', label = 'Pantheon')
    plt.step(j_bin_middles_c, j_bin_probability_c, color = 'C0', label = 'JLA')
    # plt.bar(u_bin_middles_c, u_bin_probability_c, width=u_bin_width_c, color = 'lawngreen', label = 'Unique to JLA', edgecolor='black')
    plt.xlabel('Colour')
    plt.ylabel('Probability of being in that range')
    plt.legend()
    # plt.savefig("colour_pplus_p_jla.pdf", format="pdf", bbox_inches="tight", dpi=1200)
    plt.show()
    
    p_bin_middles_x1, p_bin_probability_x1, p_bin_width_x1 = Bar(p_x1)
    j_bin_middles_x1, j_bin_probability_x1, j_bin_width_x1 = Bar(j_x1)
    pa_bin_middles_x1, pa_bin_probability_x1, pa_bin_width_x1 = Bar(pa_x1)
    # u_bin_middles_x1, u_bin_probability_x1, u_bin_width_x1 = Bar(u_x1)
    plt.figure()
    #plt.hist(x1, bins = 20, color = 'royalblue')
    plt.step(p_bin_middles_x1, p_bin_probability_x1, color = 'black', label = 'Pantheon$+$')
    plt.step(pa_bin_middles_x1, pa_bin_probability_x1, color = 'C1', label = 'Pantheon')
    plt.step(j_bin_middles_x1, j_bin_probability_x1, color = 'C0', label = 'JLA')
    # plt.bar(u_bin_middles_x1, u_bin_probability_x1, width=u_bin_width_x1, color = 'dodgerblue', label = 'Unique to JLA', edgecolor='black')
    plt.xlabel('Stretch')
    plt.ylabel('Probability of being in that range')
    plt.legend()
    # plt.savefig("x1_pplus_p_jla.pdf", format="pdf", bbox_inches="tight", dpi=1200)
    plt.show()
    
    p_bin_middles_zHD, p_bin_probability_zHD, p_bin_width_zHD = Bar(p_zHD)
    j_bin_middles_zHD, j_bin_probability_zHD, j_bin_width_zHD = Bar(j_zHD)
    pa_bin_middles_zHD, pa_bin_probability_zHD, pa_bin_width_zHD = Bar(pa_zHD)
    # u_bin_middles_zHD, u_bin_probability_zHD, u_bin_width_zHD = Bar(u_zHD)
    plt.figure()
    #plt.hist(zHD, bins = 20, color = 'lightcoral')
    plt.step(p_bin_middles_zHD, p_bin_probability_zHD, color = 'black', label = 'Pantheon$+$')
    plt.step(pa_bin_middles_zHD, pa_bin_probability_zHD, color = 'C1', label = 'Pantheon')
    plt.step(j_bin_middles_zHD, j_bin_probability_zHD, color = 'C0', label = 'JLA')
    # plt.bar(u_bin_middles_zHD, u_bin_probability_zHD, width=u_bin_width_zHD, color = 'tomato', label = 'Unique to JLA', edgecolor='black')
    plt.xlabel('Heliocentric Redshift')
    plt.ylabel('Probability of being in that range')
    plt.legend()
    # plt.savefig("zhel_pplus_p_jla.pdf", format="pdf", bbox_inches="tight", dpi=1200)
    plt.show()

def Bar(x):
    x, bin_edges = np.histogram(x, bins=20)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = x/float(x.sum())
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot

    return bin_middles, bin_probability, bin_width

def PanthPlus():
    table = np.genfromtxt('S:/ZacPantheonPlus/PantheonPlusData.txt')
    
    zHD = table[:,2]
    zHD = zHD[1:]
    
    zcmb = table[:,4]
    zcmb = zcmb[1:]
    
    x1 = table[:,17]
    x1 = x1[1:]
    
    c = table[:,15]
    c = c[1:]
    RA = table[1:,26].astype(float) # Right Ascension
    DEC = table[1:,27].astype(float) # Declination
    
    return zHD, x1, c, zcmb, RA, DEC

def Panth():
    table = np.genfromtxt('S:/ZacPantheonPlus/pantheonSNe.txt')

    zHD = table[:,9]
    
    x1 = table[:,20]
    
    c = table[:,22]
    zcmb = table[:,7]

    ra = table[:,34]
    dec = table[:,35]

    return zHD, x1, c, zcmb, ra, dec

def JLA():
    ndtypes = [('SNIa','S12'), ('zcmb',float), 
               ('zhel',float), ('e_z',float), 
               ('mb',float), ('e_mb',float), 
               ('x1',float), ('e_x1',float), 
               ('c',float), ('e_c',float), 
               ('logMst',float), ('e_logMst',float), 
               ('tmax',float), ('e_tmax',float), 
               ('cov(mb,s)',float), ('cov(mb,c)',float), 
               ('cov(s,c)',float), ('set',int), 
               ('RAdeg',float), ('DECdeg',float),
               ('bias',float)]

    delim = (12, 9, 9, 1, 10, 9, 10, 9, 10, 9, 10, 10, 13, 9, 10, 10, 10, 1, 11, 11, 10)

    data = np.genfromtxt('tablef3.dat', delimiter=delim, dtype=ndtypes, autostrip=True)

    x1 = data['x1']
    c  = data['c']
    zhel = data['zhel']
    zcmb = data['zcmb']
    ra = data['RAdeg']
    dec = data['DECdeg']
    return zhel, x1, c, zcmb, ra, dec


def boostz(z,vel,RA0,DEC0,RAdeg,DECdeg):
    # Angular coords should be in degrees and velocity in km/s
    C = 2.99792458e5 # km/s #Light
    RA = np.radians(RAdeg)
    DEC = np.radians(DECdeg)
    RA0 = np.radians(RA0)
    DEC0 = np.radians(DEC0)
    costheta = np.sin(DEC)*np.sin(DEC0) + np.cos(DEC)*np.cos(DEC0)*np.cos(RA-RA0)
    return z + (vel/C)*costheta*(1+z)


p_zHD, p_x1, p_c, p_cmb, p_ra, p_dec = PanthPlus()
pa_zHD, pa_x1, pa_c, pa_cmb, pa_ra, pa_dec = Panth()
j_zHD, j_x1, j_c, j_cmb, j_ra, j_dec = JLA()


vcmb = 371.0 # km/s #Velocity boost of CMB
l_cmb = 264.14 # CMB multipole direction (degrees)
b_cmb = 48.26 # CMB multipole direction (degrees)
# converts to
ra_cmb = 168.0118667 #Right Ascension of CMB
dec_cmb = -6.98303424 #Declination of CMB


pa_cmb = boostz(pa_zHD, vcmb, ra_cmb, dec_cmb, pa_ra, pa_dec)
j_cmb = boostz(j_zHD, vcmb, ra_cmb, dec_cmb, j_ra, j_dec)
p_cmb = boostz(p_zHD, vcmb, ra_cmb, dec_cmb, p_ra, p_dec)

#JLA = np.column_stack((zcmb,mb,x1,c,logMass,survey,zhel,ra,dec)) # The useful JLA data in an array

palist = []
plist = []
jlist = []

for p in p_cmb:
    if 0.033 < p < 0.071:
        plist.append(p)
for pa in pa_cmb:
    if 0.033 < pa < 0.071:
        palist.append(pa)
        
for j in j_cmb:
    if 0.033 < j < 0.071:
        jlist.append(j)

# u_zHD, u_x1, u_c = UniqueJLA()

#Histograms(p_zHD,p_x1,p_c,j_zHD,j_x1,j_c,u_zHD,u_x1,u_c)
Histograms(p_cmb,p_x1,p_c,j_cmb,j_x1,j_c,pa_cmb,pa_x1,pa_c,0,0,0)
