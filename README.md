# SNe-JLA-Pantheon-Analysis


Based on "Apparent cosmic acceleration from type Ia supernovae''
Dam, Heinesen, Wiltshire (2017) arxiv:1706.07236

This code requires the Multinest module (arXiv:0809.3437) and the Python interface PyMultinest to be installed ()

JLA dataset and covariance matrices used in this analysis can be downloaded from
http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/568/A22
and http://supernovae.in2p3.fr/sdss_snls_jla/covmat_v6.tgz.

Dataset used in analysis computes redshifts in CMB frame from JLA heliocentric redshifts. Running 'python build.py' 
generates the data file jla.tsv used in this analysis ordered as follows

zcmb, mb, x1, c, logMass, survey id, zhel, RA, DEC

Next, for fast likelihood evaluation, run 'python distmod.py' to produce a look up table of luminosity distances for each SNe Ia 
and for different cosmological parameter(s).

------------------------------------------------------------
Running the main script snsample.py computes the evidence 
for the model specified by the command line options

model = int(sys.argv[1])    # 1=Timescape, 2=Empty, 3=Flat ; 

z_cut = float(sys.argv[2])  # redshift cut e.g. 0.033 ;

zdep = int(sys.argv[3])     # redshift dependence in mean stretch and colour distributions (0 or 1) ;

case = int(sys.argv[4])     # redshift light curve model (1-8) ;

nsigma = int(sys.argv[5])   # 1, 2 or 3 sigma omega/fv prior ;

nlive = int(sys.argv[6])    # number of live points used in sampling ;

tol = float(sys.argv[7])    # stop evidence integral when next contribution less than tol ;

I used each model and a range of redshift cuts between 0 and 0.1, with 0, 1, 2, 1000, 0.1 as my other parameter choices ;

python jlaAnalysis.py 1 0.033 0 1 2 1000 0.1 is and example line of code
