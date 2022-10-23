# SNe-JLA-Pantheon-Analysis


Based on "Apparent cosmic acceleration from type Ia supernovae''
Dam, Heinesen, Wiltshire (2017) arxiv:1706.07236

This code requires the Multinest module (arXiv:0809.3437) and the Python interface PyMultinest to be installed ()

JLA dataset and covariance matrices used in this analysis can be downloaded from: 
http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/568/A22
and http://supernovae.in2p3.fr/sdss_snls_jla/covmat_v6.tgz.

Pantheon dataset used in this analysis can be downloaded from:

https://github.com/dscolnic/Pantheon/blob/master/data_fitres/Ancillary_G10.FITRES

Pantheon+ dataset can be downloaded from:

https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

Dataset used in analysis computes redshifts in CMB frame from JLA heliocentric redshifts. 

The Pymultinest can only be installed on linux systems, so for running the bayesian nested code, ensure you are running linux and Python 2.7

## To start the analysis:
Run 'python BuildJLACases.py' (Py3) to generate the data file, full data sets and the covariance matrices used in this analysis ordered as follows

zcmb, mb, x1, c, logMass, survey id, zhel, RA, DEC

For the full Pantheon+ dataset, run 'python pantheonLoad.py' to generate the data set as 'PantheonPlus.txt'.

Next, run 'python 3_distmod.py' (Py3) to produce an interpolation table of luminosity distances for each SNe Ia 
and for different cosmological parameter(s).

For the frequentist approach, run 'python freq_loop_code.py' (Py3) to output text files for the parameters and use 'python freq_code_analysis.py' (Py3) to graph and interpret these text files. 


For the Bayesian approach
------------------------------------------------------------
Running the main script BayesAnalysis.py computes the evidence 
for the model specified by the command line options

model = int(sys.argv[1])    # 1=Timescape, 2=Empty, 3=Flat ; 

z_cut = float(sys.argv[2])  # redshift cut e.g. 0.033 ;

zdep = int(sys.argv[3])     # redshift dependence in mean stretch and colour distributions (0 or 1) ;

case = int(sys.argv[4])     # redshift light curve model (1-8) ;

nsigma = int(sys.argv[5])   # 1, 2 or 3 sigma omega/fv prior ;

nlive = int(sys.argv[6])    # number of live points used in sampling ;

tol = float(sys.argv[7])    # stop evidence integral when next contribution less than tol ;

I used each model and a range of redshift cuts between 0 and 0.1, with 0, 1, 2, 1000, 0.00001 as my other parameter choices ;

python snsample.py 1 0.033 0 1 2 1000 0.1 is and example line of code

Graphing codes:

'480Histograms.py' for the histogram seen in Fig.~1.3

'bayes_factor_graphs.py' For the Bayesian Evidence plots

'omega_likelihood_graphs.py' for the omega likelihood graphs

'graphAnalysis.py' for the Bayesian parameter plotting and frequentist on the same graph, this relies on 'parameter_MLE.py' and 'freq_code_analysis.py'

MAKE SURE TO CHANGE INPUT DATA FILES BEFORE USING THIS CODE!
