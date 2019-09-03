import sys
sys.path.append('../../') #takes us to the directory named foreground_functions that houses the foreground_function.py
#and the w3j.npy files

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import foreground_functions as ff
from scipy.optimize import curve_fit, minimize
import emcee
import corner
from multiprocessing import Pool
#check this filepath
filename = '../mcmc_chains/test_chain.h5' #where the MCMC sampler chain will be written
backend = emcee.backends.HDFBackend(filename)

#define the input parameters for a synch only MCMC
A = 1.7e3
alpha = -3.0
beta = -3.2
gamma = -2.5
ells = np.arange(384) #corresponds to nside of 128
freqs = np.linspace(30, 300, 10)*1.e9
input_params = [A, alpha, beta, gamma]


#load in some data to be fit and do the MCMC on
data = np.load('../../power_spectra/reference_ps.npy') #this is data for 10 freqs, synch only.


#ask for some initialising position for the minimiser to find the ML parameters, then use that as the seed for the MCMC
initial = [1e3, -1., -1., -3.]
soln = minimize(ff.chi2_synch, initial, args=(ells, freqs, data),
                   method='L-BFGS-B', bounds=((None, None), (None, None), (None, None), (-6.0, -2.01)))
#these are the maximum likelihood values
A_ml, alpha_ml, beta_ml, gamma_ml = soln.x
#-------------------------------------------

#define priors
def log_prior(params):
    A, alpha, beta, gamma = params
    if 1. < A < 1.e4 and -6.0 < alpha < 0.0 and -6.0 < beta < 0.0 and -6.0 < gamma < -2.0:
        return 0.0
    return -np.inf

def log_probability(params, ells, freqs, data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp - ff.chi2_synch(params, ells, freqs, data)
#-------------------------------------------

#do the MCMC
#initialise the walkers in a gaussian ball around the max likilihood region.
pos = soln.x + 1e-4*np.random.randn(32, 4)
nwalkers, ndim = pos.shape
backend.reset(nwalkers, ndim)


max_n = 10
#can add pooling to speed up potentially.
#with Pool() as pool:
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(ells, freqs, data), store=True, backend=backend)
sampler.run_mcmc(pos, max_n, store=True)
#saving the MCMC to a file is done through backend
