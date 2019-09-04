import sys
sys.path.append('../../') #takes us to the directory named foreground_functions that houses the foreground_function.py
#and the w3j.npy files

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import foreground_functions as ff
from scipy.optimize import curve_fit, minimize
import emcee
from multiprocessing import Pool


#define the input parameters for a synch only MCMC
A = 2.257e-07
alpha = -2.6
beta = -3.1
gamma = -2.5
ells = np.arange(384) #corresponds to nside of 128
freqs = np.linspace(30, 300, 10)*1.e9
input_params = np.array([A, alpha, beta, gamma])


#load in some data to be fit and do the MCMC on
data = np.load('../../power_spectra/reference_ps.npy') #this is data for 10 freqs, synch only.

#in future we can minimize the chi2 for the initial point of the mcmc.
initial = input_params

# The definition of the log probability function
# We'll also use the "blobs" feature to track the "log prior" for each step
def log_prior(params):
    A, alpha, beta, gamma = params
    if 0 < A < 1.e4 and -6.0 < alpha < 0.0 and -6.0 < beta < 0.0 and -6.0 < gamma < -2.0:
        return 0.0
    return -np.inf

def log_prob(params, ells, freqs, data):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp - ff.chi2_synch(params, ells, freqs, data)

# Initialize the walkers
coords = initial + 1e-4*np.random.randn(32, 4)
nwalkers, ndim = coords.shape

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "tutorial.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(ells, freqs, data), backend=backend)

max_n = 100000

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(coords, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau
