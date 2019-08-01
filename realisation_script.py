#========================================================
#Write script to calculate multiple realisations for different parameters and
#store the mean power spectrum. This way we can try to avoid cosmic variance in the
#residuals at low ell.
import numpy as np
import healpy as hp
import foreground_functions as ff
import matplotlib.pyplot as plt
import time
import datetime


#-------uninteresting paramters to vary---------------
freqs = np.linspace(30,300,4)*1.e9
nside = 128
ell_max = 3 * nside
nu0 = 95e9 #to be able to converge between 30 and 300 GHz we should choose 95 GHz for best convergences
A = 1.7e3
#for power law beta
crit = 2/np.log(10)


#-------parameters to vary in realisation-------------------
#can change this so that we take inputs for parameters
# N = eval(input('Number of realisations = '))
# alpha = eval(input('alpha = '))
N = 100
alpha = -3.0
beta = -3.2
gamma = -2.1 #must be <-2 for convergence
sigma = crit/3 #beta map has this standard deviation

#time scales linearly with N
#N=10 takes ~7 sec so N=100~1minute
start=time.time()
x = ff.realisation_power(N, freqs, ell_max=ell_max, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta_0=beta, nu0=nu0, nside=nside)
#output is an (N, ell_max, freqs) array.
#get mean powerspectrum by np.mean(x,0)
#get std at each ell by np.std(x,0)

#use savez if we want to store multiple arrays, but use save if just one.
#if kwargs not given in savez then arrays assigned arr_0, arr_1 and are accessed like dictionary.
np.save('realisation', x)
np.savez('realisation', x)
total_time = time.time()-start
if time < 60.0:
    print('time='+str(np.round(total_time)+' sec.'))
else:
    print('time='+str(datetime.timedelta(seconds=total_time)))
