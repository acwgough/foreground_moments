#=====================================================================
#-------IMPORT NEEDED PACKAGES---------------------------------------
#=====================================================================
import numpy as np
import healpy as hp
from pyshtools.utils import Wigner3j
pi = 3.141592653589793238462

#define default parameters for functions
nside_default = 128
A_default = 1.7e3
alpha_default = -3.0
beta_default = -3.2
beta_sigma_default = 1.5e-6
nu0_default = 2.3e9
ell_max_default = 384


#=====================================================================
#---------FUNCTIONS FOR AMPLITUDE MAP---------------------------------
#=====================================================================
#--------powerlaw function--------------------------------------------
def powerlaw(ell, alpha):
    return (ell/ 80.)**alpha

#defines a normal planck distribution for unit conversion to kelvin
TCMB = 2.7255  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2

def normed_synch(nu, beta, nu0=nu0_default):
    if beta is not np.array:
        beta = np.array(beta)
    return np.power(nu/nu0, 2.+beta[..., np.newaxis])

#-------synch SED given a freqency and a power-------------------------
def scale_synch(nu, beta, nu0=nu0_default):
    unit = normed_synch(nu, beta, nu0=nu0) * normed_cmb_thermo_units(nu0) / normed_cmb_thermo_units(nu)
    return unit



#======================================================================
#---------MAP RELATED FUNCTIONS----------------------------------------
#======================================================================

#------generate an amplitude map---------------------------------------
def generate_map_amp(ell_max=ell_max_default, A=A_default, alpha=alpha_default, nside=nside_default):
    #returns input and output powerspectra, and the map
    ells = np.arange(1,ell_max + 1)
    #---amplitude part---
    pcls = A * powerlaw(ells, alpha)
    pcls[0] = 0
    pcls[1] = 1
    ells = ells - 1 # to shift to starting at 0
    amp_map = hp.synfast(pcls, nside, new=True, verbose=False)
    check_pcls = hp.anafast(amp_map)
    return pcls, check_pcls, amp_map


#-------generate beta map with uniform power spectrum------------------
def generate_map_beta(ell_max=ell_max_default, sigma=beta_sigma_default, beta_0=beta_default, nside=nside_default):
    ells = np.arange(1,ell_max + 1)
    ells = ells - 1
    bcls = beta_sigma * np.ones_like(ells)
    bcls[0] = 0
    bcls[1] = 0
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta_0)
    check_bcls = hp.anafast(beta_map)
    return bcls, check_bcls, beta_map

#-------function to generate series of frequency maps with constant C_ell^beta----
#-------does not return the c_ells, just the maps themselves
def generate_map_full(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, beta_sigma=beta_sigma_default, nu0=nu0_default, nside=nside_default):
    ells = np.arange(1,ell_max + 1)
    #---amplitude part---
    pcls = A * powerlaw(ells, alpha)
    pcls[0] = 0
    pcls[1] = 1
    ells = ells - 1 # to shift to starting at 0
    amp_map = hp.synfast(pcls, nside, new=True, verbose=False)
    #---beta map---------
    bcls = beta_sigma * np.ones_like(ells)
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) + 3.2)               #in future change 3.2 to beta_0 to allow to vary
    #---composite map----
    sed_scaling_beta = scale_synch(freqs, beta_map, nu0).T
    #make ''realistic maps'' from amp_map*new SED
    newmaps_beta = amp_map * sed_scaling_beta
    return newmaps_beta



#-------functions to make many realisations of the same map------------------------
def realisation(N, freqs, ell_max=ell_max_default):
    ells = np.arange(0, ell_max)
    instance = np.zeros((N,len(ells),len(freqs)))
    #instance[0,:,i] picks out the 0th instance of the ith frequency
    for i in range(N):
        maps = generate_map_full(freqs)
        for j in range(len(freqs)):
            instance[i,:,j] = hp.anafast(maps[j])
    return instance



#=====================================================================
#---------MOMENT RELATED FUNCTIONS------------------------------------
#=====================================================================

#---------GET WIGNER SUM PART OF EQUATION 35 FOR 1x1moment-------------
def get_wigner_sum(ell_sum, cls_1, cls_2):
    #ell_sum       == upper limit on sum of ell1 ell2
    #ell_physical  == upper ell physically discernable
    #cls_1,2       == the input c_ell for the amplitude and varition (beta) map
    #order of input for cls_1,2 doesn't matter as is symmetric in both
    ells_ext = np.arange(1, ell_sum + 1)  #the ells to be summed over
    #define an empty array to store the wigner sum in
    wignersum = np.zeros_like(ells_ext, dtype=float)
    #begin the sum (leave off final element so the shapes work out)
    for ell1 in ells_ext[:ell_sum-1]:
        for ell2 in ells_ext[:ell_sum-1]:
            w3j, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
            avaliable_ells = np.arange(ellmin, ellmax+1)
            #this block forces all the w3j arrays to have the same size as the wignersum array
            #cut off trailing zeros at the end of w3j
            max_nonzero_index = ellmax - ellmin
            w3j = w3j[:max_nonzero_index + 1]
            #make the w3j array the same shape as the wignersum array
            if len(w3j) < len(ells_ext):
                #if the w3j is shorter than the input ells, then pad to the end with zeros
                padding = np.zeros(len(wignersum)-len(w3j))
                w3j = np.append(w3j, padding)
            else:
                w3j=w3j
            #roll stuff into position and relabel those that roll ''around'' to 0
            w3j = np.roll(w3j, ellmin)
            w3j[:ellmin] = np.zeros_like(w3j[:ellmin])
            #cut to size of the moment that we're adding (the size of the ells matrix)
            w3j = w3j[:len(ells_ext)]

            #define wignersum to be the array with the sum of the squares of the wigner coefficients
            wignersum += w3j**2 * (2 * ell1 + 1)*(2 * ell2 + 1)/(4*pi) * cls_1[ell1] * cls_2[ell2]

    return wignersum
#---------------------------------------------------------------------
