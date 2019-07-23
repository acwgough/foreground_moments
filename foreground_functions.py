#-------Import needed packages-------------------------------
import numpy as np
import healpy as hp
from pyshtools.utils import Wigner3j

#---------POWER LAW DEFINITION-----------------------------------------
#defines the ell power law that we will use for the A_s spectrum
def powerlaw(ell, alpha):
    return (ell/ 80.)**alpha

#defines a normal planck distribution
TCMB = 2.7255  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2

def normed_synch(nu, nu0, beta):
    if beta is not np.array:
        beta = np.array(beta)
    return (nu/nu0)**(2.+beta[..., np.newaxis])

def scale_synch(nu, nu0 ,beta):
    unit = normed_synch(nu, nu0,beta) * normed_cmb_thermo_units(nu0) / normed_cmb_thermo_units(nu)
    return unit

#------------------------------------------------------------------------
#-------function to make a map, given a maximum ell
def generate_map(ells_max, A, alpha, beta_sigma, freqs, nu0):
    ells = np.arange(1,ells_max+1)
    #---amplitude part---
    pcls = A * powerlaw(ells, alpha)
    pcls[0] = 0
    #pcls[1] = 1 #don't need this as ells start at 1 (to avoid  division by 0)
    nside = 128
    amp_map = hp.synfast(pcls, nside, new=True, verbose=False)
    check_pcls = hp.anafast(amp_map)
    #---beta map---------
    bcls = beta_sigma * np.ones_like(ells)  #makes a vector [1.5e-6, ... , 1.5e-6] with the same shape as the ells
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) + 3.2)
    #drawn out beta cls
    check_bcls = hp.anafast(beta_map)

    #---composite map----
    sed_scaling_beta = scale_synch(freqs, nu0, beta_map).T

    #make ''realistic maps'' from amp_map*new SED
    newmaps_beta = amp_map * sed_scaling_beta

    return newmaps_beta
#---------------------------------------------------------------------------







#---------------------------------------------------------------------
#---------GET WIGNER SUM PART OF EQUATION 35 FOR 1x1moment-------------
def get_wigner_sum(ell_sum, ell_physical, cls_1, cls_2):
    #ell_sum       == upper limit on sum of ell1 ell2
    #ell_physical  == upper ell physically discernable
    #cls_1,2       == the input c_ell for the amplitude and varition (beta) map
    #order of input for cls_1,2 doesn't matter as is symmetric in both

    #define the relavent ells arrays
    ells = np.arange(1, ell_physical + 1) #the physical ells to be plotted
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
            #----------------------------------------


            #-------maps and normalisation-----------
            #define the normalisation factor in above equation
            factor = (2 * ell1 + 1)*(2 * ell2 + 1)/(4*pi)
            A = cls_1[ell1]
            B = cls_2[ell2]
            #------------------------------------------


            #define wignersum to be the array with the sum of the squares of the wigner coefficients
            wignersum += w3j**2 * factor * A * B

    return wignersum
#---------------------------------------------------------------------
