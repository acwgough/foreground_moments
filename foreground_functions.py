#=====================================================================
#-------IMPORT NEEDED PACKAGES---------------------------------------
#=====================================================================
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from pyshtools.utils import Wigner3j
from math import pi
import scipy.special as sp #for the zeta function sp.zeta() in the 0x2 term

#define default parameters for functions
nside_default = 128
ell_max_default = 3*nside_default
A_default = 1.7e3
alpha_default = -3.0
beta_default = -3.2
beta_sigma_default = 1.5e-6
nu0_default = 95e9 #to be able to converge between 30 and 300 GHz we should choose 95 GHz for best convergences


#for power law beta
A_beta_default = 1e-6
gamma_default = -2.1 #must be less than -2 for convergence in 0x2 term


#=====================================================================
#---------FUNCTIONS FOR AMPLITUDE MAP---------------------------------
#=====================================================================
#--------powerlaw function--------------------------------------------
def powerlaw(ell, alpha):
    #set this to power law everything except the first two elements, and set the
    #first two elements to 0
    power = np.zeros(len(ell))
    power[2:] = (ell[2:]/ 80.)**alpha
    power[0] = 0
    power[1] = 0
    return power


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
    return (nu/nu0)**(2.+beta[..., np.newaxis])

#-------synch SED given a freqency and a power-------------------------
def scale_synch(nu, beta, nu0=nu0_default):
    unit = normed_synch(nu, beta, nu0=nu0) * normed_cmb_thermo_units(nu0) / normed_cmb_thermo_units(nu)
    return unit



#======================================================================
#---------MAP RELATED FUNCTIONS----------------------------------------
#======================================================================

#------generate an amplitude map---------------------------------------
def map_amp(ell_max=ell_max_default, A=A_default, alpha=alpha_default, nside=nside_default):
    #returns input and output powerspectra, and the map
    ells = np.arange(0,ell_max)
    pcls = A * powerlaw(ells, alpha)
    amp_map = hp.synfast(pcls, nside, new=True, verbose=False)
    #check_pcls = hp.anafast(amp_map)
    return pcls, amp_map


#-------generate beta map with uniform power spectrum------------------
def map_beta(ell_max=ell_max_default, sigma=beta_sigma_default, beta_0=beta_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    bcls = sigma * np.ones_like(ells)
    bcls[0] = 0
    bcls[1] = 0
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta_0)
    # check_bcls = hp.anafast(beta_map)
    return bcls, beta_map


#--------generate power law beta----------------------------------------
def map_power_beta(ell_max=ell_max_default, A_beta=A_beta_default, gamma=gamma_default, beta_0=beta_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    bcls = A_beta * powerlaw(ells, gamma)
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta_0)
    # check_bcls = hp.anafast(beta_map)
    return bcls, beta_map



#-------function to generate series of frequency maps with constant C_ell^beta/white noise----
#-------does not return the c_ells, just the maps themselves
def map_full_white(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, beta_sigma=beta_sigma_default, beta_0=beta_default, nu0=nu0_default, nside=nside_default):
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_beta(ell_max=ell_max, sigma=beta_sigma, beta_0=beta_0, nside=nside)
    sed_scaling_beta = scale_synch(freqs, beta_map, nu0=nu0).T
    #make realistic maps
    newmaps_beta = amp_map * sed_scaling_beta
    return newmaps_beta


#-------function to generate series of frequency maps with power spectrum for beta_map-------
def map_full_power(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, A_beta=A_beta_default, gamma=gamma_default, beta_0=beta_default, nu0=nu0_default, nside=nside_default):
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_power_beta(ell_max=ell_max, A_beta=A_beta, gamma=gamma, beta_0=beta_0, nside=nside)
    sed_scaling_beta = scale_synch(freqs, beta_map, nu0=nu0).T
    #make realistic maps
    newmaps_beta = amp_map * sed_scaling_beta
    return newmaps_beta




#-------function to make many realisations of the same white noise map------------------------
def realisation(N, freqs, ell_max=ell_max_default):
    ells = np.arange(0, ell_max)
    instance = np.zeros((N,len(ells),len(freqs)))
    #instance[0,:,i] picks out the 0th realisation of the ith frequency
    for i in range(N):
        maps = map_full_white(freqs)
        for j in range(len(freqs)):
            instance[i,:,j] = hp.anafast(maps[j])
    return instance


#--------function to make many realisations of the same power map----------------------
def realisation_power(N, freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, A_beta=A_beta_default, gamma=gamma_default, beta_0=beta_default, nu0=nu0_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    realisation = np.zeros((N, len(ells), len(freqs)))
    for i in range(N):
        maps = map_full_power(freqs, ell_max=ell_max, A=A, alpha=alpha, A_beta=A_beta, gamma=gamma, beta_0=beta_0, nu0=nu0, nside=nside)
        for j in range(len(freqs)):
            realisation[i,:,j] = hp.anafast(maps[j])
    return realisation


#=====================================================================
#---------MOMENT RELATED FUNCTIONS------------------------------------
#=====================================================================

#---------DEFINE FUNCTION FOR 0X0 AUTO--------------------------------
#from paper we know that 0x0 is SED^2 C_amp^2
#this gives a set of 0x0 moments at different frequencies
def auto0x0(freqs, beta_0=beta_default, ell_max=ell_max_default, A=A_default, alpha=alpha_default, nu0=nu0_default):
    sed_scaling = scale_synch(freqs, beta_0, nu0=nu0)
    ells = np.arange(0,ell_max)
    pcls = A * powerlaw(ells, alpha)
    moment0x0 = np.zeros((len(freqs),len(ells)))
    for i in range(len(moment0x0[:])):
        moment0x0[i] = pcls * sed_scaling[i]**2
    return moment0x0

#---------GET WIGNER SUM PART OF EQUATION 35 FOR 1x1moment-------------
def get_wigner_sum(ell_sum, amp_cls, beta_cls):
    #ell_sum       == upper limit on sum of ell1 ell2
    #ell_physical  == upper ell physically discernable
    #cls_1,2       == the input c_ell for the amplitude and varition (beta) map
    #order of input for cls_1,2 doesn't matter as is symmetric in both
    ells_ext = np.arange(0, ell_sum)  #the ells to be summed over
    #define an empty array to store the wigner sum in
    wignersum = np.zeros_like(ells_ext, dtype=float)
    #begin the sum (leave off final element so the shapes work out)
    for ell1 in ells_ext[:ell_sum]:
        for ell2 in ells_ext[:ell_sum]:
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
            #roll stuff into position and relabel those that roll ''around'' to 0. Using concatenate here
            #as it is faster than np.roll.
            w3j = np.concatenate([w3j[-ellmin:],w3j[:-ellmin]])
            w3j[:ellmin] = np.zeros_like(w3j[:ellmin])
            #cut to size of the moment that we're adding (the size of the ells matrix)
            w3j = w3j[:len(ells_ext)]

            #define wignersum to be the array with the sum of the squares of the wigner coefficients
            factor = (2 * ell1 + 1)*(2 * ell2 + 1)/(4*pi)
            A = amp_cls[ell1]
            B = beta_cls[ell2]
            X = w3j**2

            wignersum += factor * A * B * X

    return wignersum
#---------------------------------------------------------------------


#---------DEFINE THE 1X1 MOMENT FOR AUTO SPECTRA----------------------
def auto1x1(freqs, A=A_default, alpha=alpha_default, A_beta=A_beta_default, gamma=gamma_default, beta_0=beta_default, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default):
    sed_scaling = scale_synch(freqs, beta_0, nu0=nu0)
    ells = np.arange(0,ell_max)
    moment1x1 = np.zeros((len(freqs),len(ells)))
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_power_beta(ell_max=ell_max, A_beta=A_beta, gamma=gamma, beta_0=beta_0, nside=nside)
    wignersum = get_wigner_sum(ell_max, pcls, bcls)
    for i in range(len(moment1x1[:])):
        moment1x1[i] =  np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * wignersum

    return moment1x1


#---------DEFINE THE 0X2 MOMENT FOR AUTO SPECTRA------------------------
#this assumes a power law for beta
def auto0x2(freqs, A=A_default, alpha=alpha_default, ell_max=ell_max_default, nu0=nu0_default, beta_0=beta_default, A_beta=A_beta_default, gamma=gamma_default, nside=nside_default):
    sed_scaling = scale_synch(freqs, beta_0, nu0=nu0)
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    ells = np.arange(0,ell_max)
    moment0x2 = np.zeros((len(freqs),len(ells)))
    #the sum part becomes
    sum = 2 * sp.zeta(-gamma-1) + sp.zeta(-gamma) - 3
    #multiply by the prefactors of the sum
    sum = A_beta / (4 * pi * 80**gamma) * sum
    for i in range(len(moment0x2[:])):
        moment0x2[i] = np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * pcls * sum
    return moment0x2




#---------GET THE POWER SPECTRUM PLOTS----------------------------------
def get_plots(freqs, beta_0=beta_default, ell_max=ell_max_default, A=A_default, alpha=alpha_default, nu0=nu0_default, A_beta=A_beta_default, gamma=gamma_default, nside=nside_default, realisation=False, N=10):
    ells = np.arange(0,ell_max)
    moment0x0 = auto0x0(freqs, beta_0=beta_0, ell_max=ell_max, A=A, alpha=alpha, nu0=nu0)
    moment1x1 = auto1x1(freqs, A=A, alpha=alpha, A_beta=A_beta, gamma=gamma, beta_0=beta_0, ell_max=ell_max, nu0=nu0, nside=nside)
    moment0x2 = auto0x2(freqs, A=A, alpha=alpha, ell_max=ell_max, nu0=nu0, beta_0=beta_0, A_beta=A_beta, gamma=gamma, nside=nside)

    newmaps = map_full_power(freqs, ell_max=ell_max, A=A, alpha=alpha, A_beta=A_beta, gamma=gamma, beta_0=beta_0, nu0=nu0, nside=nside)


    if realisation==True:
        matrix = realisation_power(N, freqs)
        mean_ps = np.mean(matrix, 0)

        fig = plt.figure(figsize=(11,28))
        st = fig.suptitle(r'N=' + str(N) + r' realisations, $\alpha$=' + str(np.round(alpha,1))  + r', $\beta_0$=' + str(np.round(beta_0,1)) + r', $\gamma$=' + str(np.round(gamma,2)) + r', $\nu_0$=' + str(np.round(nu0*1e-9,1)) + ' GHz', fontsize=14)

        for i in range(len(freqs)):
            plt.subplot(len(freqs),1,i+1)
            for j in range(N):
                plt.semilogy(matrix[j,:,i], 'k', alpha = 0.2*10/N+0.1, lw=.5)
            plt.semilogy(mean_ps[:,i], 'r', label='mean PS')
            plt.semilogy(ells, moment0x0[i], label='0x0')
            # plt.semilogy(ells, moment1x1[i], label='1x1')
            # plt.semilogy(ells, moment0x2[i], label='0x2')
            # plt.semilogy(ells, moment0x0[i]+moment1x1[i], label='0x0 + 1x1')
            # plt.semilogy(ells, moment0x0[i]+moment0x2[i], label='0x0 + 0x2')
            plt.semilogy(ells, moment0x0[i]+moment1x1[i]+moment0x2[i], 'g',label='0x0 + 1x1 + 0x2')
            # plt.semilogy(ells, hp.anafast(newmaps[i]), 'r', label='anafast')

            plt.title(r'$\nu=$' + str(np.round(freqs[i]*1e-9)) + ' GHz.')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$C_\ell$')
            plt.legend()
            #space subplots out better
            fig.tight_layout()
            # st.set_y(0.95)
            fig.subplots_adjust(top=0.95)
        plt.show()

    else:
        fig = plt.figure(figsize=(11,7))
        st = fig.suptitle(r'$\alpha$=' + str(np.round(alpha,1))  + r', $\beta_0$=' + str(np.round(beta_0,1)) + r', $\gamma$=' + str(np.round(gamma,2)) + r', $\nu_0$=' + str(np.round(nu0*1e-9,1)) + ' GHz', fontsize=14)

        for i in range(len(freqs)):
            plt.subplot(len(freqs)/2,2,i+1)
            plt.semilogy(ells, moment0x0[i], label='0x0')
            plt.semilogy(ells, moment1x1[i], label='1x1')
            plt.semilogy(ells, moment0x2[i], label='0x2')
            # plt.semilogy(ells, moment0x0[i]+moment1x1[i], label='0x0 + 1x1')
            # plt.semilogy(ells, moment0x0[i]+moment0x2[i], label='0x0 + 0x2')
            plt.semilogy(ells, moment0x0[i]+moment1x1[i]+moment0x2[i], 'k', label='0x0 + 1x1 + 0x2')
            plt.semilogy(ells, hp.anafast(newmaps[i]), 'r', label='anafast')

            plt.title(r'$\nu=$' + str(np.round(freqs[i]*1e-9)) + ' GHz.')
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$C_\ell$')
            plt.legend()
            #space subplots out better
            fig.tight_layout()
            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)
        plt.show()
    return None
