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
crit = 2/np.log(10)
sigma_default = crit/3
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
def map_beta(ell_max=ell_max_default, sigma=beta_sigma_default, beta=beta_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    bcls = sigma * np.ones_like(ells)
    bcls[0] = 0
    bcls[1] = 0
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta)
    # check_bcls = hp.anafast(beta_map)
    return bcls, beta_map


#--------generate power law beta----------------------------------------
def map_power_beta(ell_max=ell_max_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    bcls = powerlaw(ells, gamma)
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    std = np.std(beta_map)
    #update beta map to have the correct std dev
    beta_map = beta_map * sigma / std
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta)

    #update the beta_cls
    bcls = bcls * (sigma/std)**2 #scaling the map scales the C_ell by te square factor
    # check_bcls = hp.anafast(beta_map)
    return bcls, beta_map



#-------function to generate series of frequency maps with constant C_ell^beta/white noise----
#-------does not return the c_ells, just the maps themselves
def map_full_white(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, beta_sigma=beta_sigma_default, beta=beta_default, nu0=nu0_default, nside=nside_default):
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_beta(ell_max=ell_max, sigma=beta_sigma, beta=beta, nside=nside)
    sed_scaling_beta = scale_synch(freqs, beta_map, nu0=nu0).T
    #make realistic maps
    newmaps_beta = amp_map * sed_scaling_beta
    return newmaps_beta


#-------function to generate series of frequency maps with power spectrum for beta_map-------
def map_full_power(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, nu0=nu0_default, nside=nside_default):
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_power_beta(ell_max=ell_max, sigma=sigma, gamma=gamma, beta=beta, nside=nside)

    sed_scaling_beta = scale_synch(freqs, beta_map, nu0=nu0).T
    #make realistic maps
    newmaps_beta = amp_map * sed_scaling_beta
    #if only one frequency entered, cut out one dimension of the array so it is just (npix,) not (1,npix,)
    if len(newmaps_beta[:])==1:
        newmaps_beta = newmaps_beta[0]
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
def realisation_power(N, freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, nu0=nu0_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    realisation = np.zeros((N, len(ells), len(freqs)))
    for i in range(N):
        #printing progress for long runs.
        if (i/N*100)%5==0:
            print(str(np.round(i/N*100)) + '%')

        maps = map_full_power(freqs, ell_max=ell_max, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, nu0=nu0, nside=nside)
        for j in range(len(freqs)):
            realisation[i,:,j] = hp.anafast(maps[j])
    print('100%')
    return realisation


#=====================================================================
#---------MOMENT RELATED FUNCTIONS------------------------------------
#=====================================================================

#---------DEFINE FUNCTION FOR 0X0 AUTO--------------------------------
#from paper we know that 0x0 is SED^2 C_amp^2
#this gives a set of 0x0 moments at different frequencies
def auto0x0(freqs, beta=beta_default, ell_max=ell_max_default, A=A_default, alpha=alpha_default, nu0=nu0_default):
    sed_scaling = scale_synch(freqs, beta, nu0=nu0)
    ells = np.arange(0,ell_max)
    pcls = A * powerlaw(ells, alpha)
    #allows for single frequencies to be entered
    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment0x0 = np.zeros((len(freqs),len(ells)))
    for i in range(len(moment0x0[:])):
        moment0x0[i] = pcls * sed_scaling[i]**2

    if len(freqs)==1:
        moment0x0 = moment0x0[0]
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

    #defines an array for the factor later, saves time in the loop
    factor = np.zeros((ell_sum, ell_sum))
    for i in range(ell_sum):
        for j in range(ell_sum):
            factor[i,j] = (2*i+1)*(2*j+1)
    factor = factor/(4*pi)


    for ell1 in ells_ext[:ell_sum]:
        for ell2 in ells_ext[:ell_sum]:
            w3j, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
            avaliable_ells = np.arange(ellmin, ellmax+1)
            #this block forces all the w3j arrays to have the same size as the wignersum array
            #cut off trailing zeros at the end of w3j
            # max_nonzero_index = ellmax - ellmin
            w3j = w3j[:ellmax - ellmin + 1]
            #make the w3j array the same shape as the wignersum array
            if len(w3j) < len(ells_ext):
                #a set of zeros of the same shape which we'll use for padding
                reference = np.zeros(len(wignersum))
                reference[:w3j.shape[0]] = w3j
                w3j = reference

            #roll stuff into position and relabel those that roll ''around'' to 0. Using concatenate here
            #as it is faster than np.roll.
            w3j = np.concatenate([w3j[-ellmin:],w3j[:-ellmin]])
            #cut to size of the moment that we're adding (the size of the ells matrix).
            w3j = w3j[:len(ells_ext)]
            #set those that rolled around to 0
            w3j[:ellmin] = 0

            #define wignersum to be the array with the sum of the squares of the wigner coefficients
            wignersum += factor[ell1, ell2] * amp_cls[ell1] * beta_cls[ell2] * (w3j**2)

    return wignersum
#---------------------------------------------------------------------


#---------DEFINE THE 1X1 MOMENT FOR AUTO SPECTRA----------------------
def auto1x1(freqs, A=A_default, alpha=alpha_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default):
    sed_scaling = scale_synch(freqs, beta, nu0=nu0)
    ells = np.arange(0,ell_max)

    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment1x1 = np.zeros((len(freqs),len(ells)))
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_power_beta(ell_max=ell_max, sigma=sigma, gamma=gamma, beta=beta, nside=nside)
    wignersum = get_wigner_sum(ell_max, pcls, bcls)
    for i in range(len(moment1x1[:])):
        moment1x1[i] =  np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * wignersum

    if len(freqs)==1:
        moment1x1 = moment1x1[0]
    return moment1x1


#---------DEFINE THE 0X2 MOMENT FOR AUTO SPECTRA------------------------
#this assumes a power law for beta
def auto0x2(freqs, A=A_default, alpha=alpha_default, ell_max=ell_max_default, nu0=nu0_default, beta=beta_default, sigma=sigma_default, gamma=gamma_default, nside=nside_default):
    sed_scaling = scale_synch(freqs, beta, nu0=nu0)
    pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
    bcls, beta_map = map_power_beta(ell_max=ell_max, sigma=sigma, gamma=gamma, beta=beta, nside=nside)
    ells = np.arange(0,ell_max)

    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment0x2 = np.zeros((len(freqs),len(ells)))
    #the sum part becomes
    sum = 2 * sp.zeta(-gamma-1) + sp.zeta(-gamma) - 3
    #multiply by the prefactors of the sum
    #have to add due to rescaling the beta map to have same std.
    A_beta = bcls[80]

    sum = A_beta / (4 * pi * 80**gamma) * sum
    for i in range(len(moment0x2[:])):
        moment0x2[i] = np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * pcls * sum

    if len(freqs)==1:
        moment0x2 = moment0x2[0]
    return moment0x2





#--------get chi_square value for a fit from set of data and the model------
def get_chi_square(data, model):
    resid = data-model
    chi_square = 0
    for ell in range(len(resid)):
        cosmic_var = 2/(2*ell+1) * data[ell]**2
        chi_square += resid[ell]**2 / cosmic_var
    return chi_square



#this version takes out the realisations, as we'll calculate those averages separately and then pull them in.
#this version also avoids the quadruple subplot so we can put the residuals on the bottom of each plot.

#TODO: work out why the ticks on the labels are overlapping, clean up plots etc.
def get_plots(freqs, beta=beta_default, ell_max=ell_max_default, A=A_default, alpha=alpha_default, nu0=nu0_default, sigma=sigma_default, gamma=gamma_default, nside=nside_default):
    ells = np.arange(0,ell_max)
    moment0x0 = auto0x0(freqs, beta=beta, ell_max=ell_max, A=A, alpha=alpha, nu0=nu0)
    moment1x1 = auto1x1(freqs, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, ell_max=ell_max, nu0=nu0, nside=nside)
    moment0x2 = auto0x2(freqs, A=A, alpha=alpha, ell_max=ell_max, nu0=nu0, beta=beta, sigma=sigma, gamma=gamma, nside=nside)
    model = moment0x0+moment1x1+moment0x2
    newmaps = map_full_power(freqs, ell_max=ell_max, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, nu0=nu0, nside=nside)

    for i in range(len(freqs)):
        anafast=hp.anafast(newmaps[i])
        chi_square = get_chi_square(anafast,model[i])
        fig = plt.figure(i, figsize=(11,7))
        ax = plt.subplot(111)
        frame1=fig.add_axes((.1,.5,.8,.6))
        #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.semilogy(ells, moment0x0[i], label='0x0')
        plt.semilogy(ells, moment1x1[i], label='1x1')
        plt.semilogy(ells, moment0x2[i], label='0x2')
        # plt.semilogy(ells, moment0x0[i]+moment1x1[i], label='0x0 + 1x1')
        # plt.semilogy(ells, moment0x0[i]+moment0x2[i], label='0x0 + 0x2')
        plt.semilogy(ells, moment0x0[i]+moment1x1[i]+moment0x2[i], 'k', label='0x0 + 1x1 + 0x2')
        plt.semilogy(ells, anafast, 'r', label='anafast')
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        # plt.text(0.5,0.95,r'0.5, 0.95',fontsize=14, transform=ax.transAxes)
        # plt.text(0.5,1.0,r'0.5, 1.0',fontsize=14, transform=ax.transAxes)
        # plt.text(0.5,1.2,r'0.5, 1.2',fontsize=14, transform=ax.transAxes)
        plt.text(0.65,1.23,r'$\chi^2 = $' + str(format(chi_square, '.4g')), transform=ax.transAxes, bbox=dict(facecolor='white',edgecolor='0.8',alpha=0.8))
        plt.title(r'$\nu=$' + str(np.round(freqs[i]*1e-9)) + ' GHz.' + '\n' + r'$\alpha$=' + str(np.round(alpha,1))  + r', $\beta$=' + str(np.round(beta,1)) + r', $\gamma$=' + str(np.round(gamma,2)) + r'$, \sigma$=' + str(np.round(sigma,2)) +r', $\nu_0$=' + str(np.round(nu0*1e-9,1)) + ' GHz')
        # plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell$')
        plt.legend()
        plt.grid()

        #fractional error plot
        frac_error = (anafast-model[i])/anafast
        # ratio = anafast/model[i]
        frame2=fig.add_axes((.1,.3,.8,.2))
        plt.plot(ells, frac_error, '.')
        # plt.plot(ells,ratio, '.')
        plt.plot(ells, np.zeros_like(ells))
        frame2.set_xticklabels([]) #Remove x-tic labels for the 2nd frame
        plt.ylabel('frac. err.')
        plt.grid()

        #residuals plot
        resid = (anafast-model[i])**2
        frame3=fig.add_axes((.1,.1,.8,.2))
        plt.semilogy(ells, resid, '.')
        plt.ylabel(r'$\mathrm{resid}^2$')
        plt.xlabel(r'$\ell$')
        plt.grid()
    plt.show()
    return None


#write objecctive function chi2 to be minimised by the optimizer

def chi2(data, freq, param):
    #given a map of c_ells (the data), compute the model with alpha, beta, gamma, at freqency
    #freq, and calculates the chi2 value

    alpha = param[0]
    beta = param[1]
    gamma = param[2]
    # A = param[3]
    # sigma = param[4]

    mom0x0 = auto0x0(freq, beta=beta, ell_max=ell_max_default, A=A_default, alpha=alpha, nu0=nu0_default)
    mom1x1 = auto1x1(freq, A=A_default, alpha=alpha, sigma=sigma_default, gamma=gamma, beta=beta, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default)
    mom0x2 = auto0x2(freq, A=A_default, alpha=alpha, ell_max=ell_max_default, nu0=nu0_default, beta=beta, sigma=sigma_default, gamma=gamma, nside=nside_default)
    residual = data - (mom0x0 + mom1x1 + mom0x2)
    chi_square = 0
    for ell in range(len(residual)):
        cosmic_var = 2/(2*ell+1) * data[ell]**2
        chi_square += residual[ell]**2 / cosmic_var
    return chi_square


#want to make a matrix that collects chi^2 values for different sets of parameters (input parameter arrays)
#we can then slice and plot this to see how chi^2 varies with the parameters
# def chi2(freqs, alphas, betas, gammas, sigmas=sigma_default):
#     #make sure everything is an array
#     #the new axis means that these are now shape (1,) arrays
#     if type(alphas) == np.float64 or type(alphas) == int:
#         alphas = np.array(alphas)[np.newaxis]
#     if type(betas) == np.float64 or type(betas) == int:
#         betas = np.array(betas)[np.newaxis]
#     if type(gammas) == np.float64 or type(gammas) == int:
#         gammas = np.array(gammas)[np.newaxis]
#     if type(sigmas) == np.float64 or type(sigmas) == int:
#         sigmas = np.array(sigmas)[np.newaxis]
#     if type(freqs) == np.float64 or type(freqs) == int:
#         freqs = np.array(freqs)[np.newaxis]
#     # generate some data. For each element of this chi^2 matrix we need to make its own map
#
#     #set up the empty chi2 matrix
#     chi2 = np.zeros((len(freqs),len(alphas),len(betas),len(gammas),len(sigmas)))
#     #grab the individual elements and update the chi2 matrix
#     for alpha in alphas:
#         for beta in betas:
#             for gamma in gammas:
#                 for sigma in sigmas:
#                     data_maps = map_full_power(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, nu0=nu0_default, nside=nside_default)
#                     mom0x0 = auto0x0(freqs, beta=beta, ell_max=ell_max_default, A=A_default, alpha=alpha, nu0=nu0_default)
#                     mom1x1 = auto1x1(freqs, A=A_default, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default)
#                     mom0x2 = auto0x2(freqs, A=A_default, alpha=alpha_default, ell_max=ell_max_default, nu0=nu0_default, beta=beta_default, sigma=sigma_default, gamma=gamma_default, nside=nside_default)
#                     model = mom0x0 + mom1x1 + mom0x2
#                     for i in range(len(freqs)):
#                         chi_square=0
#                         # print(data_maps[i].shape)
#                         cls = hp.anafast(data_maps[i])
#                         residual = cls[i]-model[i]
#
#                         for ell in range(len(residual)):
#                             cosmic_var = 2/(2*ell + 1) * cls[ell]
#                             chi_square += residual[ell]**2 / cosmic_var
#     return chi_square, chi2
