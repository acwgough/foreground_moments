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
#add a second nside for generating data at higher nside
nside_1_default = 2*nside_default
ell_max_default = 3*nside_default
#make higher nside for generated data at higher nside
ell_max_1_default = 3*nside_1_default
A_default = 1.7e3
alpha_default = -3.0
beta_default = -3.2
beta_sigma_default = 1.5e-6
# nu0_default = 95e9 #to be able to converge between 30 and 300 GHz we should choose 95 GHz for best convergences

# #for a set of standard beta_cls to test consistency of w3j function
# beta_cls = np.load('beta_cls.npy')

#for power law beta
crit = 2/np.log(10)
sigma_default = crit/3
gamma_default = -2.5 #must be less than -2 for convergence in 0x2 term


#load in the w3j matrix
w3j = np.load('/Users/alex/Documents/foreground_w3j/w3j.npy')

#=====================================================================
#---------FUNCTIONS FOR AMPLITUDE MAP---------------------------------
#=====================================================================
#--------powerlaw function--------------------------------------------
def powerlaw(ells, amp, alpha):
    model = np.zeros(len(ells))
    model[2:] = (ells[2:] / 80.)**alpha
    return amp * model


#defines a normal planck distribution for unit conversion to kelvin
TCMB = 2.7255  # Kelvin
hplanck = 6.626070150e-34  # MKS
kboltz = 1.380649e-23  # MKS
def normed_cmb_thermo_units(nu):
    X = hplanck * nu / (kboltz * TCMB)
    eX = np.exp(X)
    return eX * X**4 / (eX - 1.)**2

def normed_synch(nu, beta):
    nu0_default = 95e9
    if beta is not np.array:
        beta = np.array(beta)
    return (nu/nu0_default)**(2.+beta[..., np.newaxis])

#-------synch SED given a (set of) freqency(ies) and a power-------------------------
def scale_synch(nu, beta):
    nu0 = 95e9
    unit = normed_synch(nu, beta) * normed_cmb_thermo_units(nu0) / normed_cmb_thermo_units(nu)
    return unit



#======================================================================
#---------MAP RELATED FUNCTIONS----------------------------------------
#======================================================================

#------generate an amplitude map---------------------------------------
def map_amp(ells, A=A_default, alpha=alpha_default):
    #returns input and output powerspectra, and the map
    nside = int(len(ells)/3)
    pcls = powerlaw(ells, A, alpha)
    amp_map = hp.synfast(pcls, nside, new=True, verbose=False)
    #check_pcls = hp.anafast(amp_map)
    return amp_map



#--------generate power law beta----------------------------------------
def map_power_beta(ells, beta=beta_default, gamma=gamma_default):
    nside=int(len(ells)/3)
    bcls = powerlaw(ells, 1, gamma)
    beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
    #model the standard deviation as a function of gamma
    #model is std = a * (-gamma)^b * exp(c * gamma)
    #best fit parameters 2019-08-08 are stored
    a = 4.16190627
    b = -3.28619789
    c = -2.56282892
    std = a * (-gamma)**b * np.exp(c*gamma)
    # std = np.std(beta_map)
    #update beta map to have the correct std dev
    beta_map = beta_map * sigma_default / std
    #update the map so that the mean is correct
    beta_map -= (np.mean(beta_map) - beta)
    return beta_map

def bcls(ells,  beta=beta_default, gamma=gamma_default):
    bcls = powerlaw(ells, 1, gamma)
    #model the standard deviation as a function of gamma
    #model is std = a * (-gamma)^b * exp(c * gamma)
    #best fit parameters 2019-08-08 are stored
    a = 4.16190627
    b = -3.28619789
    c = -2.56282892
    std = a * (-gamma)**b * np.exp(c*gamma)
    bcls *= (sigma_default/std)**2 #scaling the map scales the C_ell by the square factor
    return bcls


# #------generate maps with constant default beta---------------
# def map_full_const(ells, freqs, params):
#     A, alpha = params
#     amp_map = map_amp(ells, A=A, alpha=alpha)
#     SED = scale_synch(freqs, beta_default).T
#     newmaps = amp_map * SED[..., np.newaxis]
#     return newmaps

# #----generate maps with constant given beta-----
def map_full_const_beta(ells, freqs, params):
    A, alpha, beta = params
    amp_map = map_amp(ells, A=A, alpha=alpha)
    SED = scale_synch(freqs, beta).T
    newmaps = amp_map * SED[..., np.newaxis]
    return newmaps


#-------function to generate series of frequency maps with power spectrum for beta_map-------
def map_full_power(ells, freqs, params):
    A, alpha, beta, gamma = params
    amp_map = map_amp(ells, A=A, alpha=alpha)
    beta_map = map_power_beta(ells, beta=beta, gamma=gamma)

    sed_scaling = scale_synch(freqs, beta_map).T
    #make realistic maps
    newmaps = amp_map * sed_scaling
    #if only one frequency entered, cut out one dimension of the array so it is just (npix,) not (1,npix,)
    if len(freqs)==1:
        newmaps = newmaps[0]
    return newmaps

def ps_data(ells, freqs, params):
    A, alpha, beta, gamma = params
    long_ells = np.arange(2 * len(ells)) #make ells that are 2 times longer (corresponding to double the nside)
    data_maps = map_full_power(long_ells, freqs, params)

    #make the data at higher nside
    # data_maps = map_full_power(ells, freqs, params)
    if type(freqs)==np.ndarray:
        power_spectrum = np.zeros((len(freqs),len(long_ells)))
        for i in range(len(freqs)):
            power_spectrum[i] = hp.anafast(data_maps[i])
    else:
        power_spectrum = hp.anafast(data_maps)

    #cut the powerspectrum to the smaller ell value
    return power_spectrum[:,:len(ells)]



#=====================================================================
#---------MOMENT RELATED FUNCTIONS------------------------------------
#=====================================================================

#---------DEFINE FUNCTION FOR 0X0 AUTO--------------------------------
#from paper we know that 0x0 is SED^2 C_amp^2
#this gives a set of 0x0 moments at different frequencies
def auto0x0(ells, freqs, params):
    A, alpha, beta = params
    sed_scaling = scale_synch(freqs, beta)
    pcls = powerlaw(ells, A, alpha)

    #allows for single frequencies to be entered
    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment0x0 = np.zeros((len(freqs),len(ells)))
    for i in range(len(moment0x0[:])):
        moment0x0[i] = pcls * sed_scaling[i]**2

    if len(freqs)==1:
        moment0x0 = moment0x0[0]
    return moment0x0


def get_wigner_sum(ells, params):
    A, alpha, beta, gamma = params
    #over calcualte the wigner sum. Will need to recalculate the w3j matrix
    long_ells = np.arange(2*len(ells))
    amp_cls = powerlaw(long_ells, A, alpha)
    beta_cls = bcls(long_ells, beta=beta, gamma=gamma)
    f = 2*long_ells+1
    # w3j1 = w3j[:len(ells), :len(ells), :len(ells)]
    wignersum = np.einsum("i,i,j,j,kij", f, amp_cls, f, beta_cls, w3j, optimize=True)[:len(ells)]
    return 1/(4*pi)*wignersum


#---------DEFINE THE 1X1 MOMENT FOR AUTO SPECTRA----------------------
def auto1x1(ells, freqs, params):
    A, alpha, beta, gamma = params
    nu0 = 95e9
    sed_scaling = scale_synch(freqs, beta)

    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment1x1 = np.zeros((len(freqs),len(ells)))
    wignersum = get_wigner_sum(ells, params)
    for i in range(len(moment1x1[:])):
        moment1x1[i] =  np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * wignersum

    if len(freqs)==1:
        moment1x1 = moment1x1[0]
    return moment1x1


#---------DEFINE THE 0X2 MOMENT FOR AUTO SPECTRA------------------------
#this assumes a power law for beta
def auto0x2(ells, freqs, params):
    A, alpha, beta, gamma = params
    nu0 = 95e9
    sed_scaling = scale_synch(freqs, beta)
    pcls = powerlaw(ells, A, alpha)
    beta_cls = bcls(ells, beta=beta, gamma=gamma)

    if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
        freqs = np.array(freqs)[np.newaxis]

    moment0x2 = np.zeros((len(freqs),len(ells)))
    #the sum part becomes
    sum = 2 * sp.zeta(-gamma-1) + sp.zeta(-gamma) - 3
    #multiply by the prefactors of the sum
    #have to add due to rescaling the beta map to have same std.
    A_beta = beta_cls[80]

    sum = A_beta / (4 * pi * 80**gamma) * sum
    for i in range(len(moment0x2[:])):
        moment0x2[i] = np.log(freqs[i]/nu0)**2 * sed_scaling[i]**2 * pcls * sum

    if len(freqs)==1:
        moment0x2 = moment0x2[0]
    return moment0x2

#define the full model
def model(ells, freqs, params):
    #these return the maps, shape (number of freqs, number of pix)
    A, alpha, beta, gamma = params
    ell_max = len(ells)
    mom0x0 = auto0x0(ells, freqs, params[:-1]) #as 0x0 doesn't take gamma
    mom1x1 = auto1x1(ells, freqs, params)
    mom0x2 = auto0x2(ells, freqs, params)
    model  = mom0x0 + mom1x1 + mom0x2
    return model



def chi2(params, ells, freqs, data):
    chi2=0
    A, alpha, beta, gamma = params
    model = model(ells, freqs, params)
    
    var = np.zeros((len(freqs),len(ells)))
    for ell in range(len(ells)):
        var[:,ell] = 2/(2*ell+1)
    cosmic_var = var * model**2

    #don't count the first 30 ell in the objective function.
    chi2 = (data[:,30:] - model[:,30:])**2 / cosmic_var[:,30:]
    return np.sum(chi2)


#
# #--------get chi_square value for a fit from set of data and the model------
# def get_chi_square(data, model):
#     resid = data-model
#     chi_square = 0
#     for ell in range(2, len(resid)): #ignore monopole and dipole contributions as cosmic variance should be 0
#         cosmic_var = 2/(2*ell+1) * model[ell]**2
#         chi_square += resid[ell]**2 / cosmic_var
#     return chi_square
#


#this version takes out the realisations, as we'll calculate those averages separately and then pull them in.
#this version also avoids the quadruple subplot so we can put the residuals on the bottom of each plot.

# #TODO: work out why the ticks on the labels are overlapping, clean up plots etc.
# def get_plots(freqs, beta=beta_default, ell_max=ell_max_default, A=A_default, alpha=alpha_default, nu0=nu0_default, sigma=sigma_default, gamma=gamma_default, nside=nside_default):
#     ells = np.arange(0,ell_max)
#     moment0x0 = auto0x0(freqs, beta=beta, ell_max=ell_max, A=A, alpha=alpha, nu0=nu0)
#     moment1x1 = auto1x1(freqs, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, ell_max=ell_max, nu0=nu0, nside=nside)
#     moment0x2 = auto0x2(freqs, A=A, alpha=alpha, ell_max=ell_max, nu0=nu0, beta=beta, sigma=sigma, gamma=gamma, nside=nside)
#     model = moment0x0+moment1x1+moment0x2
#
#     newmaps = map_full_power(ells, freqs, params)
#
#     for i in range(len(freqs)):
#         anafast=hp.anafast(newmaps[i])
#         chi_square = get_chi_square(anafast,model[i])
#         fig = plt.figure(i, figsize=(11,7))
#         ax = plt.subplot(111)
#         frame1=fig.add_axes((.1,.5,.8,.6))
#         #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
#         plt.semilogy(ells[2:], moment0x0[i][2:], label='0x0')
#         plt.semilogy(ells[2:], moment1x1[i][2:], label='1x1')
#         plt.semilogy(ells[2:], moment0x2[i][2:], label='0x2')
#         # plt.semilogy(ells, moment0x0[i]+moment1x1[i], label='0x0 + 1x1')
#         # plt.semilogy(ells, moment0x0[i]+moment0x2[i], label='0x0 + 0x2')
#         plt.semilogy(ells[2:], model[i][2:], 'k', label='0x0 + 1x1 + 0x2')
#         # plt.errorbar(ells[2:], model[i][2:], yerr=cosmic_var[i][2:], fmt='.')
#         plt.semilogy(ells[2:], anafast[2:], 'r', label='anafast')
#         frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
#         # plt.text(0.5,0.95,r'0.5, 0.95',fontsize=14, transform=ax.transAxes)
#         # plt.text(0.5,1.0,r'0.5, 1.0',fontsize=14, transform=ax.transAxes)
#         # plt.text(0.5,1.2,r'0.5, 1.2',fontsize=14, transform=ax.transAxes)
#         plt.text(0.65,1.23,r'$\chi^2 = $' + str(format(chi_square, '.4g')), transform=ax.transAxes, bbox=dict(facecolor='white',edgecolor='0.8',alpha=0.8))
#         plt.title(r'$\nu=$' + str(np.round(freqs[i]*1e-9)) + ' GHz.' + '\n' + r'$\alpha$=' + str(np.round(alpha,1))  + r', $\beta$=' + str(np.round(beta,1)) + r', $\gamma$=' + str(np.round(gamma,2)) + r'$, \sigma$=' + str(np.round(sigma,2)) +r', $\nu_0$=' + str(np.round(nu0*1e-9,1)) + ' GHz')
#         # plt.xlabel(r'$\ell$')
#         plt.ylabel(r'$C_\ell$')
#         plt.legend()
#         plt.grid()
#
#         #fractional error plot
#         frac_error = (anafast-model[i])/anafast
#         # ratio = anafast/model[i]
#         frame2=fig.add_axes((.1,.3,.8,.2))
#         plt.plot(ells[2:], frac_error[2:], '.')
#         # plt.plot(ells,ratio, '.')
#         plt.plot(ells[2:], np.zeros_like(ells[2:]))
#         frame2.set_xticklabels([]) #Remove x-tic labels for the 2nd frame
#         plt.ylabel('frac. err.')
#         plt.grid()
#
#         #residuals plot
#         resid = (anafast-model[i])**2
#         frame3=fig.add_axes((.1,.1,.8,.2))
#         plt.semilogy(ells[2:], resid[2:], '.')
#         plt.ylabel(r'$\mathrm{resid}^2$')
#         plt.xlabel(r'$\ell$')
#         plt.grid()
#
#     plt.show()
#     return None


#write objecctive function chi2 to be minimised by the optimizer
#
# def chi2(data, freq, param):
#     #given a map of c_ells (the data), compute the model with alpha, beta, gamma, at freqency
#     #freq, and calculates the chi2 value
#     alpha = param[0]
#     beta = param[1]
#     gamma = param[2]
#     # A = param[3]
#     # sigma = param[4]
#     mom0x0 = auto0x0(freq, beta=beta, ell_max=ell_max_default, A=A_default, alpha=alpha, nu0=nu0_default)
#     mom1x1 = auto1x1(freq, A=A_default, alpha=alpha, sigma=sigma_default, gamma=gamma, beta=beta, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default)
#     mom0x2 = auto0x2(freq, A=A_default, alpha=alpha, ell_max=ell_max_default, nu0=nu0_default, beta=beta, sigma=sigma_default, gamma=gamma, nside=nside_default)
#     model = mom0x0 + mom1x1 + mom0x2
#     residual = data - model
#     chi_square = 0
#     for ell in range(2,len(residual)):
#         cosmic_var = 2/(2*ell+1) * model[ell]**2
#         chi_square += residual[ell]**2 / cosmic_var
#     return chi_square










#old functions that are no longer in use

#-------generate beta map with uniform power spectrum------------------
# def map_beta(ell_max=ell_max_default, sigma=beta_sigma_default, beta=beta_default, nside=nside_default):
#     ells = np.arange(0,ell_max)
#     bcls = sigma * np.ones_like(ells)
#     bcls[0] = 0
#     bcls[1] = 0
#     beta_map = hp.synfast(bcls, nside, new=True, verbose=False)
#     #update the map so that the mean is correct
#     beta_map -= (np.mean(beta_map) - beta)
#     # check_bcls = hp.anafast(beta_map)
#     return bcls, beta_map


# #-------function to generate series of frequency maps with constant C_ell^beta/white noise----
# #-------does not return the c_ells, just the maps themselves
# def map_full_white(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, beta_sigma=beta_sigma_default, beta=beta_default, nu0=nu0_default, nside=nside_default):
#     pcls, amp_map = map_amp(ell_max=ell_max, A=A, alpha=alpha, nside=nside)
#     bcls, beta_map = map_beta(ell_max=ell_max, sigma=beta_sigma, beta=beta, nside=nside)
#     sed_scaling_beta = scale_synch(freqs, beta_map, nu0=nu0).T
#     #make realistic maps
#     newmaps_beta = amp_map * sed_scaling_beta
#     return newmaps_beta

#
# #-------function to make many realisations of the same white noise map------------------------
# def realisation(N, freqs, ell_max=ell_max_default):
#     ells = np.arange(0, ell_max)
#     instance = np.zeros((N,len(ells),len(freqs)))
#     #instance[0,:,i] picks out the 0th realisation of the ith frequency
#     for i in range(N):
#         maps = map_full_white(freqs)
#         for j in range(len(freqs)):
#             instance[i,:,j] = hp.anafast(maps[j])
#     return instance
#
#
# #--------function to make many realisations of the same power map----------------------
# def realisation_power(N, freqs, ell_max=ell_max_default, A=A_default, alpha=alpha_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, nu0=nu0_default, nside=nside_default):
#     ells = np.arange(0,ell_max)
#     realisation = np.zeros((N, len(ells), len(freqs)))
#     for i in range(N):
#         #printing progress for long runs.
#         if (i/N*100)%5==0:
#             print(str(np.round(i/N*100)) + '%')
#
#         maps = map_full_power(freqs, ell_max=ell_max, A=A, alpha=alpha, sigma=sigma, gamma=gamma, beta=beta, nu0=nu0, nside=nside)
#         for j in range(len(freqs)):
#             realisation[i,:,j] = hp.anafast(maps[j])
#     print('100%')
#     return realisation


#---------GET WIGNER SUM PART OF EQUATION 35 FOR 1x1moment-------------
# def get_wigner_sum(ell_max=ell_max_default, alpha=alpha_default, A=A_default, sigma=sigma_default, gamma=gamma_default, beta=beta_default, nside=nside_default):
#     ells = np.arange(0, ell_max)  #the ells to be summed over
#     #define an empty array to store the wigner sum in
#     wignersum = np.zeros_like(ells, dtype=float)
#     amp_cls = powerlaw(ells, A, alpha)
    # beta_cls = bcls(ell_max=ell_max, sigma=sigma, gamma=gamma, beta=beta, nside=nside)

    # beta_cls = map_power_beta(ell_max=ell_max, sigma=sigma, gamma=gamma, beta=beta, nside=nside)
#     #defines an array for the factor later, saves time in the loop
#     # factor = np.zeros((ell_max, ell_max))
#     # for i in range(ell_max):
#     #     for j in range(ell_max):
#     #         factor[i,j] = (2*i+1)*(2*j+1)
#     # factor = factor/(4*pi)
#
#     #can do the array factor with one loop that just stores (2ell+1)/4pi and call two different elements
#     # factor = np.zeros(ell_max)
#     # for i in range(ell_max):
#     #     factor[i] = (2*i+1)
#     # factor = factor/(np.sqrt(4*pi))
#     #sqrt 4 pi on bottom so that when two are multiplied together we have
#     #(2ell1 + 1)(2ell2 + 1)/4pi
#     factor = np.array([2*i+1 for i in range(384)])/(np.sqrt(4*pi))
#     #better to just use
      # factor = 2*ells+1
#
#     for ell1 in ells:
#         A1 = factor[ell1]
#         B = amp_cls[ell1]
#         for ell2 in ells:
#             #define wignersum to be the array with the sum of the squares of the wigner coefficients
#             A2 = factor[ell2]
#             C = beta_cls[ell2]
#             D = w3j[:,ell1,ell2]
#             E = A1 * A2 * B * C * D
#
#             wignersum += E
#
#     return wignersum


# #function for producing something we can hopefully do a 1d curve fit to...
# #
# def model_single(ells, A, alpha):
#     #start with this just calculating the 0x0 moment and trying to fit that to data.
#     #can fit for just
#
#     ell_max = len(ells)
#     pcls = powerlaw(ells, A, alpha)
#     sed_scaling = scale_synch(30e9, beta_default)
#     bcls = map_power_beta(ell_max=ell_max)
#
#
#     moment0x0 = pcls * sed_scaling**2
#
#     wignersum = get_wigner_sum(ell_max=ell_max, A=A, alpha=alpha)
#     moment1x1 =  np.log(30e9/nu0_default)**2 * sed_scaling**2 * wignersum
#
#     # sum = 2 * sp.zeta(-gamma_default-1) + sp.zeta(-gamma_default) - 3
#     # A_beta = bcls[80]
#     # sum *= A_beta / (4 * pi * 80**gamma_default)
#     # moment0x2 = np.log(30e9/nu0_default)**2 * sed_scaling**2 * pcls * sum
#
#     moment0x2 = auto0x2(30e9, A=A, alpha=alpha)
#
#     model = moment0x0 + moment1x1 + moment0x2
#     return model


#
# #wrtie function for full model that is independent of the auto functions as written
# #checked, this produces the same as the one defining each term indeividually 2019-08-09
# def full_model(ells, freqs, A, alpha, beta, gamma):
#     ell_max = len(ells)
#     pcls =powerlaw(ells, A, alpha)
#     sed_scaling = scale_synch(freqs, beta)
#
#     #better to make the bcls and not the beta map, faster
#     bcls = powerlaw(ells, 1, gamma)
#     a = 4.16190627
#     b = -3.28619789
#     c = -2.56282892
#     std = a * (-gamma)**b * np.exp(c*gamma)
#     bcls = bcls * (sigma_default/std)**2 #scaling the map scales the C_ell by the square factor
#
#     # bcls, beta_map = map_power_beta(ell_max=ell_max, beta=beta, gamma=gamma)
#     #allows for single frequencies to be entered
#     if type(freqs)==np.float64 or type(freqs)==int or type(freqs)==float:
#         freqs = np.array(freqs)[np.newaxis]
#
#     #get the auto0x0 term
#     moment0x0 = np.zeros((len(freqs),len(ells)))
#     for i in range(len(moment0x0[:])):
#         moment0x0[i] = pcls * sed_scaling[i]**2
#
#
#     if len(freqs)==1:
#         moment0x0 = moment0x0[0]
#
#     #get auto1x1 term
#     moment1x1 = np.zeros((len(freqs),len(ells)))
#     wignersum = get_wigner_sum(ell_max=ell_max, A=A, alpha=alpha, beta=beta, gamma=gamma)
#     # wignersum = get_wigner_sum(ell_max, pcls, bcls)
#     for i in range(len(moment1x1[:])):
#         moment1x1[i] =  np.log(freqs[i]/nu0_default)**2 * sed_scaling[i]**2 * wignersum
#
#
#     if len(freqs)==1:
#         moment1x1 = moment1x1[0]
#
#     #get the auto0x2 term
#     moment0x2 = np.zeros((len(freqs),len(ells)))
#     #the sum part becomes
#     sum = 2 * sp.zeta(-gamma-1) + sp.zeta(-gamma) - 3
#     #multiply by the prefactors of the sum
#     #have to add due to rescaling the beta map to have same std.
#     A_beta = bcls[80]
#
#     sum *= A_beta / (4 * pi * 80**gamma)
#     for i in range(len(moment0x2[:])):
#         moment0x2[i] = np.log(freqs[i]/nu0_default)**2 * sed_scaling[i]**2 * pcls * sum
#
#
#     if len(freqs)==1:
#         moment0x2 = moment0x2[0]
#     return moment0x0 + moment1x1 + moment0x2
