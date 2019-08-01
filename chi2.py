#want a script that takes in a set of parameters (alphas, betas, gammas) and returns
#a matrix of chi2 values for each fit with those parameters.
import numpy as np
import healpy as hp
import foreground_functions as ff
import matplotlib.pyplot as plt
import time
import datetime

alpha = eval(input("alpha = "))
print(type(alpha))
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
#                     data_maps = map_full_power(freqs, ell_max=ell_max_default, A=A_default, alpha=alpha, sigma=sigma, gamma=gamma, beta_0=beta, nu0=nu0_default, nside=nside_default)
#                     mom0x0 = auto0x0(freqs, beta_0=beta, ell_max=ell_max_default, A=A_default, alpha=alpha, nu0=nu0_default)
#                     mom1x1 = auto1x1(freqs, A=A_default, alpha=alpha, sigma=sigma, gamma=gamma, beta_0=beta, ell_max=ell_max_default, nu0=nu0_default, nside=nside_default)
#                     mom0x2 = auto0x2(freqs, A=A_default, alpha=alpha_default, ell_max=ell_max_default, nu0=nu0_default, beta_0=beta_default, sigma=sigma_default, gamma=gamma_default, nside=nside_default)
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
