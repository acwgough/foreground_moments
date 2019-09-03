# Foreground moments

This is a set of files used to calculate moments in a parametric expansion of the galactic polarisation foreground for use in CMB experiments. Currently it calculates the 0th and 1st order moments in synchrotron and dust emission.

# File structure
`foreground_functions.py`

`w3j.py`

(`w3j.npy`)

----2019-notebooks

    |----2019-07

    |----2019-08

    |----2019-09

----power_spectra

----mcmc

    |----mcmc_scripts

    |----mcmc_chains

## 2019-notebooks
These store the jupyter notebooks used to experiment with different aspects of writing this code for this project. The are further subdivided by month for organisation purposes. The naming convention for the files is `yyyy-mm-dd <description>.ipynb`. Many of these notebooks with the same description are copies of earlier notebooks.

## power_spectra
This directory stores various different power spectra generated by the notebooks to be loaded in and fit for reference. This is done to avoid differences in realisations changing fits so we can test for consistency when fits are being done. The most important files in this directory are the CAMB.dat files, which contain the data used for making faux CMB to try to extract. There is both a primordial CMB signal and a lensed signal.

## mcmc
This directory houses two subdirectories: mcmc_chains and mcmc_scripts. The mcmc_scripts directory should be used for what it says on the tin, writing different mcmc scripts to run. Currently there is one test mcmc script there called `MCMC.py`. The mcmc_chains directory should be where chains from the scripts should be saved to be loaded into notebooks and things for plotting.

# Function files
## `w3j.py`
This python3 script calculates the Wigner 3j coefficients needed for the 1x1 moment of the model. As these are computationally intensive, the w3j.py script writes them to a numpy file in the same directory called `w3j.npy`. This file is too large to push to GitHub, so `w3j.py` should be run to produce this file before anything else is done with this repo. `w3j.py` takes one input which is the maximum ell you'd like to consider (e.g. 384 for a map with nside=128). It then calculates an array of shape (ell_max, ell_max, ell_max) with elements [i,j,k] corresponding to the value

`/ i  j  k \
\ 0  0  0 /`

and then squares this array elementwise (as the 1x1 moment depends only on the square of these elements).

## `foreground_functions.py`
This is the most important file in the repo. Here are the functions used to generate data from a set of parameters, and also generate the model from a set of parameters.

### Dependencies
This file requires `numpy`, `matplotlib.pyplot`, `healpy`, `scipy.special.zeta` and that you've generated the appropriately sized `w3j.npy` file.

### Functions
`foreground_functions.py` contains a few different kinds of files. The code is fairly commented, but I'll outline here the main sorts of functions contained.
1. ***SED functions.*** There are functions called `scale_dust` and `scale_synch` that are scaled SEDs for dust and synchrotron radiation into CMB units.
1. ***Map functions.*** These include amplitude/template maps, beta maps, and full foreground maps, both in the case of constant beta and spatially varying beta.
1. ***Power spectrum functions.*** This includes a function to calculate the C_ells from a beta map without actually constructing the full map (faster), as well as functions that construct full foreground maps to then `hp.anafast` and returns the power spectrum. _Note that different functions will return either cls or dls. In general, the only things that will produce cls are functions called bcls (for beta cls). Everything else is converted to dls for consistency with other papers and the CMB files._
