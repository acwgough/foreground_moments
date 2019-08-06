#write a script to calculate the w3j matrix sum up to a maximum index, and save this to file so that
#it can be loaded into the auto1x1 to save time.
ell_sum = eval(input('Maximum ell = '))

import numpy as np
import healpy as hp
from pyshtools.utils import Wigner3j
from math import pi

ells_ext = np.arange(0, ell_sum)  #the ells to be summed over
#define an empty array to store the wigner sum in
wignersum = np.zeros_like(ells_ext, dtype=float)

#an array where we'll store the
#/ ell  ell_1  ell_2\
#\0      0       0  /
#as the [ell, ell_1, ell_2] elements
big_w3j = np.zeros((ell_sum, ell_sum, ell_sum))

for ell1 in ells_ext[:ell_sum]:
    for ell2 in ells_ext[:ell_sum]:
        w3j, ellmin, ellmax = Wigner3j(ell1, ell2, 0, 0, 0)
        # this block forces all the w3j arrays to have the same size as the wignersum array
        # cut off trailing zeros at the end of w3j
        #max_nonzero_index = ellmax - ellmin
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

        big_w3j[:,ell1,ell2] = w3j

big_w3j = big_w3j**2
#the array w3j now holds all the (ell ell1 ell2) elements in an array. Grab with [: ell1 ell2] element.
#write this to file
filename = 'w3j'
np.save(filename, big_w3j)

        #define wignersum to be the array with the sum of the squares of the wigner coefficients
        # wignersum += factor[ell1, ell2] * amp_cls[ell1] * beta_cls[ell2] * (w3j**2)
