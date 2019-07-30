#========================================================
#Write script to calculate multiple realisations for different parameters and
#store the mean power spectrum. This way we can try to avoid cosmic variance in the
#residuals at low ell.

import numpy as np
import healpy as hp
import foreground_functions as ff
import matplotlib.pyplot as plt

freqs = np.linspace(30,300,4)*1.e9

#-------calculate multiple realisations-------------------
N = 10

x = ff.realisation_power(N, freqs)

mean_ps = np.mean(x, 0)
np.savetxt('test.txt', mean_ps)
np.save('output', mean_ps)
