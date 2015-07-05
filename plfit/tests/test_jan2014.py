from __future__ import print_function
import scipy.io
import plfit
import time
import os
import pdb
import numpy as np
from matplotlib import pyplot as plt
#pip install https://pypi.python.org/pypi/powerlaw 
import powerlaw

datadir=os.path.dirname(os.path.realpath(__file__))
m = scipy.io.loadmat('{}/AUD_Ret_1000.mat'.format(datadir))
A = m['A'].squeeze()
Pnpy = plfit.plfit(A)
#Pfor = plfit.plfit(A)
#Pfor_nosmall = plfit.plfit(A)
#Pcy = plfit.plfit(A)
Py = plfit.plfit_py(A)


# for comparison
# r_for_nosmall = Pfor_nosmall.plfit(usefortran=False, verbose=True, quiet=False, discrete=False, nosmall=True)

t0 = time.time()
# r_for = Pfor.plfit(usefortran=False, verbose=True, quiet=False, discrete=False, nosmall=False)
t1 = time.time()
# r_cy  = Pcy.plfit(usecy=True, verbose=True, quiet=False, discrete=False, nosmall=False)
t2 = time.time()
r_ppy = Py.plfit(nosmall=False)
t3 = time.time()
r_npy = Pnpy.plfit(usefortran=False, usecy=False, verbose=True, quiet=False, discrete=False, nosmall=False)
t4 = time.time()
#pdb.set_trace()


print("xmin,alpha for 4 different implementations: ")
print("npy: ",r_npy)
print("ppy: ",r_ppy)
# print("for: ",r_for)
# print("cy:  ",r_cy)
# print("nosmall: ",r_for_nosmall)
print()

if 'results' not in locals():
    t5 = time.time()
    results = powerlaw.Fit(A[A>0])
    t6 = time.time()

print("Timing and xmin,alpha for powerlaw & 4 implementations: ")
print(t6-t5,"powerlaw: ",results.power_law.xmin, results.power_law.alpha)
print(t4-t3,"npy: ",r_npy)
print(t3-t2,"ppy: ",r_ppy)
# print(t2-t1,"cy: ",r_cy)
# print(t1-t0,"cy:  ",r_for)
print()


print("Differences between plfit and powerlaw: ")
print("Powerlaw alpha, xmin: ",results.power_law.alpha, results.power_law.xmin)
print("npy: ",[(x1-x2) for x1,x2 in zip(r_npy,(results.power_law.xmin, results.power_law.alpha, ))])
print("ppy: ",[(x1-x2) for x1,x2 in zip(r_ppy,(results.power_law.xmin, results.power_law.alpha, ))])
# print("for: ",[(x1-x2) for x1,x2 in zip(r_for,(results.power_law.xmin, results.power_law.alpha, ))])
# print("cy:  ",[(x1-x2) for x1,x2 in zip(r_cy ,(results.power_law.xmin, results.power_law.alpha, ))])


# Below are some plots used for debugging

plt.close('all')

fig1,ax1=plt.subplots(1,1)
ax1.plot(results.sigmas[np.isfinite(results.sigmas)], label='powerlaw')
ax1.plot(Py._sigma, label='py')
#plot(Pfor._sigma, label='for')
#plot(Pfor_nosmall._sigma, label='for_nosmall')
ax1.plot(Pnpy._sigma, label='npy')
#plot(Pcy._sigma, label='cy')
ax1.set_ylim(0,5)
ax1.set_ylabel("$\sigma$")
ax1.set_xlabel("$x_{min}$ index")
ax1.legend(loc='best')

fig2,ax2=plt.subplots(1,1)
ax2.plot(np.array(results.Ds)[np.isfinite(results.alphas)], linewidth=2, label='powerlaw')
ax2.plot(Py._xmin_kstest, linewidth=3, alpha=0.6, linestyle='--', label='py')
ax2.plot(Pnpy._xmin_kstest, linewidth=3, alpha=0.6, linestyle=':', label='npy')
#plot(Pcy._xmin_kstest, linewidth=3, alpha=0.6, linestyle=':', label='cy')
#plot(Pfor._xmin_kstest, linewidth=3, alpha=0.6, linestyle=':', label='for')
#plot(Pfor_nosmall._xmin_kstest, linewidth=3, alpha=0.6, linestyle='-', label='for_nosmall')
ax2.set_ylabel("$D_{KS}$")
ax2.set_xlabel("$x_{min}$ index")
ax2.legend(loc='best')

plt.show()

print("What is the KS distance of the very last value, which in reality is undefined?")
print("(As of Jan 2014, this value is explicitly excluded)")
the_vars=locals()
print([(x,the_vars[x]._xmin_kstest[-1]) for x in "Py,Pnpy".split(',')])

