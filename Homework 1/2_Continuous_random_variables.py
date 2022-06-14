import numpy as np
import scipy.stats as scpy
import matplotlib.pyplot as plt
import random as rnd

N = 10000

#UNIFORM
uni = scpy.uniform.rvs(size=1000)

plt.subplot(1, 2, 1)
plt.title('Uniform graph')
plt.plot(uni[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Uniform histogram')
plt.hist(uni)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Uniform distribution')
plt.show()

#EXPONENTIAL
expon = scpy.expon.rvs(size=1000)

plt.subplot(1, 2, 1)
plt.title('Exponential graph')
plt.plot(expon[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Exponential histogram')
plt.hist(expon)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Exponential distribution')
plt.show()

#GAUSSIAN 1 1
gauss1 = scpy.norm.rvs(loc=1, scale=1, size=1000) #loc stands for mean and scale stands for var

plt.subplot(1, 2, 1)
plt.title('Gasussian graph')
plt.plot(gauss1[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Gaussian histogram')
plt.hist(gauss1)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Gaussian distribution mean=1 var=1')
plt.show()

#GAUSSIAN 1 5
gauss2 = scpy.norm.rvs(loc=1, scale=5, size=1000) #loc stands for mean and scale stands for var

plt.subplot(1, 2, 1)
plt.title('Gasussian graph')
plt.plot(gauss2[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Gaussian histogram')
plt.hist(gauss2)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Gaussian distribution mean=1 var=5')
plt.show()
