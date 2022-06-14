import numpy as np
import scipy.stats as scpy
import matplotlib.pyplot as plt
import random as rnd

N = 10000
p = 3/4

#BERNOULLI
ber = scpy.bernoulli.rvs(p, size=N)

plt.subplot(1, 2, 1)
plt.title('Bernoulli graph')
plt.plot(ber[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Bernoulli histogram')
plt.hist(ber)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Bernoulli distribution')
plt.show()

#BINOMIAL
bin = scpy.binom.rvs(10, p, size=N)


plt.subplot(1, 2, 1)
plt.title('Binomial graph')
plt.plot(bin[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Binomial histogram')
plt.hist(bin)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Binomial distribution')
plt.show()

#GEOMETRIC
geo = scpy.geom.rvs(p, size=N)

plt.subplot(1, 2, 1)
plt.title('Geometric graph')
plt.plot(geo[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Geometric histogram')
plt.hist(geo)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Geometric distribution')
plt.show()

#POISSON
poi = scpy.poisson.rvs(1, size=N)

plt.subplot(1, 2, 1)
plt.title('Poisson graph')
plt.plot(poi[1:1000])
plt.ylabel('X(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('Poisson histogram')
plt.hist(poi)
plt.xlabel('X(m)')
plt.ylabel('m')
plt.suptitle('Poisson distribution')
plt.show()