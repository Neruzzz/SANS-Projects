import numpy as np
from numpy import linalg as la
import scipy.stats as scpy
import matplotlib.pyplot as plt
import random as rnd

M =  10000 #Sample number

u1 = 3     #Mean (u1, u2)
u2 = 3

c11 = 5/2  #Matrix COV
c12 = -1/2
c21 = -1/2
c22 = 5/2

Mean = np.array([u1, u2])
Cov = np.array([[c11, c12], [c21, c22]])
Gaussian = scpy.multivariate_normal.rvs(mean = Mean, cov = Cov, size = M)

x = []
y = []

eval, evec = la.eig(Cov)
#origin = np.array([[u1, u2], [u1, u2]])
origin = [u1, u2]

for i in range(M):
    x = np.append(x, [Gaussian[i][0]], axis = 0)
    y = np.append(y, [Gaussian[i][1]], axis = 0)

print(eval, evec)

soa = np.array([[u1, u2, 
                 eval[0] * evec[0][0], 
                 eval[0] * evec[1][0]]])

soa1 = np.array([[u1, u2, 
                  eval[1] * evec[0][1], 
                  eval[1] * evec[1][1]]])

X, Y, U, V = zip(*soa)
X1, Y1, U1, V1 = zip(*soa1)

plt.scatter(x, y)

plt.quiver(X, Y, U, V, angles='xy', scale_units='xy',color = ['r'], scale=1)
plt.quiver(X1, Y1, U1, V1, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.xlim(-4,10)
plt.ylim(-4,10)
plt.axis('equal')
plt.show()