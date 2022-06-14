import numpy as np
import scipy.stats as scpy
import matplotlib.pyplot as plt
import random as rnd

N = 100 #sets
M = 1000 #samples

#BERNOULLI
M_ber = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    M_ber = np.append(M_ber, [np.array(scpy.bernoulli.rvs(3/4, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

X_asterisk = np.empty((0,M), float)
for columna in M_ber.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= N
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X*(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X*(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X*(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X*(m)')
plt.ylabel('m')
plt.suptitle('Bernoulli distribution')
plt.show()

#UNIFORM
M_unif = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    M_unif = np.append(M_unif, [np.array(scpy.uniform.rvs(1, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

X_asterisk = np.empty((0,M), float)
for columna in M_unif.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= N
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X*(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X*(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X*(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X*(m)')
plt.ylabel('m')
plt.suptitle('Uniform distribution')
plt.show()

#GAUSSIAN
M_gauss = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    M_gauss = np.append(M_gauss, [np.array(scpy.norm.rvs(loc=1, scale=1, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

X_asterisk = np.empty((0,M), float)
for columna in M_gauss.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= N
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X*(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X*(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X*(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X*(m)')
plt.ylabel('m')
plt.suptitle('Gaussian distribution')
plt.show()