import numpy as np
import scipy.stats as scpy
import matplotlib.pyplot as plt
import random as rnd

N = 100 #sets
M = 1000 #samples

#BERNOULLI (binom with n = 1)
p = 3/4
n = 1
EofX = n*p
VarofX = EofX*(1-p)

A = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    A = np.append(A, [np.array(scpy.bernoulli.rvs(p, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

Z = np.multiply(np.subtract(A, EofX), 1/VarofX)

X_asterisk = np.empty((0,M), float)
for columna in Z.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= np.sqrt(N)
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X**(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X**(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X**(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X**(m)')
plt.ylabel('m')
plt.suptitle('Bernoulli distribution')
plt.show()


#UNIFORM [a, b]
a = 0
b = 1
EofX = (b + a)/2
VarofX = ((b-a)^2)/12

A = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    A = np.append(A, [np.array(scpy.uniform.rvs(loc = a, scale = b - a, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

Z = np.multiply(np.subtract(A, EofX), 1/VarofX)

X_asterisk = np.empty((0,M), float)
for columna in Z.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= np.sqrt(N)
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X**(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X**(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X**(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X**(m)')
plt.ylabel('m')
plt.suptitle('Uniform distribution')
plt.show()

#EXPONENTIAL
lamb = 1
EofX = 1/lamb
VarofX = 1/(lamb^2)
A = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    A = np.append(A, [np.array(scpy.expon.rvs(scale = 1/lamb, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

Z = np.multiply(np.subtract(A, EofX), 1/VarofX)

X_asterisk = np.empty((0,M), float)
for columna in Z.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= np.sqrt(N)
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X**(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X**(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X**(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X**(m)')
plt.ylabel('m')
plt.suptitle('Exponential distribution')
plt.show()

#GAUSSIAN
EofX = 1
VarofX = 1

A = np.empty((0, M), int) #crea una matriz de 0 rows y 1000 columns que está vacia y va a recibir floats

for fila in range(N):
    A = np.append(A, [np.array(scpy.norm.rvs(loc = EofX, scale = VarofX, size=M))], axis = 0)#añade la fila (axis = 0 es filas, 1 para columnas), esta fila es la generación de 1000 samples

Z = np.multiply(np.subtract(A, EofX), 1/VarofX)

X_asterisk = np.empty((0,M), float)
for columna in Z.T:
    sum = 0.0
    for sample in columna:
       sum += sample
    sum /= np.sqrt(N)
    X_asterisk = np.append(X_asterisk, sum)

plt.subplot(1, 2, 1)
plt.title('X**(m) graph')
plt.plot(X_asterisk)
plt.ylabel('X**(m)')
plt.xlabel('m')
plt.subplot(1, 2, 2)
plt.title('X**(m) histogram')
plt.hist(X_asterisk)
plt.xlabel('X**(m)')
plt.ylabel('m')
plt.suptitle('Gaussian distribution')
plt.show()


