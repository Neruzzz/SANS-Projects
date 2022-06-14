import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math

def transf(m00, m01, m10, m11):
  transformation = [[],[]]
  for i in range(55):
    x0 = m00*math.cos(i*2*math.pi/50)
    x1 = m10*math.cos(i*2*math.pi/50)
    y0 = m01*math.sin(i*2*math.pi/50)
    y1 = m11*math.sin(i*2*math.pi/50)
    transformation = np.append(transformation, np.array([[x0 + y0], [x1 + y1]]), axis = 1)

  return transformation

#Matrix A (2x2)
a11 = np.sqrt(2)/2
a12 = np.sqrt(2)/2
a21 = np.sqrt(2)/2
a22 = -np.sqrt(2)/2
A = np.array([[a11, a12], [a21, a22]])
evalA, evecA = la.eig(A)

#Matrix B (2x2)
b11 = 3
b12 = 5
b21 = 5
b22 = 2
B = np.array([[b11, b12], [b21, b22]])
evalB, evecB = la.eig(B)

#Matrix C (2x2)
c11 = 3
c12 = 1
c21 = 1
c22 = 2
C = np.array([[c11, c12], [c21, c22]])
evalC, evecC = la.eig(C)



print('Eigenvalues from A: ', evalA)
print('\n')
print('Eigenvectors from A: ', evecA)
print('\n')

tranformA = transf(a11,a12,a21,a22)

soa = np.array([[0, 0,
                 evalA[0] * evecA[0][0],
                 evalA[0] * evecA[1][0]]])

soa1 = np.array([[0, 0,
                  evalA[1] * evecA[0][1],
                  evalA[1] * evecA[1][1]]])

X, Y, U, V = zip(*soa)
X1, Y1, U1, V1 = zip(*soa1)

plt.title("Matrix A, Eigenvectors")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.plot(tranformA[0,:], tranformA[1,:], 'b')
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.quiver(X1, Y1, U1, V1, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.axis('equal')
plt.show()

print('Eigenvalues from B: ', evalB)
print('\n')
print('Eigenvectors from B: ', evecB)
print('\n')

tranformB = transf(b11,b12,b21,b22)

soa = np.array([[0, 0,
                 evalB[0] * evecB[0][0],
                 evalB[0] * evecB[1][0]]])

soa1 = np.array([[0, 0,
                  evalB[1] * evecB[0][1],
                  evalB[1] * evecB[1][1]]])

X, Y, U, V = zip(*soa)
X1, Y1, U1, V1 = zip(*soa1)

plt.title("Matrix B, Eigenvectors")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.plot(tranformB[0,:], tranformB[1,:], 'b')
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.quiver(X1, Y1, U1, V1, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.axis('equal')
plt.show()

print('Eigenvalues from C: ', evalC)
print('\n')
print('Eigenvectors from C: ', evecC)

tranformC = transf(c11,c12,c21,c22)

soa = np.array([[0, 0,
                 evalC[0] * evecC[0][0],
                 evalC[0] * evecC[1][0]]])

soa1 = np.array([[0, 0,
                  evalC[1] * evecC[0][1],
                  evalC[1] * evecC[1][1]]])

X, Y, U, V = zip(*soa)
X1, Y1, U1, V1 = zip(*soa1)

plt.title("Matrix C, Eigenvectors")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.plot(tranformC[0,:], tranformC[1,:], 'b')
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.quiver(X1, Y1, U1, V1, angles='xy', scale_units='xy', color = ['r'], scale=1)
plt.axis('equal')
plt.show()
