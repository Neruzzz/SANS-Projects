import numpy as np
from numpy import linalg as la
import scipy.stats as scpy
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random as rnd
from sympy import *
import math

n = 50

def subspaces_matrix(matrix):
    u, s, vt = la.svd(matrix)
    u_tra, s_tra, vt_tra = la.svd(matrix.T)
    u_result = []
    vt_result = []
    vt_row_result = []
    vt_result_tra = []

    for i in range(la.matrix_rank(matrix)):
        u_result = np.append(u_result, [u[:, i]]) #col(matrix)
        vt_result = np.append(vt_result, [vt[-1*(i+1)]]) #ker(matrix)
        vt_result_tra = np.append(vt_result_tra, [vt_tra[-1*(i+1)]]) #ker(matrixT)
        vt_row_result = np.append(vt_row_result, [vt[i,:]]) #col(matrixT)

    return u_result, vt_result, vt_row_result, vt_result_tra

def transf(m00, m01, m10, m11):
  transformation = [[],[]]
  for i in range(n):
    x0 = m00*math.cos(i*2*math.pi/50)
    x1 = m10*math.cos(i*2*math.pi/50)
    y0 = m01*math.sin(i*2*math.pi/50)
    y1 = m11*math.sin(i*2*math.pi/50)
    transformation = np.append(transformation, np.array([[x0 + y0], [x1 + y1]]), axis = 1)

  return transformation

#Matrix A (2x2)
a11 = 2
a12 = 2
a21 = 3
a22 = 3

#Matrix B (2x2)
b11=2
b12=1
b21=1
b22=3

#Matrix C (1x2)
c1=2
c2=-2

#Matrix D (2x1)
d1=2
d2=3

A = np.array([[a11, a12], [a21, a22]])
evalA, evecA = la.eig(A)
B = np.array([[b11, b12], [b21, b22]])
evalB , evecB = la.eig(B)
C = np.array([[c1, c2]])
D = np.array([[d1], [d2]])

origin = [0,0]

col_a, ker_a, row_a, ker_at = subspaces_matrix(A)
print("Col(A), Ker(A), Row(A), Ker(At)")
print(col_a, ker_a, row_a, ker_at)
print("\n")

col_b, ker_b, row_b, ker_bt = subspaces_matrix(B)
print("Col(B), Ker(B), Row(B), Ker(Bt)")
print(col_b, ker_b, row_b, ker_bt)
print("\n")

col_c, ker_c, row_c, ker_ct = subspaces_matrix(C)
print("Col(C), Ker(C), Row(C), Ker(Ct)")
print(col_c, ker_c, row_c, ker_ct)
print("\n")


col_d, ker_d, row_d, ker_dt = subspaces_matrix(D)
print("Col(D), Ker(D), Row(D), Ker(Dt)")
print(col_d, ker_d, row_d, ker_dt)
print("\n")

red_patch = mpatches.Patch(color='red', label = 'Col(A)')
blue_patch = mpatches.Patch(color='blue', label = 'Ker(At)')
plt.title("Matrix A")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *col_a[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_at[:], color = ['b'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

red_patch = mpatches.Patch(color='red', label = 'Row(A)')
blue_patch = mpatches.Patch(color='blue', label = 'Ker(A)')
plt.title("Matrix A")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *row_a[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_a[:], color = ['b'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

red_patch = mpatches.Patch(color='red', label = 'Col(B)')
plt.title("Matrix B")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, col_b[0],col_b[1], color = ['r'], scale=5)
plt.quiver(*origin, col_b[2],col_b[3], color = ['r'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch])
plt.show()

red_patch = mpatches.Patch(color='red', label = 'Row(B)')
plt.title("Matrix B")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, row_b[0], row_b[1], color = ['r'], scale=5)
plt.quiver(*origin, row_b[2], row_b[3], color = ['r'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch])
plt.show()

"""
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *col_c[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_ct[:], color = ['b'], scale=5)
plt.axis('equal')
plt.show()
"""

red_patch = mpatches.Patch(color='red', label = 'Row(C)')
blue_patch = mpatches.Patch(color='blue', label = 'Ker(C)')
plt.title("Matrix C")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *row_c[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_c[:], color = ['b'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

red_patch = mpatches.Patch(color='red', label = 'Col(D)')
blue_patch = mpatches.Patch(color='blue', label = 'Ker(Dt)')
plt.title("Matrix D")
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *col_d[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_dt[:], color = ['b'], scale=5)
plt.axis('equal')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

"""
plt.axvline(x=0, ymax=1, color='black')
plt.axhline(y=0, xmax=1, color='black')
plt.quiver(*origin, *row_d[:], color = ['r'], scale=5)
plt.quiver(*origin, *ker_d[:], color = ['b'], scale=5)
plt.axis('equal')
plt.show()
"""

#####################AAAAAAAAAAAAAAAAAAA#############################
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
plt.plot(tranformA[0,:], tranformA[1,:], 'b', zorder=1)
plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = ['r'], scale=1, zorder=2)
plt.quiver(X1, Y1, U1, V1, angles='xy', scale_units='xy', color = ['r'], scale=1,zorder=2)
plt.axis('equal')
plt.show()

#################BBBBBBBBBB####################
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
