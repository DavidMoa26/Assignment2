# Group members: 
# 1. Shifra Avigdor 207067125
# 2. Daniel Boguslavsky 207915729
# 3. David Moalem 203387337
# 4. Ethan Stryzhack 336244959

import numpy as np
from numpy.linalg import norm
from colors import bcolors
import numpy as np


def jacobi_iterative(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")

    while k <= N:
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * X0[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)

def gauss_seidel(A, b, X0, TOL=1e-16, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    print( "Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)

def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))
    s = np.sum(np.abs(mat), axis=1) - d 
    return np.all(d > s)


if __name__ == '__main__':
    A = np.array([
        [3, -1, 1],
        [0, 2, -1], 
        [1, 1, -3]
        ])
    b = np.array([4, -1, -3])
    x = np.zeros_like(b, dtype=np.double)

    
    try:
        if not is_diagonally_dominant(A):
            print("The matrix does not have diagonally dominant")
        else:
            print("======== Jacobi Method ===========")
            solution = jacobi_iterative(A,b,x)
            print(bcolors.OKBLUE,"\nApproximate solution:", solution)

            print("======== Gauss seidel Method ===========")
            solution = gauss_seidel(A, b, x)
            print(bcolors.OKBLUE,"\nApproximate solution:", solution)
            

    except ValueError as e:
        print(str(e))
