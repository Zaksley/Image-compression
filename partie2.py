import numpy as np


def bidiagonal_transformation(A,n,m):
    '''
        Cette fonction calcule la factorisation de A sous la forme bidiagonale.

        @param A : Matrice initiale de l'image.
        @param n : Le nombre de lignes de la matrice.
        @param m : Le nombre de colonnes de la matrice.
        @returns {Qleft,BD,Qright} 
    '''
    k = min(n,m)
    Qleft = np.eye((m,k))
    BD = np.copy(A)
    Qright = np.eye((k,n))
    for i in range(k-1) :
        X = BD[i:,i]
        alpha = -np.sign(X[0])*np.linalg.norm(X)
        Y = np.zeros(i)
        Y[0] = alpha
        