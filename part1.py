import numpy as np
from math import *

# Algorithme de la transformation de Householder non optimisée
# Prend X et Y vecteurs de même taille et de même norme.
# Complexité de n**2


def houseHolder(X, Y):
    size = np.shape(X)[0]
    U = X - Y
    norme = np.linalg.norm(np.dot(U, U.T))
    H = np.identity(size) - (2 * np.dot(U, U.T))/norme
    return H

# Algorithme du calcul de la matrice de Householder par un vecteur
# Prend X, Y, V vecteurs de même taille dont X et Y sont de même norme.
# Complexité de 4n


def houseHolderOpti(X, Y, V):
    size = np.shape(X)[0]
    U = (X - Y)/np.linalg.norm(X-Y)
    alpha = np.dot(U.T, V)
    return V - 2*alpha*U


if __name__ == '__main__':
    print("Test de la transformation de Householder non optimisée")
    X = np.array([[3], [4], [0]])
    Y = np.array([[0], [0], [5]])
    print(houseHolder(X, Y), "\n")

    print("Test du calcul de la matrice Householder par un vecteur optimisée")
    X = np.array([[3], [4], [0]])
    Y = np.array([[0], [0], [5]])
    V = np.array([[5], [10], [-5]])
    print(houseHolderOpti(X, Y, V), "\n")
