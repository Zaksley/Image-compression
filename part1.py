import numpy as np
from math import *

# Algorithme de la transformation de Householder non optimisée
# Prend X et Y de même taille.
# Complexité de n**2
def houseHolder(X,Y):
    size = np.shape(X)[0] 
    U = X - Y
    norme = np.linalg.norm(np.dot(U,U.T))
    H = np.identity(size) - (2 * np.dot(U, U.T))/norme
    return H

if __name__=='__main__':
    print("Test de la transformation de Householder non optimisée")
    X = np.array( [[3], [4], [0]] )
    Y = np.array( [[0], [0], [5]] )
    print(houseHolder(X, Y))