import numpy as np

# Algorithme de la transformation de Householder non optimisée.
# Prend X et Y vecteurs de même taille et de même norme.
# Complexité de n**2 avec n la taille des vecteurs.
def houseHolder(X, Y):
    size = np.shape(X)[0]
    U = X - Y
    matU=np.array([U])
    Udot = np.dot(matU.T,matU)
    norme = np.linalg.norm(Udot)
    H = np.identity(size) - (2 * Udot)/norme
    return H

# Algorithme du calcul de la matrice de Householder par un vecteur.
# Prend X, Y, V vecteurs de même taille dont X et Y sont de même norme.
# Complexité de 4n avec n la taille des vecteurs.
def houseHolderOpti(X, Y, V):
    size = np.shape(X)[0]
    U = (X - Y)/np.linalg.norm(X-Y)
    alpha = np.dot(U, V)
    return V - 2*alpha*U

# Algorithme du calcul de la matrice de Householder avec un ensemble de vecteurs.
# Prend X, Y vecteurs de même taille et de même norme et M une matrice dont le nombre de colonnes est le même que la taille de X et Y.
# Complexité de 4n**2
def houseHolderGen(X, Y, M):
    lines = np.shape(M)[0]
    columns = np.shape(X)[0]
    res = np.zeros((lines, columns))
    for i in range(lines):
        res[i] = houseHolderOpti(X, Y, M[i])
    return res

#Tests des différentes fonctions
if __name__ == '__main__':
    print("Test de la transformation de Householder non optimisée:")
    X = np.array([3, 4, 0])
    Y = np.array([0, 0, 5])
    print(houseHolder(X, Y), "\n")

    print("Test du calcul de la matrice Householder par un vecteur optimisée:")
    X = np.array([3, 4, 0])
    Y = np.array([0, 0, 5])
    V = np.array([5, 10, -5])
    print(houseHolderOpti(X, Y, V), "\n")

    print("Test du calcul de la matrice Householder avec un ensemble de vecteurs:")
    X = np.array([3, 4, 0])
    Y = np.array([0, 0, 5])
    V = np.array([5, 10, -5])
    M = np.array([V, V, V, V, V])
    print(houseHolderGen(X, Y, M),"\n")
