import numpy as np

def houseHolder(X, Y):
    '''
    Fait le calcul de la transformation de Householder non optimisée. 
    Complexité de n**2 avec n la taille des vecteurs.
    @param X: un vecteur de taille n
    @param Y: un vecteur de taille n de même norme que X
    @return une matrice carrée de taille n
    '''
    size = np.shape(X)[0]
    U = X - Y
    matU=np.array([U])
    Udot = np.dot(matU.T,matU)
    norme = np.linalg.norm(Udot)
    return np.identity(size) - (2 * Udot)/norme


def houseHolderOpti(X, Y, V):
    '''
    Fait le calcul de la matrice de Householder par un vecteur. 
    Complexité de 4n avec n la taille des vecteurs.
    @param X: un vecteur de taille n
    @param Y: un vecteur de taille n de même norme que X
    @param V: un vecteur de taille n
    @return un vecteur de taille n
    '''
    size = np.shape(X)[0]
    U = (X - Y)/np.linalg.norm(X-Y)
    alpha = np.dot(U, V)
    return V - 2*alpha*U


def houseHolderGen(X, Y, M):
    '''
    Fait le calcul de la matrice de Householder avec un ensemble de vecteurs.
    Complexité de 4n**2.
    @param X: un vecteur de taille p
    @param Y: un vecteur de taille p et de même norme que X
    @param M: une matrice de taille (n,p)
    @return une matrice de taille (n,p)
    '''
    
    lines = np.shape(M)[0]
    columns = np.shape(X)[0]
    res = np.zeros((lines, columns))
    for i in range(lines):
        res[i] = houseHolderOpti(X, Y, M[i])
    return res

#Tests des différentes fonctions
if __name__ == '__main__':
    def test_houseHolder():
        '''
        Test de la transformation de Householder non optimisée.
        '''
        print("Test de la transformation de Householder non optimisée:")
        X = np.array([3, 4, 0])
        Y = np.array([0, 0, 5])
        print(houseHolder(X, Y), "\n")
        
    def test_houseHolderOpti():
        '''
        Test du calcul de la matrice Householder par un vecteur optimisée.
        '''
        print("Test du calcul de la matrice Householder par un vecteur optimisée:")
        X = np.array([3, 4, 0])
        Y = np.array([0, 0, 5])
        V = np.array([5, 10, -5])
        print(houseHolderOpti(X, Y, V), "\n")
        
    def test_houseHolderGen():
        '''
        Test du calcul de la matrice Householder avec un ensemble de vecteurs.
        '''
        print("Test du calcul de la matrice Householder avec un ensemble de vecteurs:")
        X = np.array([3, 4, 0])
        Y = np.array([0, 0, 5])
        V = np.array([5, 10, -5])
        M = np.array([V, V, V, V, V])
        print(houseHolderGen(X, Y, M),"\n")

    test_houseHolder()
    test_houseHolderOpti()
    test_houseHolderGen()