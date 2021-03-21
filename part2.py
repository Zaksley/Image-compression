import numpy as np
from part1 import houseHolder 
import math


def bidiagonal_transformation(A,n,m):
    '''
        Cette fonction calcule la factorisation de A sous la forme bidiagonale.

        @param A : Matrice initiale de l'image.
        @param n : Le nombre de lignes de la matrice.
        @param m : Le nombre de colonnes de la matrice.
        @returns {Qleft,BD,Qright} 
    '''
    k = min(n,m)
    BD = np.array(A.copy(), dtype='float64')
    Qleft = np.eye(n)
    Qright = np.eye(m)
    for i in range(k) : # 0 ... k-1
        if i < n-1 :
            X = BD[i:,i]
            alpha = -np.sign(X[0])*np.linalg.norm(X)
            Y = np.zeros(len(X))
            Y[0] = alpha
            H = houseHolder(X,Y)
            replace = np.dot(H,BD[i:,i:])
            BD[i:,i:] = replace
            Qleft[:,i:] = np.dot(Qleft[:,i:],H)
<<<<<<< HEAD
        # print(np.linalg.norm(A-np.dot(Qleft,np.dot(BD,Qright))))
        if i < m-2:
=======
        if i < n-2:
>>>>>>> db98fcf (Adding support in houseHolder to handler X==Y. Remove wrong bidiag in part2.)
            X = BD[i,(i+1):]
            alpha = -np.sign(X[0])*np.linalg.norm(X)
            Y = np.zeros(len(X))
            Y[0] = alpha
            H = houseHolder(X,Y)
            BD[i,(i+1):] = Y
            replace = np.dot(BD[(i+1):,(i+1):],H)
            BD[(i+1):,(i+1):] = replace
            Qright[(i+1):,:] = np.dot(H,Qright[(i+1):,:])
        # print(np.linalg.norm(A-np.dot(Qleft,np.dot(BD,Qright))))

    return (Qleft,BD,Qright)

def testBidiagonalCarree():
    A = np.array([[8,9,48],
                  [5,6,61],
                  [0,6,8]])
    (Ql1,DB1,Qr1) = bidiagonal_transformation(A,3,3)
    print("Matrice Qleft :")
    print(Ql1)
    print("Matrice Bidiagonale :")
    print(DB1)
    print("Matrice Qright :")
    print(Qr1)
    print("Matrice Qleft*BiDiag*Qright :")
    res = np.dot(Ql1,np.dot(DB1,Qr1));
    print(res)
    estDiag = True
    (n,m) = DB1.shape
    epsilon = 10**(-14)
    # Test si la diagonale n'est pas nulle
    for i in range(n):
        if(DB1[i,i] > -epsilon and DB1[i,i] < epsilon):
            estDiag=False
    # Test si la deuxième diagonale n'est pas nulle nonplus
    for i in range(n-1):
        if(DB1[i,i+1] > -epsilon and DB1[i,i+1] < epsilon):
            estDiag=False
    
    # Test si la partie triangle bas de la matrice bidiagonale est bien nulle.
    for i in range(n-1):  
        for j in range(i+1): 
            if(DB1[i+1,j] < -epsilon or DB1[i+1,j] > epsilon):
                estDiag=False

    # Test si la partie triangle haut de la matrice bidiagonale est bien nulle.
    for i in range(n-2):  
        for j in range(i+1): 
            if(DB1[j,i+2] < -epsilon or DB1[j,i+2] > epsilon):
                estDiag=False

    # Test si  les deux matrices, celle de départ et celle calculée avec les valeurs de sortie de bidiag sont les même ou non.
    isEqual = True
    epsilon = 10**(-10)
    for i in range(n):
        for j in range(n):
            d=abs(A[i,j] - res[i,j])
            if(d>epsilon):
                print(" " + str(A[i,j]) + " != " + str(res[i,j]) )
                isEqual = False

    print("[TEST] Carree : Est-ce que la matrice resultante est bien égale à la matrice initiale ? "+str(isEqual))
    print("[TEST] Carree : Est-ce que la matrice BiDiag est bidiagonale ? " + str(estDiag))


def testBidiagonalRect():
    A = np.array([[21,598,3,98],
                  [43, 6,9,159],
                  [3,161,32,78]])
    (Ql1,DB1,Qr1) = bidiagonal_transformation(A,3,4)
    print("Matrice Qleft :")
    print(Ql1)
    print("Matrice Bidiagonale :")
    print(DB1)
    print("Matrice Qright :")
    print(Qr1)
    print("Matrice Qleft*BiDiag*Qright :")
    res = np.dot(Ql1,np.dot(DB1,Qr1));
    print(res)
    estDiag = True
    (n,m) = DB1.shape
    epsilon = 10**(-14)
    # Test si la diagonale n'est pas nulle
    for i in range(n):
        if(DB1[i,i] > -epsilon and DB1[i,i] < epsilon):
            estDiag=False
    # Test si la deuxième diagonale n'est pas nulle nonplus
    for i in range(n-1):
        if(DB1[i,i+1] > -epsilon and DB1[i,i+1] < epsilon):
            estDiag=False
    
    # Test si la partie triangle bas de la matrice bidiagonale est bien nulle.
    for i in range(n-1):  
        for j in range(i+1): 
            if(DB1[i+1,j] < -epsilon or DB1[i+1,j] > epsilon):
                estDiag=False

    # Test si la partie triangle haut de la matrice bidiagonale est bien nulle.
    for i in range(n-2):  
        for j in range(i+1): 
            if(DB1[j,i+2] < -epsilon or DB1[j,i+2] > epsilon):
                estDiag=False

    # Test si  les deux matrices, celle de départ et celle calculée avec les valeurs de sortie de bidiag sont les même ou non.
    isEqual = True
    epsilon = 10**(-10)
    for i in range(n):
        for j in range(n):
            d=abs(A[i,j] - res[i,j])
            if(d>epsilon):
                isEqual = False

    print("[TEST] Rectangle : Est-ce que la matrice resultante est bien égale à la matrice initiale ? "+str(isEqual))
    print("[TEST] Rectangle : Est-ce que la matrice BiDiag est bidiagonale ? " + str(estDiag))


if __name__ == '__main__':
    testBidiagonalCarree()
    testBidiagonalRect()