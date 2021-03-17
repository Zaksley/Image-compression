import numpy as np
from part1 import houseHolder 

def bidiagonal_transformation(A,n,m):
    '''
        Cette fonction calcule la factorisation de A sous la forme bidiagonale.

        @param A : Matrice initiale de l'image.
        @param n : Le nombre de lignes de la matrice.
        @param m : Le nombre de colonnes de la matrice.
        @returns {Qleft,BD,Qright} 
    '''
    k = min(n,m)
    Qleft = np.eye(n)
    BD = np.copy(A)
    Qright = np.eye(m)
    for i in range(k) :
        if i < m-1 :
            X = BD[i:,i]
            alpha = -np.sign(X[0])*np.linalg.norm(X)
            Y = np.zeros(1 if i==0 else i)
            Y[0] = alpha
            H = houseHolder(X,Y)
            replace = np.dot(H,BD[i:,i:])
            BD[i:,i:] = replace
            Qleft[:,i:] = np.dot(Qleft[:,i:],H)
        if i < n-2:
            X = BD[i,(i+1):]
            alpha = -np.sign(X[0])*np.linalg.norm(X)
            Y = np.zeros(1 if i==0 else i)
            Y[0] = alpha
            H = houseHolder(X,Y)
            BD[i,(i+1):] = Y
            replace = np.dot(BD[(i+1):,(i+1):],H)
            BD[(i+1):,(i+1):] = replace
            Qright[(i+1):,:] = np.dot(H,Qright[(i+1):,:])
        # print("===DEB===")
        # print(Qleft)
        # print("=========")
        # print(BD)
        # print("=========")
        # print(Qright)
        # print("===FIN===")
    return (Qleft,BD,Qright)


def tests():
    A = np.array([[21,2,3],
                  [4, 6,9],
                  [3,11,32]])
    (Ql1,DB1,Qr1) = bidiagonal_transformation(A,3,3)
    (Ql2,DB2,Qr2) = np.linalg.svd(A)
    
    print(Ql1)
    # print(Ql2)
    # print("=========")
    print(DB1)
    # print(DB2)
    # print("=========")
    print(Qr1)
    # print(Qr2)
    res = np.dot(Ql1,np.dot(DB1,Qr1));
    print(res)

tests()