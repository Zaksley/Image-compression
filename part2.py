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
    BD = np.array(A.copy(), dtype='float64')
    Qright = np.eye(m)
    for i in range(k) : # 0 ... k-1
        if i < m-1 :
            X = BD[i:,i]
            alpha = -np.sign(X[0])*np.linalg.norm(X)
            Y = np.zeros(len(X))
            Y[0] = alpha
            H = houseHolder(X,Y)
            replace = np.dot(H,BD[i:,i:])
            BD[i:,i:] = replace
            Qleft[:,i:] = np.dot(Qleft[:,i:],H)
        # print(np.linalg.norm(A-np.dot(Qleft,np.dot(BD,Qright))))
        if i < n-2:
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

def testBidiagonal():
    A = np.array([[21,2,3],
                  [4, 6,9],
                  [3,11,32]])
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
    for i in range(n):
        if(DB1[i,i] > -epsilon and DB1[i,i] < epsilon):
            estDiag=False
    for i in range(n-1):
        if(DB1[i,i+1] > -epsilon and DB1[i,i+1] < epsilon):
            estDiag=False
    for i in range(n-1): # 0 1  
        for j in range(i+1): # 
            if(DB1[i+1,j] < -epsilon or DB1[i+1,j] > epsilon):
                estDiag=False
    print("Is Resulting matrix is equal to A ? " + str(res == A))
    print("Est-ce que la matrice BiDiag est bidiagonale ? " + str(estDiag))

if __name__ == '__main__':
    testBidiagonal()