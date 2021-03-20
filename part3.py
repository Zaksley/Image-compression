import numpy as np
import matplotlib.pyplot as plt
from random import randint
from part1 import houseHolder

def SVD_Decomposition_BidiagonalMatrix(BD,NMax):
    '''
        BD représente la matrice bi-diagonale
    '''
    (n,m) = BD.shape
    S = BD.copy()
    U = np.eye(n)
    V = np.eye(m)
    
    for i in range(NMax):
        (Q1,R1) = np.linalg.qr(np.transpose(S))
        (Q2,R2) = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U,Q2)
        V = np.dot(np.transpose(Q1),V)
        
    return (U,S,V)


def QR_Transformation(BD):
    '''
        BD est la matrice bi-diagonale inférieure 
    '''
    (n,m) = BD.shape
    R = BD.copy()
    Q = np.eye(n)

    for i in range(min(m-2,n-1)):
        
        X = np.array([R[i][i],R[i+1][i]])
        Y = np.zeros(2)
        Y[0] = np.linalg.norm(X)
    
        H = houseHolder(X,Y)
        R[i:i+2,i] = Y
        
        if (i < n-1):
            R[i:i+2,i+1] = H*R[i:i+2,i+1]
            
        Q[:,i:i+2] = np.dot(Q[:,i:i+2],H)
    
    return (Q,R)


def SVD_Decomposition_BidiagonalMatrix_Opti(BD,NMax):
    '''
        BD représente la matrice bi-diagonale
    '''
    (n,m) = BD.shape
    S = BD.copy()
    U = np.eye(n)
    V = np.eye(m)
    
    for i in range(NMax):
        (Q1,R1) = QR_Transformation(np.transpose(S))
        (Q2,R2) = QR_Transformation(np.transpose(R1))
        S = R2
        U = np.dot(U,Q2)
        V = np.dot(np.transpose(Q1),V)
        
    return (U,S,V)

def somme(BD):

    (n,m) = BD.shape
    res = 0

    for i in range(n):
        for j in range(m):
            if (i != j):
                res = res + abs(BD[i][j])

    return res

def convergence(BD,NMax):

    values = np.zeros(NMax)
    nValues = np.zeros(NMax)
    
    for i in range(NMax):
        S = SVD_Decomposition_BidiagonalMatrix_Opti(BD,i)[1]
        values[i] = somme(S)
        nValues[i] = i

    return (values,nValues)

def test():
    BD = np.array([[5,0,0,0,0],[1,3,0,0,0],[0,4,6,0,0],[0,0,7,10,0]])
                
    (values,nValues) = convergence(BD,500)
    
    plt.plot(nValues,values)
    plt.show()

test()
