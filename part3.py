import numpy as np

def SVD_Decomposition_BidiagonalMatrix(BD):
    '''
        BD représente la matrice bi-diagonale
    '''
    (n,m) = BD.shape
    S = BD.copy()
    U = np.eye(n)
    V = np.eye(m)
    
    for i in range(min(m-2,n-1)):
        (Q1,R1) = np.linalg.qr(np.transpose(S))
        (Q2,R2) = np.linalg.qr(np.transpose(R1))
        S = R2
        U = np.dot(U,Q2)
        V = np.dot(np.transpose(Q1),V)
        
    return (U,S,V)
