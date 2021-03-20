import numpy as np

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
    R = A.copy()
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


def SVD_Decomposition_BidiagonalMatrix_Opti(BD):
    '''
        BD représente la matrice bi-diagonale
    '''
    (n,m) = BD.shape
    S = BD.copy()
    U = np.eye(n)
    V = np.eye(m)
    
    for i in range(min(m-2,n-1)):
        (Q1,R1) = QR_Transformation(np.transpose(S))
        (Q2,R2) = QR_Transformation(np.transpose(R1))
        S = R2
        U = np.dot(U,Q2)
        V = np.dot(np.transpose(Q1),V)
        
    return (U,S,V)
