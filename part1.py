import numpy as np
from math import *

    # Return the matrice U = (X-Y) / || X - Y ||
def get_U(X, Y):
    
    U = X - Y
    
    norme = np.linalg.norm(U)
    U = U / norme
    return U

def houseHolder(U):

    size = np.shape(U)[1]
    

    H = np.identity(size) - (2 * np.dot(U, U.T))
    
    return H

X = np.array( [[3, 4, 0]] )
Y = np.array( [[0, 0, 5]] )


print(houseHolder(get_U(X, Y)))