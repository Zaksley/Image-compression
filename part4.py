    # Librairies pour afficher les images & gérer les matrices
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


from part2 import bidiagonal_transformation
from part3 import QRDecomposition

# Transforme A en A = U * S * V
def factorisation_SVD(A): 
    (n, m) = (A.shape[0], A.shape[1])
    
    # Récupération de la bidiagonale
    BD = bidiagonal_transformation(A, n, m)[1]

    # Application des transformations QR sur la matrice bidiagonale
    (U, S, V) = QRDecomposition(BD) 
    return (U, S, V) 


# Compresse l'image représenté par la matrice A par un coefficient k
def compress(A, k): 

    # Récupération de la matrice bidiagonale
    (n, m) = (A.shape[0], A.shape[1])

    R = np.zeros((n, m))
    G = np.zeros((n, m))
    B = np.zeros((n, m))

    # Créer 3 matrices RGB
    for i in range(n):
        for j in range(m):
            R[i][j] = A[i][j][0]
            G[i][j] = A[i][j][1]
            B[i][j] = A[i][j][2]

    #Applique transformations SVD
    (U_B, S_B, V_B) = factorisation_SVD(R)
    (U_G, S_G, V_G) = factorisation_SVD(G)
    (U_B, S_B, V_B) = factorisation_SVD(G)

    # Annulation des termes diagonaux dans S
    for i in range(k, n):
        S_B[i][i] = 0
        S_G[i][i] = 0
        S_R[i][i] = 0

    # Récuperation des matrices RGB
    R = np.dot(np.dot(U_R, S_R), V_R)
    G = np.dot(np.dot(U_G, S_G), V_G)
    B = np.dot(np.dot(U_B, S_B), V_B)

    # Insertion des valeurs précédemment calculés
    for i in range(n):
        for j in range(m):
            A[i][j][0] = R[i][j]
            A[i][j][1] = G[i][j]
            A[i][j][2] = B[i][j]

    return A

        # TESTS
# ====================

    # Calcule compression de l'image
image = mpimg.imread("essai.png")
k = 15
compressed_image = compress(image, k)

    # Affiche résultats
plt.imshow(image)
plt.show()  

plt.imshow(compressed_image)
plt.show()

# ===================== 
