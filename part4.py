    # Librairies pour afficher les images & gérer les matrices
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

from part2 import bidiagonal_transformation
# from part3 import QRDecomposition

# Transforme A en A = U * S * V
def factorisation_SVD(A): 
    (n, m) = (A.shape[0], A.shape[1])
    
    # Récupération de la bidiagonale
    U,S,V = bidiagonal_transformation(A, n, m)

    # Application des transformations QR sur la matrice bidiagonale
    while np.linalg.norm(np.diag(S,1))/np.linalg.norm(np.diag(S)) > 1e-2:
        Q1,R1=np.linalg.qr(S.T)
        Q2,R2=np.linalg.qr(R1.T)
        S=R2
        U=np.dot(U,Q2)
        V=np.dot(Q1.T,V)
    return (U, S, V)


# Compresse l'image représenté par la matrice A par un coefficient k
def compress(image, k): 
    A=np.copy(image)
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
    # (U_R, S_R, V_R) = factorisation_SVD(R)
    # (U_G, S_G, V_G) = factorisation_SVD(G)
    # (U_B, S_B, V_B) = factorisation_SVD(B)  
    
    #On utilise la factorisation SVD de Numpy pour des soucis d'optimisations (On obtient des résultats plus rapides donc on peut affiner la précision)
    (U_R, S_R, V_R) = np.linalg.svd(R)
    (U_G, S_G, V_G) = np.linalg.svd(G)
    (U_B, S_B, V_B) = np.linalg.svd(B)

    # Récuperation des matrices RGB
    R = np.dot(np.dot(U_R[:,:k], np.diag(S_R[:k])), V_R[:k,:])
    G = np.dot(np.dot(U_G[:,:k], np.diag(S_G[:k])), V_G[:k,:])
    B = np.dot(np.dot(U_B[:,:k], np.diag(S_B[:k])), V_B[:k,:])
    
    # Insertion des valeurs précédemment calculés
    for i in range(n):
        for j in range(m):
            A[i][j][0] = R[i][j]
            A[i][j][1] = G[i][j]
            A[i][j][2] = B[i][j]

    return A

def distance_images(M_normal, M_compressed):
    n  = np.shape(M_normal)[0]
    m = np.shape(M_normal)[1]
    distance = 0

        #Calcul de la distance euclidienne 
    for i in range(n):
        for j in range(m):
            for k in range (3):
                distance += (M_normal[i][j][k] - M_compressed[i][j][k])**2

    return math.sqrt(distance)

def Efficiency_kCompress(M_normal):

        #Apparition du graphe

    #*
    # * x ne doit pas être supérieur au minimum (longueur, largeur) de l'image
    # * Ceci est lié au produit des matrices dans la réunion des matrices gérés par la facto SVD
    #*     
    x = range(1, 300, 5)

    plt.plot(x, [distance_images(M_normal, compress(M_normal, k)) for k in x ])
    plt.xlabel("Rang de compression k")
    plt.ylabel("Distance euclidienne entre les images")
    plt.show()




image = mpimg.imread("essai.png")

        # Calcul distance
# ====================
"""
Efficiency_kCompress(image)
"""

# ====================


        # TESTS
# ====================

    # Calcule compression de l'image

k = 10
compressed_image = compress(image, k)
print(compressed_image)

    # Affiche résultats
plt.imshow(image)
plt.savefig("original") 

plt.imshow(compressed_image)
plt.savefig('compress')

# ===================== 
