    # Librairies pour afficher les images & gérer les matrices
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


from part2 import bidiagonal_transformation
from part3 import QRDecomposition

# Transforme A en A = U * S * V
def factorisation_SVD(A): 

    # Récupération de la matrice bidiagonale
    (n, m) = (A.shape[0], A.shape[1])
    BD = bidiagonal_transformation(A, n, m) 

    # Application des transformations QR sur la matrice bidiagonale
    (U, S, V) = transform_QR(BD) 
    return (U, S, V) 


# Compresse l'image représenté par la matrice A par un coefficient k
def compress(A, k): 

    (U, S, V) = factorisation_SVD(A)
        # Annulation des termes diagonaux dans S
    for i in range(k, np.shape(A)[0]):
        S[i][i] = 0
    print("erreur")
    return (np.dot(np.dot(U, S), V))

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
