    # Librairies pour afficher les images & gérer les matrices
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from part2 import bidiagonal_transformation
# from part3 import QRDecomposition
from part1 import houseHolder

def factorisation_SVD(A): 
    n, m = A.shape
    
    # Récupération de la bidiagonale
    U,S,V = bidiagonal_transformation(A,n,m)
    # print(f"NORME : {np.linalg.norm(A - np.dot(Ql,np.dot(S,Qr)))}")
    # U = np.eye(n)
    # V = np.eye(n)
    # Application des transformations QR sur la matrice bidiagonale
    # cpt = 0
    # res = np.linalg.norm(np.diag(S,1))/np.linalg.norm(np.diag(S))
    # for i in range(30):
    #     (Q1,R1) = np.linalg.qr(S.T)
    #     (Q2,R2) = np.linalg.qr(R1.T)
    #     S=R2
    #     U=np.dot(U,Q2)
    #     V=np.dot(Q1.T,V)
    
    # for i in range(15,n):
    #     S[i,i] = 0
    # SD = np.diagonal(S)
    # SDABS = np.abs(SD)
    # RSD = np.array([])
    # RU = np.zeros(np.shape(U))
    # RV = np.zeros(np.shape(V))
    
    # for i in range(len(SD)):
    #     maxi = np.argmax(SDABS)
    #     # print("Maxi index : " + str(maxi) + " = " + str(SD[maxi]))
    #     RSD = np.append(RSD,SD[maxi])
    #     SDABS = np.delete(SDABS,maxi)
    #     RU[i,:] = U[maxi,:]
    #     RV[:,i] = V[:,maxi]

    # SD = SD[::-1]
    # print("SD VALUESS : ")
    # print(RSD[:15])
    # print("RU & RV shape : " + str(np.shape(RU)) + " " + str(np.shape(RV))) 
    # print("U & V shape : " + str(np.shape(U)) + " " + str(np.shape(V))) 
    # print(" NORM DE OUF 2: " + str(np.linalg.norm(A - RU.dot(RSD.dot(RV)))))
    # res = np.dot(np.dot(Ql, np.dot(U, np.dot(S, V))),Qr)
    #print(f"Return Norm : {np.linalg.norm(res)}")
    return (U, S, V)



# Compresse l'image représenté par la matrice A par un coefficient k
def compress(image, k): 
    A=np.copy(image)
    # Récupération de la matrice bidiagonale
    n, m = A.shape[0],A.shape[1]
    print("Shape de A : " + str(n)+ " " + str(m))
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

    # Ql,BD,Qr = bidiagg(G)
    # print(f"NORME FIN : {np.linalg.norm(G-Ql.dot(BD.dot(Qr)))}")

    (U_R, S_R, V_R) = factorisation_SVD(R)
    (U_G, S_G, V_G) = factorisation_SVD(G)
    #(U_B, S_B, V_B) = factorisation_SVD(B)
    
    # (U_R, S_R, V_R) = np.linalg.svd(R)
    # (U_G, S_G, V_G) = np.linalg.svd(G)
    (U_B, S_B, V_B) = np.linalg.svd(B)

    print("//////// " + str(np.linalg.norm(U_R)) + " " + str(np.linalg.norm(S_R)) + " " + str(np.linalg.norm(V_R)))
    print(" " + str(np.shape(U_R)) + " " + str(np.shape(S_R)) + " " + str(np.shape(V_R)))
    print(" " + str(np.shape(U_R[:,:k])) + " " + str(np.shape(np.diag(S_R[:k]))) + " " + str(np.shape(V_R[:k,:])))

    # Récuperation des matrices RGB
    # R = np.dot(np.dot(U_R[:,:k], np.diag(S_R[:k])), V_R[:k,:])
    # G = np.dot(np.dot(U_G[:,:k], np.diag(S_G[:k])), V_G[:k,:])
    B = np.dot(np.dot(U_B[:,:k], np.diag(S_B[:k])), V_B[:k,:])
    R = S_R
    G = S_G
    print("//////// " + str(np.shape(R)))
    print("//////// " + str(np.linalg.norm(R)) + " " + str(np.linalg.norm(G)) + " " + str(np.linalg.norm(B)))
    # Insertion des valeurs précédemment calculés
    for i in range(n):
        for j in range(m):
            A[i][j][0] = R[i][j]
            A[i][j][1] = G[i][j]
            A[i][j][2] = B[i][j]

    return A



# =========TESTS===========

# Calcule compression de l'image
image = mpimg.imread("essai2.png")

k = 30
compressed_image = compress(image, k)
# print(compressed_image)

# Affiche résultats
plt.imshow(image)
plt.savefig("original") 

plt.imshow(compressed_image)
plt.savefig('compress')

# ===================== 
