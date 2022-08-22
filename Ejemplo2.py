import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('download.jpg', 0)
u, s, vh=np.linalg.svd(img, full_matrices=True)

nCoef=1
print("Cantidad de observaciones")
print(len(u[0]))

while (nCoef < 20):  #len(u[0])-1
    nCoef=int(nCoef)
    s_mat=np.zeros((nCoef, nCoef))
    d=s[:nCoef]
    s_mat[:nCoef, :nCoef]=np.diag(d)

    u_sampled=u[:,:nCoef]
    vh_sampled=vh[:nCoef,:]

    A = np.matmul(u_sampled, np.matmul(s_mat, vh_sampled));

    cv2.imshow('reconstruccion', A/np.max(A))
    cv2.waitKey(500)

    nCoef=nCoef+1
    print(nCoef)

cv2.destroyAllWindows()

plt.plot(np.cumsum(s)/np.sum(s))
plt.xlabel('Coeficientes')
plt.ylabel('% Energy')
plt.show()

plt.semilogy(s)
plt.xlabel('Coeficientes')
plt.ylabel('value')
plt.show()