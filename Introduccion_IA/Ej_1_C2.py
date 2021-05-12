import numpy as np


centroide_1 = np.array([[1,0,0,0]])
centroide_2 = np.array([[0,1,0,0]])

k=2     #constante que separa los centroides

centroide_1 = centroide_1*k

centroide_2 = centroide_2

n = 200

cluster1 = np.repeat(centroide_1, (n/2),axis=0).transpose()
cluster2 = np.repeat(centroide_2, (n/2),axis=0).transpose()

stddev = 5

noise1 = np.random.normal(0,stddev,cluster1.shape)
noise2 = np.random.normal(0,stddev,cluster2.shape)

cluster1 = cluster1 + noise1
cluster2 = cluster2 + noise2

cluster_idx = np.random.randint(0,2,size=n)