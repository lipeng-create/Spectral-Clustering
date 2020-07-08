import numpy as np
import matplotlib.pyplot as plt
from sklearn import  datasets
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

n_samples = 600
# 第一个图
X1,y1 = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
clustering = KMeans(n_clusters=2, random_state=0).fit(X1)
z1=clustering.labels_

nPoints = len(X1)
scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange']
for i in range(7):
    color = scatterColors[i % len(scatterColors)]
    w1 = [];  v1 = []
    for j in range(nPoints):
        if z1[j] == i:
            w1.append(X1[j, 0])
            v1.append(X1[j, 1])
    plt.figure(num=1,figsize=(8,7))
    plt.subplot(2,2,1)
    plt.scatter(w1, v1, c=color, alpha=1, marker='.')
    plt.title('k-means')


# 第二个图
X2,y2 = datasets.make_moons(n_samples=n_samples, noise=.05)  
clustering = KMeans(n_clusters=2, random_state=0).fit(X2)
z2=clustering.labels_

nPoints = len(X2)
scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange']
for i in range(7):
    color = scatterColors[i % len(scatterColors)]
    w1 = [];  v1 = []
    for j in range(nPoints):
        if z2[j] == i:
            w1.append(X2[j, 0])
            v1.append(X2[j, 1])
    plt.figure(num=1)
    plt.subplot(2,2,2)
    plt.scatter(w1, v1, c=color, alpha=1, marker='.')
    plt.title('k-means')


# 第三个图
X3,y3 = X1,y1
# X3,y3 = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", eigen_solver='arpack',
                                affinity="nearest_neighbors").fit(X3)
z3=clustering.labels_

nPoints = len(X3)
scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange']
for i in range(7):
    color = scatterColors[i % len(scatterColors)]
    w1 = [];  v1 = []
    for j in range(nPoints):
        if z3[j] == i:
            w1.append(X3[j, 0])
            v1.append(X3[j, 1])
    plt.figure(num=1)
    plt.subplot(2,2,3)
    plt.scatter(w1, v1, c=color, alpha=1, marker='.')
    plt.title('Spectral Clustering')


# 第四个图
X4,y4 = X2,y2
# X4,y4 = datasets.make_moons(n_samples=n_samples, noise=.05)       
clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", eigen_solver='arpack',
                                affinity="nearest_neighbors").fit(X4)
z4=clustering.labels_

nPoints = len(X4)
scatterColors = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange']
for i in range(7):
    color = scatterColors[i % len(scatterColors)]
    w1 = [];  v1 = []
    for j in range(nPoints):
        if z4[j] == i:
            w1.append(X4[j, 0])
            v1.append(X4[j, 1])
    plt.figure(num=1)
    plt.subplot(2,2,4)
    plt.scatter(w1, v1, c=color, marker='.')
    plt.title('Spectral Clustering')

plt.show() 