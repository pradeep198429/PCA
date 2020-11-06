import pandas as pd
import numpy as np

winedata=pd.read_csv("wine.csv")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

Normalised_winedata= scale(winedata)
pca = PCA(n_components = 6)
pca_values= pca.fit_transform(Normalised_winedata)
print(pca_values)

# The amount of variance that each PCA explains is
var = pca.explained_variance_ratio_
pca.components_[0]

# Cumulative variance

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
print(var1)
# Variance plot for PCA components obtained
plt.plot(var1,color="red")
plt.show()


# to apply Kmeans clustering

df_new = pd.DataFrame(pca_values[:,0:4])

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(2, 15)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(df_new)
    kmeanModel.fit(df_new)

    distortions.append(sum(np.min(cdist(df_new, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / df_new.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(df_new, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / df_new.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key,val in mapping1.items():
    print(str(key)+' : '+str(val))


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()



for key,val in mapping2.items():
    print(str(key)+' : '+str(val))



plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


#Hclsuster

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


z=linkage(df_new,method='complete',metric='euclidean')

plt.figure(figsize=(15,5));
plt.title("Hierarchical Culster Dendogram");
plt.xlabel("Index");
plt.ylabel('Distance');
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=8);
plt.show()

help(AgglomerativeClustering)

h_clust=AgglomerativeClustering(n_clusters=10,linkage='complete', affinity='euclidean').fit(df_new)
h_clust.labels_
winedata["clust"]=h_clust.labels_
df=winedata.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

