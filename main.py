from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import os # accessing directory structure
import plotly.express as px
import seaborn as sns
import pandas as pd



df = pd.read_excel(r'Cluster_data.xlsx')
df.dataframe = 'Cluster_data.xlsx'
print(df.shape)
print(df.head(5))

print(df.isnull().sum())

df1 = df.dropna(how='any',axis=0)

print(df1.shape)

print(df1.isnull().sum())

print(df1.head())


ss = StandardScaler()
ss.fit_transform(df1.iloc[:, [14,33]].values)

X= df1.iloc[:, [14,33]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss,)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


kmeansmodel = KMeans(n_clusters=4 , init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)
labels = kmeansmodel.labels_
print(labels)
df1["label"] = labels
df1.loc[df1['label'] == 3, 'Category'] = 'Cluster_4'
df1.loc[df1['label'] == 2, 'Category'] = 'Cluster_3'
df1.loc[df1['label'] == 1, 'Category'] = 'Cluster_2'
df1.loc[df1['label'] == 0, 'Category'] = 'Cluster_1'

print(df1)

file_name = 'Clustered_data.xlsx'

# saving the excel
df1.to_excel(file_name)

print("clustered data saved to excel")