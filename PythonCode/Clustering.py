
from sklearn import datasets
iris = datasets.load_iris()
import numpy as np
iris_df = pd.DataFrame(data= np.c_[iris["data"],iris["target"]],columns = iris["feature_names"] +["Target"] )

type(iris_df)
iris_df.shape
iris_df.columns
iris_df.head(5)

iris_df["Target"].unique()

iris["target_names"]

from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X = std.fit_transform(iris_df.values[:,:-1])

from sklearn.cluster import KMeans

 kmeans = KMeans(n_clusters=3)
 
 kmeans.fit(X)
 
 kmeans.labels_
 kmeans.predict(X)
 
 
 
 