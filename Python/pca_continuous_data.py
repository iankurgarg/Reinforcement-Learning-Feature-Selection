import pandas as pd
from sklearn.decomposition import PCA

pca = PCA()
original_data = pd.read_csv('data/continuous_data.csv')
temp = pca.fit_transform(original_data)
temp = pd.DataFrame(temp)