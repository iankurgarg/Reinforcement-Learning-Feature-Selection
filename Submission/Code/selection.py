import pandas as pd

temp = pd.read_csv('data/MDP_Original_data2.csv')
vec = list(temp.columns)
categorical_features_details = []
categorical_features = []
continuous_features = vec[6:]

# separating out categorical features
for v in vec:
	if (temp[v].dtype == 'int64') and (max(temp[v]) < 10) :
		categorical_features_details.append((v,max(temp[v])))
		categorical_features.append(v)

for c in categorical_features:
	continuous_features.remove(c)

# Writing the features to different files
temp[categorical_features].to_csv('data/categorical_data.csv', index = False)
temp[continuous_features].to_csv('data/continuous_data.csv', index=False)


# Removing correlated attributes from the continuous data
import numpy as np

continuous_data = pd.read_csv('data/continuous_data.csv')
continuous_data.corr().to_csv('data/correlation_continuous_data.csv')

#np.corrcoef(continuous_data)
# plot the correlation matrix
from matplotlib import pyplot as plt
from matplotlib import cm as cm
def plotCorrMatrix(data,title='Continuous Feature Correlation Figure 1'):
	fig = plt.figure(figsize = (50,50) )
	ax1 = fig.add_subplot(111)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(data.corr(), interpolation="nearest", cmap=cmap)
	ax1.grid(True)
	plt.title(title)
	labels=list(data.columns)
	ax1.set_xticklabels(labels,fontsize=6)
	ax1.set_yticklabels(labels,fontsize=6)
	# Add colorbar, make sure to specify tick locations to match desired ticklabels
	fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
	plt.show()

# Removing correlated variables
# Stage 1 Rules for removing them: 1. It should be highly correlated to at least 3 variables
corr_matrix = pd.read_csv('data/correlation_continuous_data.csv')
vec = list(corr_matrix.iloc[:,0])
correlated_features = set()
cutoff_limit = 0.8
cutoff_count = 3
for i in range(1,len(corr_matrix.columns)):
	for j in range(i+2,len(corr_matrix.columns)):
		if corr_matrix.iloc[i,j] >= cutoff_limit or corr_matrix.iloc[i,j] <= -cutoff_limit:
			correlated_features.add(corr_matrix.iloc[i,0])
			break

# >>> correlated_features
# set(['RightApp', 'cumul_UseCount', 'CurrPro_avgProbTimeWE', 'cumul_RightApp', 'MorphCount', 'cumul_WrongApp', 'DirectProofActionCount', 'cumul_Interaction', 'cumul_TotalPSTime', 'cumul_DirectProofActionCount', 'cumul_FDActionCount', 'cumul_AppRatio', 'cumul_WrongSemanticsApp', 'WrongApp', 'WrongSemanticsApp', 'AppRatio', 'BlankRatio', 'cumul_MorphCount', 'cumul_OptionalCount', 'RrightAppRatio', 'cumul_TotalTime', 'FDActionCount', 'CurrPro_avgProbTime', 'TotalTime', 'cumul_actionCount', 'cumul_WrongSyntaxApp', 'cumul_PrepCount', 'CurrPro_avgProbTimePS', 'UseCount', 'actionCount'])
# >>> len(correlated_features)
# 30
vec = list(continuous_data.columns)
for v in correlated_features:
	vec.remove(v)
uncorrelated_continuous_data = continuous_data[vec]
uncorrelated_continuous_data.to_csv('data/uncorrelated_continuous_data.csv')
plotCorrMatrix(uncorrelated_continuous_data, 'Matrix after removing highly correlated features')


# PCA on uncorrelated continuous data

from sklearn.decomposition import PCA

pca = PCA()
original_data = pd.read_csv('data/uncorrelated_continuous_data.csv')
temp = pca.fit_transform(original_data)
temp = pd.DataFrame(temp)
#choose top 6 principle components as they capture 98.09% of the variance
temp[[0,1,2,3,4,5]].to_csv('data/pca_data.csv',index=False)

