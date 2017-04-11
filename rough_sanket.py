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
temp[vec[0:6]].to_csv('data/default_data.csv', index=False)

# print categorical_features_details
# print categorical_features
# print continuous_features

### OUTPUT OF THE SCRIPT
# Range between 10 to 20
# ['UseWindowInfo', 'SystemInfoHintCount', 'PreviousStepClickCountWE', 'cumul_SystemInfoHintCount']

# Range between 0-10
# ['symbolicRepresentationCount', 'Level', 'probDiff', 'difficultProblemCountSolved', 'difficultProblemCountWE', 'easyProblemCountSolved', 'easyProblemCountWE', 'probAlternate', 'easyProbAlternate', 'RuleTypesCount', 'NewLevel', 'SolvedPSInLevel', 'SeenWEinLevel', 'probIndexinLevel', 'probIndexPSinLevel', 'cumul_difficultProblemCountSolved', 'cumul_difficultProblemCountWE', 'cumul_easyProblemCountSolved', 'cumul_easyProblemCountWE', 'cumul_probAlternate', 'cumul_easyProbAlternate', 'cumul_RuleTypesCount', 'cumul_probIndexinLevel', 'CurrPro_NumProbRule']

#*********************************************************
# processing continuous features checking correlated variables.

import pandas as pd
import numpy as np

continuous_data = pd.read_csv('data/continuous_data.csv')
#continuous_data.corr().to_csv('data/correlation_continuous_data.csv')

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

# choose top principle components
# >>> pca.explained_variance_ratio_
# array([  7.28355759e-01,   1.36513237e-01,   7.00681792e-02,
#          2.24008293e-02,   1.36150173e-02,   1.01658098e-02,
#          8.51554398e-03,   2.97274733e-03,   2.48436776e-03,
#          1.35684401e-03,   1.00168968e-03,   8.62959445e-04,
#          6.45274791e-04,   2.71804320e-04,   1.93545898e-04,
#          1.00928853e-04,   8.01383261e-05,   6.41223789e-05,
#          5.61715459e-05,   4.63120553e-05,   4.07903092e-05,
#          2.86934870e-05,   2.39776430e-05,   2.27703187e-05,
#          1.82581720e-05,   1.64110265e-05,   1.57070157e-05,
#          1.05048042e-05,   8.38789485e-06,   6.19827540e-06,
#          5.99996174e-06,   4.99218374e-06,   4.04541487e-06,
#          3.97654081e-06,   3.58440295e-06,   3.40945252e-06,
#          2.28647680e-06,   2.23156009e-06,   1.44794111e-06,
#          1.00013525e-06,   6.95350337e-07,   5.77928360e-07,
#          5.34495107e-07,   4.85241409e-07,   3.68088086e-07,
#          2.57587873e-07,   1.53666907e-07,   1.25434023e-07,
#          1.10053985e-07,   1.07315718e-07,   1.00947956e-07,
#          9.63018423e-08,   8.82122159e-08,   8.10120914e-08,
#          6.31823480e-08,   4.96288327e-08,   4.57827687e-08,
#          3.18633791e-08,   1.70812503e-08,   1.57049386e-08,
#          1.42198389e-08,   8.67150082e-09,   5.14313629e-09,
#          4.62111225e-09,   3.28222210e-09,   1.87835063e-09,
#          9.95415882e-10,   6.26398816e-10,   1.97026179e-10,
#          1.24571813e-10])

# Try clustering on PCA's to find out how many bins to make
from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd
pca_data = pd.read_csv('data/new_pca_data.csv')
pca_data.head()
ms = MeanShift(n_jobs=-1)
ms.fit(np.array(pca_data['0']).reshape(-1,1))
np.unique(ms.labels_)








# Using variance inflation factor for removing features
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X):
	thresh = 5.0
	variables = range(X.shape[1])
	dropped=True
	while (dropped==True):
		dropped=False
		vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
		maxloc = vif.index(max(vif))
		if max(vif) > thresh:
			print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
			del variables[maxloc]
			dropped=True
	print('Remaining variables:')
	print(X.columns[variables])
	return X[variables]

clean_data = calculate_vif_(pd.read_csv('data/continuous_data.csv'))







# *****************************************************
# Scaling and trying pca
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

temp = pd.read_csv('data/uncorrelated_continuous_data.csv')
vec = list(temp.columns)
for v in vec:
	temp[v] = preprocessing.scale(temp[v])

pca = PCA()
temp_pca = pca.fit_transform(temp)
temp_pca = pd.DataFrame(temp_pca)

original_pca = pd.read_csv('data/pca_data.csv')

exp_variance = list(pca.explained_variance_ratio_)
# this method has the result
#>>> pca.explained_variance_ratio_
# array([  1.39458344e-01,   8.39514305e-02,   6.02347227e-02,
#          4.81988162e-02,   4.50864646e-02,   4.25902448e-02,
#          3.45768707e-02,   2.89915597e-02,   2.78930736e-02,
#          2.29913987e-02,   2.06589697e-02,   2.00705059e-02,
#          1.96919986e-02,   1.84801665e-02,   1.73147673e-02,
#          1.70365470e-02,   1.61552689e-02,   1.53394033e-02,
#          1.46319386e-02,   1.41629299e-02,   1.39003986e-02,
#          1.34651075e-02,   1.28739435e-02,   1.21999702e-02,
#          1.20389418e-02,   1.11338309e-02,   1.07671967e-02,
#          1.02045663e-02,   9.68397813e-03,   9.43327353e-03,
#          9.08713157e-03,   8.58203678e-03,   8.42318665e-03,
#          8.18170110e-03,   8.00373216e-03,   7.69822602e-03,
#          7.63279563e-03,   7.29912331e-03,   6.89340834e-03,
#          6.69221610e-03,   6.25081012e-03,   6.11137477e-03,
#          5.96352336e-03,   5.60343244e-03,   5.53101607e-03,
#          5.18297758e-03,   5.11380522e-03,   4.91954091e-03,
#          4.52901679e-03,   4.29227543e-03,   4.04294017e-03,
#          3.90545881e-03,   3.62389214e-03,   3.42820942e-03,
#          3.40916989e-03,   3.25337744e-03,   2.90756117e-03,
#          2.88185421e-03,   2.65656573e-03,   2.63735610e-03,
#          2.39198649e-03,   1.95843698e-03,   1.70261948e-03,
#          1.48899166e-03,   1.32654191e-03,   1.19682026e-03,
#          1.03399514e-03,   5.22008215e-04,   4.23509458e-04,
#          7.46923519e-07])
# Almost 60 principle components are required to explain 98% variance which is not helpful

# Will not try normalizing the data

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

temp = pd.read_csv('data/uncorrelated_continuous_data.csv')
vec = list(temp.columns)
preprocessing.normalize(temp,axis=1,copy=False)

pca = PCA()
temp_pca = pca.fit_transform(temp)
temp_pca = pd.DataFrame(temp_pca)

original_pca = pd.read_csv('data/pca_data.csv')

exp_variance = list(pca.explained_variance_ratio_)


# ********************************************************************
# Scaling the 124 features for using in neural network
import pandas as pd
from sklearn import preprocessing
temp = pd.read_csv('data/MDP_Original_data2.csv')
vec = temp.columns
temp = temp[vec[6:]]
temp = preprocessing.scale(temp)
temp.head()


# ********************************************************************
# Comparing results from both neural network training
import pandas as pd
import numpy as np
import tensorflow as tf
original = pd.read_csv('data/scaled124_generated.csv')
output_1 = pd.read_csv('data/nn_final_layer_output.csv')
output_2 = pd.read_csv('data/nn_scrap/nn_final_layer_output.csv')

original = np.array(original)
output_1 = np.array(output_1)
output_2 = np.array(output_2)

error1 = tf.reduce_sum(tf.abs(original - output_1))
error2 = tf.reduce_sum(tf.abs(original - output_2))
session = tf.Session()
session.run(tf.global_variables_initializer())
e1 = session.run(error1)
e2 = session.run(error2)

print 'error 1:',e1
print 'error 2:',e1
