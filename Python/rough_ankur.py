import collections
import numpy as np
import pandas as pd
import mdptoolbox, mdptoolbox.example
import argparse
from MDP_function2 import *
from scipy.stats import chisquare
from sklearn.cluster import MeanShift
import matplotlib.pylab as plt




# categorical_data = pd.read_csv('data/categorical_data.csv')

# cols = list(categorical_data.columns)
# N = len(cols)

# chisquareValues = np.zeros((N, N))

# for i in range(N):
# 	for j in range(N):
# 		if (i != j):
# 			chisquareValues[i,j] = chisquare(categorical_data[cols[i]], categorical_data[cols[j]])[1]



# input = pd.read_csv('data/pca_data.csv')
# cols = list(input.columns)

# #for c in cols:

# b = []
# clus = []


# for bw in range(640, 720, 2):
# 	ms = MeanShift(bandwidth=bw, n_jobs=-1)
# 	ms.fit(np.array(input['0']).reshape(-1,1))
# 	n = len(np.unique(ms.labels_))
# 	b.append(bw)
# 	clus.append(n)


# b2 = [640, 642, 644, 646, 648, 650, 652, 654, 656, 658, 660, 662, 664, 666, 668, 670, 672, 674, 676, 678, 680, 682, 684, 686, 688, 690, 692, 694, 696, 698, 700, 702, 704, 706, 708, 710, 712, 714, 716, 718]
# clus2 = [5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3]

# plt.plot(b2, clus2)
# plt.show()



# input = pd.read_csv('data/FAMD_features.csv')
# cols = list(input.columns)
# output = pd.DataFrame()
# for col in cols:
# 	ms = MeanShift(n_jobs=-1)
# 	ms.fit(np.array(input[col]).reshape(-1,1))
# 	temp = pd.DataFrame(ms.labels_)
# 	output = pd.concat([output, temp], axis=1)


# ms = MeanShift(n_jobs=-1)
# ms.fit(np.array(input))
# temp = pd.DataFrame(ms.labels_)
# output = pd.concat([output, temp], axis=1)

# output.columns = ['0', '1', '2', '3', '4', '5', '6']
# output.to_csv('data/pca_discretized_natural_bins.csv', index=False)




original_data = pd.read_csv('data/MDP_Original_data2.csv')
input_data = original_data.iloc[:,0:6]
two_features = original_data['Level']

input_data = pd.concat([input_data, two_features], axis=1)
features = ['Level']


bins_list = []
ecr_list = []
max_ecr = 0.0
max_i = 4
col = 'cumul_Interaction'
for i in range(827, 1000, 2):
	x = pd.cut(original_data[col], i, labels=False)
	t = pd.concat([input_data, x], axis=1)
	selected_features = features+[col]
	ecr = induce_policy_MDP2(t, selected_features)
	bins_list.append(i)
	ecr_list.append(ecr)
	if (ecr > max_ecr):
		max_ecr = ecr
		max_i = i
#plt.plot(bins_list, ecr_list)
#plt.savefig('images/nn_bins/col'+str(col)+'.png')
print 'Max ECR for col: cumul_Interaction is ' + str(max_ecr) + ' for ' + str(max_i) + ' bins'
#perfect_bins[col] = max_i

