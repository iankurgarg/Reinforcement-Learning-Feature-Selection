import pandas as pd
from MDP_function2 import *
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
import numpy as np
from matplotlib import style
style.use('ggplot')
from MDP_function2 import *

def discritize_fixed_bins(bins=5):
	original_data = pd.read_csv('data/uncorrelated_continuous_data.csv')
	#print original_data.iloc[:,[8]].head()
	vec = list(original_data.columns)

	temp = pd.DataFrame()
	for i in range(0,len(vec)):
		if original_data[vec[i]].dtype != 'int64':
			x = pd.cut(original_data[vec[i]],bins,labels=False)
			temp[vec[i]] = x

	temp.to_csv('data/discritize_fixed_bins.csv', index=False)

def discretizeGreedyApproach(nn_data):
	original_data = pd.read_csv('data/MDP_Original_data2.csv')

	input_data = pd.DataFrame()

	input_data = original_data.iloc[:,0:6]
	two_features = original_data[['Level', 'cumul_Interaction']]
	input_data = pd.concat([input_data, two_features], axis=1)
	
	cols = list(nn_data.columns)
	features = ['Level', 'cumul_Interaction']
	perfect_bins = {}

	max_ecr2 = 0.0
	max_col = ''
	max_bins = 0

	cols = ['Dim.6']
	for col in cols:
		bins_list = []
		ecr_list = []
		max_ecr = 0.0
		max_i = 4
		for i in range(2, 6, 2):
			x = pd.cut(nn_data[col], i, labels=False)
			t = pd.concat([input_data, x], axis=1)
			selected_features = features+[col]
			ecr = induce_policy_MDP2(t, selected_features)
			bins_list.append(i)
			ecr_list.append(ecr)
			if (ecr > max_ecr):
				max_ecr = ecr
				max_i = i
			if (max_ecr > max_ecr2):
				max_ecr2 = max_ecr
				max_col = col
				max_bins = i
		print 'Max ECR for col: ' + str(col) + ' is ' + str(max_ecr) + ' for ' + str(max_i) + ' bins'
		perfect_bins[col] = max_i
	
	x = pd.cut(nn_data[max_col], max_bins, labels=False)
	print max_bins
	print max_col
	final_data = pd.concat([input_data, x], axis=1)
	final_data.to_csv('data/Training_data_generated.csv')


def discritize_pca_components():
	pca_features = pd.read_csv('data/uncorrelated_continuous_data.csv')
	nbins_list = []

	t = pd.DataFrame()
	# processing component 1:
	for i in range(50,105,5):
		x = pd.cut(pca_features['0'],i,labels=False)
		t = pd.concat([t,pd.DataFrame(x)], axis=1)
		## curr_ecr = getEcrWith this feature

	# processing component 2
	for i in range(9,30,3):
		x = pd.cut(pca_features['1'],i,labels=False)
		t = pd.concat([t,pd.DataFrame(x)], axis=1)

	# processing component 3
	for i in range(50,200,10):
		x = pd.cut(pca_features['2'],i,labels=False)
		t = pd.concat([t,pd.DataFrame(x)], axis=1)

	# 0 to 10 columns are for pc1
	# 11 to 18 are for pc2
	# 19 to 32 are for pc3
	t.to_csv('data/trial_pca_discretization.csv')

# Try clustering on PCA's to find out how many bins to make
def discretizeByClustering(data):
	data.head()
	ms = MeanShift(n_jobs=-1)
	ms.fit(np.array(data['0']).reshape(-1,1))
	np.unique(ms.labels_)

if __name__ == "__main__":
    famd_data = pd.read_csv('data/FAMD_features.csv')
    discretizeGreedyApproach(famd_data)