import pandas as pd
from MDP_function2 import *
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')
from MDP_function2 import *

def discritize_all():
	original_data = pd.read_csv('data/MDP_Original_data2.csv')
	#print original_data.iloc[:,[8]].head()
	vec = list(original_data.columns)

	#original_data[vec[7]].plot()
	#plt.show()

	temp = original_data.copy()
	for i in range(6,len(temp.columns)):
		if temp[vec[i]].dtype != 'int64':
			x = pd.cut(temp[vec[i]],10,labels=False)
			temp[vec[i]] = x

	temp.to_csv('discreitized_float_data.csv')

def discretizePCABasedOnECR():
	original_data = pd.read_csv('data/MDP_Original_data2.csv')
	pca_data = pd.read_csv('data/pca_data.csv')

	input_data = pd.DataFrame()

	input_data = original_data.iloc[:,0:6]
	two_features = original_data[['Level', 'cumul_Interaction']]

	input_data = pd.concat([input_data, two_features], axis=1)

	features = ['Level', 'cumul_Interaction', '5']

	# Best ECR for PCA feature 5 with 10 bins. Adding it to the list and then checking feature 0 and 2
	x = pd.cut(pca_data['5'], 10, labels=False)
	input_data = pd.concat([input_data, x], axis=1)

	cols = list(pca_data.columns)

	perfect_bins = {}

	cols = ['1', '3','4']
	for col in cols:
		bins_list = []
		ecr_list = []
		max_ecr = 0.0
		max_i = 4
		for i in range(2, 46, 2):
			x = pd.cut(pca_data[col], i, labels=False)
			t = pd.concat([input_data, x], axis=1)
			selected_features = features+[col]
			ecr = induce_policy_MDP2(t, selected_features)
			bins_list.append(i)
			ecr_list.append(ecr)
			if (ecr > max_ecr):
				max_ecr = ecr
				max_i = i
		plt.plot(bins_list, ecr_list)
		plt.savefig('images/pca5+[0,2]/col'+str(col)+'.png')
		print 'Max ECR for col: ' + str(col) + ' is ' + str(max_ecr) + ' for ' + str(max_i) + ' bins'
		perfect_bins[col] = max_i

	print perfect_bins


def discritize_pca_components():
	pca_features = pd.read_csv('data/pca_features_top3.csv')
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

if __name__ == "__main__":
    discretizePCABasedOnECR()