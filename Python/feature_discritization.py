import pandas as pd
from MDP_function2 import *
import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')
from MDP_function2 import *

def discritize_all():
	original_data = pd.read_csv('data/nn/nn_scrap/nn_layer2_output.csv')
	#print original_data.iloc[:,[8]].head()
	vec = list(original_data.columns)

	#original_data[vec[7]].plot()
	#plt.show()

	temp = pd.DataFrame()
	for i in range(0,len(vec)):
		if original_data[vec[i]].dtype != 'int64':
			x = pd.cut(original_data[vec[i]],5,labels=False)
			temp[vec[i]] = x

	temp.to_csv('data/nn/nn_scrap/nn_discreitized_data.csv', index=False)

def discretizeBasedOnECR():
	original_data = pd.read_csv('data/MDP_Original_data2.csv')

	famd_data = pd.read_csv('data/famd/FAMD_features.csv')
	pca_data = pd.read_csv('data/pca/pca_data.csv')
	nn_data = pd.read_csv('data/nn/nn_scrap/nn_layer2_output.csv')

	input_data = pd.DataFrame()

	input_data = original_data.iloc[:,0:6]
	two_features = original_data[['Level', 'cumul_Interaction']]

	# Best ECR for PCA feature 5 with 10 bins. Adding it to the list and then checking feature 0 and 2
	x = pd.cut(famd_data['Dim.6'], 2, labels=False)
	input_data = pd.concat([input_data, two_features], axis=1)
	
	cols = list(nn_data.columns)
	features = ['Level', 'cumul_Interaction']
	perfect_bins = {}

	#cols = ['Dim.5', 'Dim.6','Dim.7', 'Dim.8']
	for col in cols:
		bins_list = []
		ecr_list = []
		max_ecr = 0.0
		max_i = 4
		for i in range(2, 20, 2):
			x = pd.cut(nn_data[col], i, labels=False)
			t = pd.concat([input_data, x], axis=1)
			selected_features = features+[col]
			ecr = induce_policy_MDP2(t, selected_features)
			bins_list.append(i)
			ecr_list.append(ecr)
			if (ecr > max_ecr):
				max_ecr = ecr
				max_i = i
		plt.plot(bins_list, ecr_list)
		plt.savefig('images/nn_bins/col'+str(col)+'.png')
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
    discretizeBasedOnECR()