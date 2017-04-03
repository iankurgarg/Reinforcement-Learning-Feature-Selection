import pandas as pd
from MDP_function2 import *
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')

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
    discritize_pca_components()