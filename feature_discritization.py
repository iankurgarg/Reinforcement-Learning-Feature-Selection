import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')

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

dic = {}
for i in x:
	if i not in dic.keys():
		dic[i] = 1
	else:
		dic[i] = dic[i]+1

print dic