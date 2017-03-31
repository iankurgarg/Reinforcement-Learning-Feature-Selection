import pandas as pd

original_data = pd.read_csv('data/MDP_Original_data2.csv')
continuous_features = []
column_names = list(original_data.columns)

for column in column_names:
	if original_data[column].dtype == 'float64':
		continuous_features.append(column)

temp = original_data[continuous_features]

temp.to_csv('data/continuous_features_only.csv')
