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
