import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
from MDP_function2 import *

def checkFeatures(original_data):
    cols = list(original_data.columns)
    #print cols
    max_ecr = 0.0
    max_selected_features = []
    selected_features = ['cumul_Interaction','Level']
    for i in range(6, len(cols)):
        #selected_features = max, cols[i]]
        print cols[i]
        ecr = induce_policy_MDP2(original_data, selected_features+[cols[i]])
        if (ecr > max_ecr):
            max_ecr = ecr
            max_selected_features = [cols[i]]
    print 'best feature 3:', max_selected_features
    print max_ecr
    selected_features = selected_features + max_selected_features

    max_selected_features = []
    for i in range(6, len(cols)):
        #print cols[i],max_selected_features[0]
        if (cols[i] not in selected_features):
            #selected_features = [max_selected_features,cols[i]]
            ecr = induce_policy_MDP2(original_data, selected_features+[cols[i]])
            if (ecr > max_ecr):
                max_ecr = ecr
                max_selected_features = [cols[i]]
    print 'best feature 4:', max_selected_features
    selected_features = selected_features+max_selected_features
    print max_ecr
    print selected_features
    return max_selected_features


if __name__ == "__main__":
    #original_data = pandas.read_csv('data/MDP_Original_data2.csv')
    #original_data = original_data.iloc[:,0:6]
    #pca_discretized = pandas.read_csv('data/pca_discretized.csv')
    #selected_features = ['Level', 'probDiff']
    #selected_features = ['symbolicRepresentationCount']
    temp = pandas.read_csv('data/trial_data.csv')
    ECR_value = induce_policy_MDP2(temp, ['cumul_Interaction','Level'])
    print ECR_value
    #print checkFeatures(original_data)
    #temp = pandas.concat([original_data,pca_discretized],axis=1)
     #print checkFeatures(temp)

#75.8031768352
#75.8014789316