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
    original_features = ['cumul_Interaction', 'Level', 'probDiff']
    original_features = []
    max_selected_features = []
    for i in range(6, len(cols)):
        if (original_data[cols[i]].dtype == 'int64' and cols[i] != 'cumul_Interaction' and cols[i] != 'Level' and cols[i] != 'probDiff'):
            selected_features = original_features+[cols[i]]
            ecr = induce_policy_MDP2(original_data, selected_features)
            if (ecr > max_ecr):
                max_ecr = ecr
                max_selected_features = selected_features
    print 'best feature 1:', max_selected_features
    print max_ecr

    # for i in range(6, len(cols)):
    #     print cols[i],max_selected_features[0]
    #     if (cols[i] != max_selected_features[0]):
    #         selected_features = [max_selected_features[0],cols[i]]
    #         ecr = induce_policy_MDP2(original_data, selected_features)
    #         if (ecr > max_ecr):
    #             max_ecr = ecr
    #             max_selected_features = selected_features
    # print 'best feature 2:', max_selected_features
    # print max_ecr
    return max_selected_features

def SelectBestPCAFeatureAfterDiscretization(original_data):
    discretePCA = pandas.read_csv('data/trial_pca_discretization.csv')
    cols = list(discretePCA.columns)

    new_data = pandas.concat([original_data, discretePCA], axis=1)

    max_ecr = 0.0
    ecr_list = []
    max_features = []
    max_selected_features = []

    for i in range(0, 11):
        selected_features = [cols[i]]
        ecr = induce_policy_MDP2(new_data, selected_features)
        ecr_list.append(ecr)
        if (ecr > max_ecr):
            max_ecr = ecr
            max_selected_features = [cols[i]]
    print 'best feature 3:', max_selected_features
    print max_ecr
    selected_features = selected_features + max_selected_features

    max_features.append(max_selected_features)

    ecr_list2 = []
    max_ecr = 0.0
    for i in range(11, 19):
        selected_features = [cols[i]]
        ecr = induce_policy_MDP2(new_data, selected_features)
        ecr_list2.append(ecr)
        if (ecr > max_ecr):
            max_ecr = ecr
            max_selected_features = selected_features
    print 'best feature 2:', max_selected_features
    print max_ecr

    max_features.append(max_selected_features)

    ecr_list3 = []
    max_ecr = 0.0
    for i in range(19, 32):
        selected_features = [cols[i]]
        ecr = induce_policy_MDP2(new_data, selected_features)
        ecr_list3.append(ecr)
        if (ecr > max_ecr):
            max_ecr = ecr
            max_selected_features = selected_features
    print 'best feature 3:', max_selected_features
    print max_ecr

    max_features.append(max_selected_features)

    return max_features


if __name__ == "__main__":
    PCA_data = pandas.read_csv('data/')
    original_data = pandas.read_csv('data/MDP_Original_data2.csv')
    FAMD_data = pandas.read_csv('data/FAMD_features_discretized.csv')
    #pca_data = pandas.read_csv('data/pca_discretized_natural_bins.csv')
    #selected_features = ['Level', 'probDiff']
    #selected_features = ['symbolicRepresentationCount']
    #ECR_value = induce_policy_MDP2(original_data, selected_features)
    total_data = pandas.concat([original_data, FAMD_data], axis=1)
    selected_features = ['Level', 'cumul_Interaction']
    induce_policy_MDP2(total_data,selected_features)
    #checkFeatures(total_data)
