import collections
import numpy as np
import pandas
import mdptoolbox, mdptoolbox.example
import argparse
from MDP_function2 import *

def checkFeatures(original_data):
    cols = list(original_data.columns)

    max_ecr = 0.0
    max_selected_features = []
    for i in range(7, len(cols)):
        selected_features = [cols[i]]
        ecr = induce_policy_MDP2(original_data, selected_features)
        if (ecr > max_ecr):
            max_ecr = ecr
            max_selected_features = selected_features
    print 'best feature 1:', max_selected_features
    print max_ecr

    for i in range(7, len(cols)):
        print cols[i],max_selected_features[0]
        if (cols[i] != max_selected_features[0]):
            selected_features = [max_selected_features[0],cols[i]]
            ecr = induce_policy_MDP2(original_data, selected_features)
            if (ecr > max_ecr):
                max_ecr = ecr
                max_selected_features = selected_features
    print 'best feature 2:', max_selected_features
    print max_ecr
    return max_selected_features


if __name__ == "__main__":
    original_data = pandas.read_csv('data/discreitized_float_data.csv')
    #selected_features = ['Level', 'probDiff']
    #selected_features = ['symbolicRepresentationCount']
    #ECR_value = induce_policy_MDP2(original_data, selected_features)
    print checkFeatures(original_data)
