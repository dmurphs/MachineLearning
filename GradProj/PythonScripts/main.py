# -*- coding: utf-8 -*-

#------------READING IN DATA---------------------------------------------------
import pandas as pd
import numpy as np

mushroom_data = pd.read_csv('../Data/agaricus-lepiota.data')
#shuffle the data
mushroom_data = mushroom_data.iloc[np.random.permutation(len(mushroom_data))]
mushroom_data.reset_index(drop=True)

num_test = 100
training_records = mushroom_data[num_test:].values
test_records = mushroom_data[:num_test].values

#------------K NEAREST NEIGHBOR CLASSIFICATION---------------------------------
from KNN.helpers import get_closest_k,vote_by_neighbor_weights
from KNN.knnreadable import KNNReadable

print 'K-Nearest Neighbor Classification'
print 'Computing...'

k = 5

#assign numerical values to each value so that numerical classifiers can use information
def get_value_map(unique_values):
    return {unique_values[i]: i for i in range(len(unique_values))}
    
distinct_col_vals = {col: get_value_map(mushroom_data[col].unique()) for col in mushroom_data.columns}

def get_knn_readable(v):
    classification = v[0]
    values = []
    for i in range(1,len(mushroom_data.columns)):
        col = mushroom_data.columns[i]
        col_val = v[i]
        numerical_val = distinct_col_vals[col][col_val]
        values.append(numerical_val)
    return KNNReadable(values,classification)
    
training_data = [get_knn_readable(v) for v in training_records]

test_data = [get_knn_readable(v) for v in test_records]

correct = 0
total = 0
for test_obj in test_data:
    k = len(training_data) if k == 0 else k
    closest_k = get_closest_k(test_obj,training_data,k)
    test_obj.guess = vote_by_neighbor_weights(test_obj,closest_k)

num_correct = len([td for td in test_data if td.name == td.guess])
print 'Results: %i correct out of %i' %(num_correct,num_test)


#--------------DECISION TREE CLASSIFICATION------------------------------------
print '\n'
from DTree.helpers import *

print 'Decision Tree Classification'
print 'Computing...'

classification_attributes = [col for col in mushroom_data.columns if col != 'class']
# Create the decision tree to use for classification
train_data = mushroom_data.iloc[[i for i in range(len(mushroom_data)) if i >= num_test]]
test_data = mushroom_data.iloc[[i for i in range(len(mushroom_data)) if i < num_test]]
attr_vals = {attr: list(set(mushroom_data[attr])) for attr in classification_attributes}

decision_tree = id3(train_data,classification_attributes,attr_vals)

# loop over and compare classification from tree to actual classification
# and keep track of the number correct
test_records = [record[1] for record in test_data.iterrows()]
total_test_records = len(test_data)
num_correct = 0
index = 0
for record in test_records:
    tree = decision_tree
    guess = classify_test_case(tree,record)
    if guess == record['class']:
        num_correct += 1
    index += 1

print 'Results: %i correct out of %i' %(num_correct,total_test_records)

#------------DETERMINE IF THERE IS ANY PARTICULAR FEATURE ASSOCIATED WITH EDIBILITY
print '\n'
print 'Analysis of correlation for each classification attribute'
print 'Computing...'
correct = []
for attr in classification_attributes:
    num_correct = 0
    index = 0
    d_tree = id3(train_data,[attr],attr_vals)
    for record in test_records:
        guess = classify_test_case(d_tree,record)
        if guess == record['class']:
            num_correct += 1
        index += 1
    print '%s: %i correct out of %i' %(attr,num_correct,index)
    correct.append(num_correct)
print '%i over 70 percent correct' %(len([c for c in correct if c >= 70]))
    
#------------ANALYZE WHICH ODORS ARE CORRELATED WITH EDIBILITY-----------------
class_odors = {}
for classification in mushroom_data['class'].unique():
    class_data = mushroom_data[mushroom_data['class'] == classification]
    unique_odors = class_data['spore-print-color'].unique()
    odor_freqs = class_data['spore-print-color'].value_counts()
    class_odors[classification] = odor_freqs
    
  
'''  
print class_odors

for c in class_odors:
    abbrev_map = {'a': 'almond',
        'l': 'anise',
        'c': 'creosote',
        'y': 'fishy',
        'f': 'foul',
        'm': 'musty',
        'n': 'none',
        'p': 'pungent',
        's': 'spicy'}
    if c == 'p':
        name = 'Poisonous'
    else:
        name = 'Edible'
    
    print name
    
    freqs = class_odors[c]
    for freq in freqs.keys():
        numstr = str(freqs[freq])
        print '\t' + numstr + ' ' + abbrev_map[freq]'''
        
        
    
