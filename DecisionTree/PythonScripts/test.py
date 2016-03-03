import pandas as pd
from helpers import calculate_entropy, calculate_information_gain, get_most_common_class,hxt
from collections import Counter

training_data = pd.read_csv('../Data/train.csv')

clump_thickness_column = training_data['Clump Thickness']

#test_entropy = calculate_entropy(clump_thickness_column)
for col in training_data:
    print calculate_entropy(training_data[col])
#print test_entropy

test_hxt = hxt('Clump Thickness',training_data)
#print test_hxt

test_info_gain = calculate_information_gain(training_data, 'Clump Thickness')
for col in training_data:
    print calculate_information_gain(training_data,col)
#print test_info_gain
