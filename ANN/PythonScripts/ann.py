import pandas as pd
import numpy as np
from helpers import *
from collections import Counter

classification_field = 'Species'
num_hidden_layer_nodes = 3
data_file_path = '../Data/iris.csv'

data = pd.read_csv(data_file_path)

fields = [col for col in data.columns if col != classification_field]
num_classes = len(Counter(data[classification_field]))

hidden_weights = np.random.rand(len(fields),num_hidden_layer_nodes)
hidden_bias = np.random.rand(1,num_hidden_layer_nodes)

output_weights = np.random.rand(num_hidden_layer_nodes,num_classes)
output_bias = np.random.rand(1,num_classes)

test_record = data.iloc[0]
test_input = np.asarray([test_record[col] for col in fields])

input_forward = np.dot(test_input,hidden_weights)

print input_forward
