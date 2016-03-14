from csv import DictReader
from helpers import generalize_S,prune_G,specialize_G,all_attributes_consistent,S_and_G_equal,k_fold_cross_validation

training_data_cols = []
training_data = []
with open('../Data/trainingDataCandElim.csv','r') as f:
    training_data_iterable = DictReader(f)
    training_data = list(training_data_iterable)


print k_fold_cross_validation(training_data,10,'class')
