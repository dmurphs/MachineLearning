import pandas as pd
from helpers import id3, classify_test_case, print_tree

#read in the training and test data with pandas read_csv function
training_data = pd.read_csv('../Data/train.csv')
test_data = pd.read_csv('../Data/test.csv')

# remove class from attributes to classify on
attributes_to_process = [c for c in training_data if c != 'class']

# Create the decision tree to use for classification
decision_tree = id3(training_data,attributes_to_process)
#print print_tree(decision_tree)

#get test records as a list
test_records = [record[1] for record in test_data.iterrows()]

# loop over and compare classification from tree to actual classification
# and keep track of the number correct
total_test_records = len(test_records)
num_correct = 0
index = 0
for record in test_records:
    tree = decision_tree
    guess = classify_test_case(tree,record)
    if guess == record['class']:
        num_correct += 1
    index += 1

print '%i correct out of %i' %(num_correct,total_test_records)
print '%f percent correct' %(100*num_correct/float(total_test_records))
