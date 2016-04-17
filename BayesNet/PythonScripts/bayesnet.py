from helpers import k2,get_matching_records,find_matching_data,find_class_prob,get_best_network
from random import randint,shuffle
from csv import DictReader
from operator import itemgetter
from itertools import permutations

with open('../Data/forestFireData.csv','r') as f:
    reader = DictReader(f)
    convert_to_ints = lambda rec: {dim: int(rec[dim]) for dim in rec}
    data = [convert_to_ints(rec) for rec in list(reader)]

classification_dims = ['Storms','BusTourGroup','Lightning','Campfire','Thunder']
target_class_values = [0,1]

positive_records = get_matching_records(data,'Class',1)
negative_records = get_matching_records(data,'Class',0)

while len(positive_records) < len(negative_records):
    index = randint(0,len(positive_records) - 1)
    value = positive_records[index]
    positive_records.append(value)

bootstrapped_data = positive_records + negative_records

shuffle(bootstrapped_data)

k = 10

subset_size = len(bootstrapped_data)/k

try_random_orderings = False
num_k2_random_orderings = 90

fold_accuracies = {}
for i in range(k):
    cf_lookup = {1: 'positive', 0: 'negative'}
    confustion_matrix = {'positive': {'positive': 0, 'negative': 0}, 'negative': {'positive': 0, 'negative': 0}}

    test_start_index,test_end_index = i*subset_size, i*subset_size + subset_size
    test_set = [bootstrapped_data[j] for j in range(test_start_index,test_end_index)]
    head_indices = [j for j in range(0,test_start_index)]
    tail_indices = [j for j in range(test_end_index,len(bootstrapped_data)-1)]
    train_set = [bootstrapped_data[j] for j in head_indices + tail_indices]

    if try_random_orderings:
        all_perms = list(permutations(classification_dims))
        shuffle(all_perms)
        network = get_best_network(train_set,3,all_perms[:num_k2_random_orderings])
    else:
        network = k2(train_set,5,classification_dims)[0]

    for node in network:
        print 'Node %s has parents %s' %(node,str(network[node]))
    correct = 0
    total = 0
    for record in test_set:
        class_probs = {}
        for class_value in target_class_values:
            class_prob = find_class_prob(record,train_set,network,'Class',class_value,classification_dims)
            class_probs[class_value] = class_prob
        guess = max(class_probs.iteritems(),key=itemgetter(1))[0]
        total += 1
        actual = record['Class']
        if guess == actual:
            correct += 1
        confustion_matrix[cf_lookup[actual]][cf_lookup[guess]] += 1
    fold_acc = correct/float(total)
    fold_accuracies[i] = fold_acc
    print 'Fold number %i had %f percent accuracy' %(i+1,fold_acc*100)
    print 'Confusion Matrix: ', confustion_matrix
    print '\n'

all_accuracies = [fold_accuracies[i] for i in fold_accuracies]
average = sum(all_accuracies)/float(len(all_accuracies))
print 'Average Accuracy: %f percent' %(average*100)
