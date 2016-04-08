from helpers import k2,get_matching_records,get_count_table,find_matching_data
from random import randint,shuffle
from csv import DictReader
from operator import itemgetter

with open('../Data/forestFireData.csv','r') as f:
    reader = DictReader(f)
    convert_to_ints = lambda rec: {dim: int(rec[dim]) for dim in rec}
    data = [convert_to_ints(rec) for rec in list(reader)]

dimensions = ['Storms','BusTourGroup','Lightning','Campfire','Thunder']
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
classification_dims = [dim for dim in dimensions]

def find_class_prob(record,train_set,network,target_dim,class_val,dims):
    probs = []
    for dim in dims:
        #find probability dim value indicates class_val given its parents
        p_dims = network[dim]
        p_vals = {d: record[d] for d in p_dims}
        matching_parents = find_matching_data(train_set,p_dims,p_vals)
        matching_class_and_parents = [rec for rec in matching_parents if rec[target_dim] == class_val and rec[dim] == record[dim]]
        probs.append(len(matching_class_and_parents)/float(len(matching_parents)))
    return reduce(lambda x,y: x*y,probs)

fold_accuracies = {}
for i in range(k):
    test_start_index,test_end_index = i*subset_size, i*subset_size + subset_size
    test_set = [bootstrapped_data[j] for j in range(test_start_index,test_end_index)]
    head_indices = [j for j in range(0,test_start_index)]
    tail_indices = [j for j in range(test_end_index,len(bootstrapped_data)-1)]
    train_set = [bootstrapped_data[j] for j in head_indices + tail_indices]
    network = k2(data,3,classification_dims)
    correct = 0
    total = 0
    for record in test_set:
        class_probs = {}
        for class_value in target_class_values:
            class_prob = find_class_prob(record,train_set,network,'Class',class_value,dimensions)
            class_probs[class_value] = class_prob
        guess = max(class_probs.iteritems(),key=itemgetter(1))[0]
        total += 1
        if guess == record['Class']:
            correct += 1
    fold_accuracies[i] = correct/float(total)

print fold_accuracies
all_accuracies = [fold_accuracies[i] for i in fold_accuracies]
average = sum(all_accuracies)/float(len(all_accuracies))
print average
