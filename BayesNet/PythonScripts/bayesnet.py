from helpers import k2,get_matching_records,get_data_table,find_matching_data
from random import randint,shuffle
from csv import DictReader
from operator import itemgetter

with open('../Data/forestFireData.csv','r') as f:
    reader = DictReader(f)
    convert_to_ints = lambda rec: {dim: int(rec[dim]) for dim in rec}
    data = [convert_to_ints(rec) for rec in list(reader)]

dimensions = ['Storms','BusTourGroup','Lightning','Campfire','Thunder']

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

all_accuracies = []
for i in range(k):
    test_start_index,test_end_index = i*subset_size, i*subset_size + subset_size
    test_set = [bootstrapped_data[i] for i in range(test_start_index,test_end_index)]
    head_indices = [i for i in range(0,test_start_index)]
    tail_indices = [i for i in range(test_end_index,len(bootstrapped_data)-1)]
    train_set = [bootstrapped_data[i] for i in head_indices + tail_indices]
    network = k2(train_set,3,classification_dims)
    network = {'Thunder': ['Lightning'], 'Campfire': ['BusTourGroup'], 'Storms': [], 'BusTourGroup': [], 'Lightning': ['Storms']}
    data_table = get_data_table(train_set,network,[0,1])
    total = 0
    correct = 0
    for record in test_set:
        total += 1
        all_class_probs = {0: [], 1: []}
        probs = {}
        for c in [0,1]:
            for dim in network:
                record_val = record[dim]
                val_data_table = data_table[c][dim][record_val]
                p_dims = network[dim]
                parent_vals = {p_dim: record[p_dim] for p_dim in p_dims}
                all_fields_match = lambda p_vals: len([dim for dim in p_vals if p_vals[dim] != parent_vals[dim]])
                if len(p_dims) > 0:
                    matching_parent_count = [t[0] for t in val_data_table if all_fields_match(t[1])][0] if len(p_dims) > 0 else None
                    total_parents_count = sum([t[0] for t in val_data_table])
                    prob = (1+matching_parent_count)/(1000 + float(total_parents_count))
                    all_class_probs[c].append(prob)
                else:
                    matching_count = val_data_table[0][0]
                    total_in_class = len(get_matching_records(data,'Class',c))
                    prob = (1 + matching_count)/(1000 + float(total_in_class))
                    all_class_probs[c].append(prob)
            probs[c] = reduce(lambda x,y: x*y, all_class_probs[c])
        guess = max(probs.iteritems(),key=itemgetter(1))[0]
        if guess == record['Class']:
            correct += 1

    all_accuracies.append(correct/float(total))

print sum(all_accuracies)/float(len(all_accuracies))
