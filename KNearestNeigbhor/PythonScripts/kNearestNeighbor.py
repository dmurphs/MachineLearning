import pandas as pd
from argparse import ArgumentParser
from helpers import get_distance,normalize,get_closest_k,vote_by_neighbor_weights
from collections import Counter
from operator import itemgetter
from knnreadable import KNNReadable

parser = ArgumentParser()
program_desc = 'Perform K Nearest Neigbhorhood Algorithm on Fruit Data'
parser = ArgumentParser(description=program_desc)
parser.add_argument('k', type=int, help='K-Value for K Nearest Neighbors Algorithm')
args = parser.parse_args()

k = int(args.k)

training_df = pd.read_csv('../Data/fruit.csv')
test_df = pd.read_csv('../Data/testFruit.csv')

get_knn_readable = lambda v: KNNReadable([float(v[0]),float(v[1]),float(v[2]),float(v[3])],v[4])

training_data = normalize([get_knn_readable(v) for v in training_df.values])
test_data = normalize([get_knn_readable(v) for v in test_df.values])

if k < 0:
    print 'Invalid Argument'

else:
    if k > len(training_data):
        print 'Not enough elements in training set for specified k value, setting k to size of training set.'
        k = len(training_data)
        
    for test_obj in test_data:
        k = len(training_data) if k == 0 else k
        closest_k = get_closest_k(test_obj,training_data,k)
        test_obj.guess = vote_by_neighbor_weights(test_obj,closest_k)
        #Performance: 1: 94%, 5: 96%, 10: 98%, 20: 98%, 50: 98%, 100: 99%

        #Commented region below simply takes votes of closest k items but has a
        #tie-breaker if 2 classes in the closest k occur at the same frequency.
        #If this happens just pick which class has the closest object
        '''if k == len(training_data):
            test_obj.guess = vote_by_neighbor_weights(test_obj,closest_k)
        elif k > 0:
            #get all of most common of closest k class names and pick the closest
            closest_k_names = [f[0].name for f in closest_k]
            freq_counter = Counter(closest_k_names).most_common()
            num_highest_occurence = freq_counter[0][1]
            highest_occuring_names = [name for name,count in freq_counter if count == num_highest_occurence]
            highest_occuring_dists = [(f.name,dist) for f,dist in closest_k if f.name in highest_occuring_names]
            test_obj.guess = min(highest_occuring_dists,key=itemgetter(1))[0]'''
            #Performance: 1: 94%, 5: 95%, 10: 98%, 20: 98%, 50: 98%, 100: 99%

    num_correct = len([f for f in test_data if f.guess == f.name])
    print 'Num correct: {0}'.format(num_correct)
    print 'Num wrong: {0}'.format(len(test_data) - num_correct)
    print 'Percent classified correctly: {0}'.format(num_correct*100/float(len(test_data)))
