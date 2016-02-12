from math import sqrt
from operator import itemgetter
from collections import Counter
from knnreadable import KNNReadable

def get_distance(c1,c2):
    '''Take two lists of attributes and compute euclidean distance'''
    return sqrt(sum([(c2[i]-c1[i])**2 for i in range(len(c1))]))

def get_closest_k(test_obj,training_data,k):
    '''Get the closest k training items to test item'''
    #If we are using the whole training set return sorted list of tuples containing objects and their distances
    if k == len(training_data):
        return sorted([(obj,get_distance(test_obj.measurements,obj.measurements))
            for obj in training_data],key=itemgetter(1))

    #Otherwise we compute a list of the closest k from the training set
    closest_k = []
    for training_obj in training_data:
        dist = get_distance(test_obj.measurements,training_obj.measurements)
        closest_k.append((training_obj, dist))
        closest_k = sorted(closest_k,key=itemgetter(1))[:k]
    return closest_k

def vote_by_neighbor_weights(test_obj,neighbors):
    '''Get weighted vote from list of training_data'''
    weighted_neighbors = [(n[0],1/(n[1]**2)) for n in neighbors]
    votes = {}
    for wn in weighted_neighbors:
        obj_name = wn[0].name
        weight = wn[1]
        if obj_name in votes:
            votes[obj_name] += weight
        else:
            votes[obj_name] = weight
    return max(votes.iteritems(),key=itemgetter(1))[0]

def normalize(data):
    '''Normalize measurements in list of KNNReadable objects'''
    measurements = [obj.measurements for obj in data]
    split_measurements = {}
    for m in measurements:
        for i in range(len(m)):
            if i in split_measurements:
                split_measurements[i].append(m[i])
            else:
                split_measurements[i] = [m[i]]
    measurement_max_min = {i:(max(split_measurements[i]),min(split_measurements[i])) for i in split_measurements}

    for obj in data:
        for i in range(len(obj.measurements)):
            val = obj.measurements[i]
            meas_max = measurement_max_min[i][0]
            meas_min = measurement_max_min[i][1]
            obj.measurements[i] = (val - meas_min)/(meas_max - meas_min)

    return data
