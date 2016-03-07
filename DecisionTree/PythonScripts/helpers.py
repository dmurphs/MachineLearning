from math import log
from collections import Counter
from operator import itemgetter

def calculate_entropy(attribute_data):
    '''Calcuate the entropy of given attribute data'''
    frequencies = Counter(attribute_data)
    probabilities = [frequencies[c]/float(len(attribute_data)) for c in frequencies]
    return -sum([p*log(p,2) for p in probabilities])

def hxt(attribute, dataset):
    '''Calculates H(X,T) for for an attribute X and dataset T'''
    dataset_size = len(dataset)
    attribute_vals = set(dataset[attribute])
    hxt_val = 0
    for i in attribute_vals:
        t_i = dataset[dataset[attribute] == i]['class']
        prob_t_i = len(t_i)/float(dataset_size)
        hxt_val += prob_t_i*calculate_entropy(t_i)
    return hxt_val

def calculate_information_gain(training_dataset, attribute):
    '''Calculates information gain from a particular attribute'''
    output_data = training_dataset['class']
    info_gain = calculate_entropy(output_data) - hxt(attribute,training_dataset)
    return info_gain

def get_most_common_class(dataset):
    '''return the most common class in a dataset'''
    most_common = Counter(dataset['class']).most_common(1)
    return most_common[0][0]

def id3(dataset, attributes):
    '''Perform recursive id3 algorithm on a dataset and given attributes'''
    root_node = {'class_name': None, 'decision_attribute': None, 'children': {}}

    unique_data_classes = list(set(dataset['class']))
    if len(unique_data_classes) == 1:
        root_node['class_name'] = unique_data_classes[0]

    elif len(attributes) == 0:
        root_node['class_name'] = get_most_common_class(dataset)

    else:
        col_gains = [(col,calculate_information_gain(dataset,col)) for col in attributes]
        largest_info_gain_attr = max(col_gains, key=itemgetter(1))[0]
        root_node['decision_attribute'] = largest_info_gain_attr

        for attr_value in range(0,11):
            remaining_data = dataset[dataset[largest_info_gain_attr] == attr_value]
            if len(remaining_data) == 0:
                root_node['children'][attr_value] = {
                    'class_name': get_most_common_class(dataset),
                    'decision_attribute': None,
                    'children': None
                }
            else:
                remaining_attributes = [a for a in attributes if a != largest_info_gain_attr]
                root_node['children'][attr_value] = id3(remaining_data,remaining_attributes)
    return root_node

def classify_test_case(decision_tree, test_record):
    '''classify a test record using a decision tree'''
    tree = decision_tree
    class_name = tree['class_name']
    while class_name == None:
        children = tree['children']
        decision_attribute = tree['decision_attribute']
        test_decision_attr_val = test_record[decision_attribute]
        tree = children[test_decision_attr_val]
        class_name = tree['class_name']
    return class_name

def remove_n_lowest_info_gain_cols(n,attributes,dataset):
    info_gains = [(attr,calculate_information_gain(dataset,attr)) for attr in attributes]

    hightest_info_gain_attrs = attributes
    for i in range(n):
        min_attr = min(info_gains,key=itemgetter(1))[0]
        info_gains = [ig for ig in info_gains if ig[0] != min_attr]
        hightest_info_gain_attrs = [a for a in hightest_info_gain_attrs if a != min_attr]

    return hightest_info_gain_attrs
