from random import shuffle
from collections import Counter
from csv import DictReader

def get_generalized_val(s_val,training_val):
    '''Returns the value if they are equal otherwise returns True'''
    if s_val == training_val:
        return s_val
    elif s_val == False:
        return training_val
    else:
        return True

def is_inconsistent(val1,val2):
    '''Returns true if the two values are inconsistent'''
    return (val1 != True and val2 != True) and (val1 != val2)

def all_attributes_consistent(record1,record2):
    '''Returns a boolean indicating whether two records are consistent in all attributes'''
    inconsistent_values = [attr for attr in record1 if is_inconsistent(record1[attr],record2[attr])]
    return len(inconsistent_values) == 0

def get_all_possible_attr_vals():
    '''Looks through the dataset to get all possible values for each attribute'''
    training_data_cols = []
    training_data = []
    with open('../Data/trainingDataCandElim.csv','r') as f:
        training_data_iterable = DictReader(f)
        training_data = list(training_data_iterable)

    attr_vals = {}
    for record in training_data:
        for attr in record:
            if attr in attr_vals and record[attr] not in attr_vals[attr]:
                attr_vals[attr].append(record[attr])
            else:
                attr_vals[attr] = [record[attr]]
    return attr_vals

def can_restrict_G_attr(attr,training_record,G,S):
    '''Check if it is possible to further restrict an attribute G based on a training record and S'''
    no_restriction_in_G = len(filter(lambda x: x != True, [rec[attr] for rec in G])) == 0
    restricted_in_S = S[attr] != True
    s_not_equal_training = training_record[attr] != S[attr]

    return (no_restriction_in_G and restricted_in_S and s_not_equal_training)

def get_unrestricted_specs(attr,G_val,possible_vals,bad_val):
    '''Given a negative record and a value of False for S in given attribute, get all possibilities for that attribute in G'''
    filtered_vals = [val for val in possible_vals if val != bad_val]
    return [{g_attr: (G_val[g_attr] if g_attr != attr else val) for g_attr in G_val} for val in filtered_vals]

def increase_specializations(G_val,training_record,G,S):
    '''Returns all possible specializations of a given 'G_val'''
    get_attr_val = lambda g_attr,attr: S[attr] if g_attr == attr else G_val[g_attr]
    get_specs = lambda attr: {g_attr: get_attr_val(g_attr,attr) for g_attr in G_val}
    new_specs_by_S = [get_specs(attr) for attr in G_val if can_restrict_G_attr(attr,training_record,G,S) and S[attr] != False]

    all_possible_attr_vals = get_all_possible_attr_vals()
    unrestricted_specs = [get_unrestricted_specs(attr,G_val,all_possible_attr_vals[attr],training_record[attr]) for attr in G_val
        if can_restrict_G_attr(attr,training_record,G,S) and S[attr] == False]
    union_unrestricted_specs = [spec for spec_set in unrestricted_specs for spec in spec_set]

    return new_specs_by_S + union_unrestricted_specs

def prune_G(training_record,G):
    '''Prune G to remove descriptions inconsistent with the positive example'''
    return [record for record in G if all_attributes_consistent(record,training_record)]

def generalize_S(training_record,S):
    '''Generalize S to include positive training example'''
    return {attr: get_generalized_val(S[attr],training_record[attr]) for attr in S}

def specialize_G(training_record,G,S):
    '''Specialize G to exclude negative training example while staying consistent with S'''
    bad_specs = [spec for spec in G if all_attributes_consistent(spec,training_record)]
    new_G = [spec for spec in G if spec not in bad_specs]

    for spec in bad_specs:
        new_specs = increase_specializations(spec,training_record,new_G,S)
        for new_spec in new_specs:
            new_G.append(new_spec)
    return new_G

def S_and_G_equal(S,G):
    '''Check if S and G have converged'''
    num_unequal_attrs = len([attr for attr in G[0] if G[0][attr] != S[attr]])
    return len(G) == 1 and (num_unequal_attrs == 0)

def get_S_and_intermediate_hypotheses(G_val,S):
    '''Recursively get all intermediate_hypotheses between G and S as well as S'''
    get_specified_attrs = lambda hypthesis: [attr for attr in hypthesis if hypthesis[attr] != True]
    specified_in_G = get_specified_attrs(G_val)
    specified_in_S = get_specified_attrs(S)
    num_specified_in_G = len(specified_in_G)
    num_specified_in_S = len(specified_in_S)

    if num_specified_in_G >= num_specified_in_S - 1:
        return [S]

    else:
        get_generalized_S_attr_val = lambda S,attr,gen_target: S[attr] if attr != gen_target else True
        split_S = [{attr: get_generalized_S_attr_val(S,attr,target_attr) for attr in S} for target_attr in specified_in_S]
        intermediates = [get_S_and_intermediate_hypotheses(G_val,S_val) for S_val in split_S]
        return [S] + [hypothesis for intermediate in intermediates for hypothesis in intermediate]

def candidate_elimination_learner(training_data,class_label_col,positive_val,negative_val):
    '''Perform candidate elimination algorithm'''
    attr_cols = [col for col in training_data[0] if col != class_label_col]
    num_attr_cols = len(attr_cols)

    positive = 'Enjoy Sport'
    negative = 'Do Not Enjoy'

    G = [{attr_cols[i]: True for i in range(num_attr_cols)}]
    S = {attr_cols[i]: False for i in range(num_attr_cols)}

    i = 0
    while not S_and_G_equal(S,G) and i < len(training_data):
        record = training_data[i]
        if record[class_label_col] == positive:
            G = prune_G(record,G)
            S = generalize_S(record,S)
        else:
            G = specialize_G(record,G,S)
        i += 1

    all_hypotheses = [G[0]] + get_S_and_intermediate_hypotheses(G[0],S)

    return all_hypotheses

def classify_test_data(test_data,all_hypotheses,class_label_col,positive_val,negative_val):
    '''Returns decimal accuracy of hypothesis returned by classifier and confustion_matrix'''
    correct = 0
    total = 0
    confusion_matrix = {positive_val: {positive_val: 0, negative_val: 0}, negative_val: {positive_val: 0, negative_val: 0}}
    for record in test_data:
        actual_class = record[class_label_col]
        record_classifier_data = {attr: record[attr] for attr in record if attr != class_label_col}
        get_estimate = lambda hypothesis: positive_val if all_attributes_consistent(record_classifier_data,hypothesis) else negative_val
        estimates = [get_estimate(hypothesis) for hypothesis in all_hypotheses]
        most_common_estimate = Counter(estimates).most_common(1)[0][0]
        confusion_matrix[actual_class][most_common_estimate] += 1
        if actual_class == most_common_estimate:
            correct += 1
        total += 1

    return correct/float(total), confusion_matrix

def k_fold_cross_validation(training_data,k,class_label_col):
    '''Perform k_fold_cross_validation on dataset'''
    shuffle(training_data)
    subset_size = len(training_data)/k
    fold_accuracies = []
    all_confusion_matrices = []
    positive = 'Enjoy Sport'
    negative = 'Do Not Enjoy'
    for i in range(k):
        test_data_start_index,test_data_end_index = i*subset_size, i*subset_size + subset_size
        fold_test_data = [training_data[i] for i in range(test_data_start_index,test_data_end_index)]
        training_indices1 = range(0,test_data_start_index)
        training_indices2 = range(test_data_end_index,len(training_data)- 1)
        training_indices = training_indices1 + training_indices2
        fold_training_data = [training_data[i] for i in training_indices]
        all_hypotheses = candidate_elimination_learner(fold_training_data,class_label_col,positive,negative)
        classification_results = classify_test_data(fold_test_data,all_hypotheses,class_label_col,positive,negative)
        fold_accuracy = classification_results[0]
        confusion_matrix = classification_results[1]
        fold_accuracies.append(fold_accuracy)
        all_confusion_matrices.append(confusion_matrix)

        aggregated_confusion_matrix = {positive: {positive: 0, negative: 0}, negative: {positive: 0, negative: 0}}
        for cm in all_confusion_matrices:
            for actual_class in cm:
                for predicted_class in cm[actual_class]:
                    aggregated_confusion_matrix[actual_class][predicted_class] += cm[actual_class][predicted_class]

    print 'Cross Validation Classification accuracy: %.2f percent' % (sum(fold_accuracies)/len(fold_accuracies)*100)
    print 'Confusion Matrix: ', aggregated_confusion_matrix
