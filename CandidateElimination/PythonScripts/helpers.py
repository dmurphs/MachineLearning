from random import shuffle

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

def can_restrict_G_attr(attr,training_record,G,S):
    no_restriction_in_G = len(filter(lambda x: x != True, [rec[attr] for rec in G])) == 0
    restricted_in_S = S[attr] != True
    s_not_equal_training = training_record[attr] != S[attr]

    return (no_restriction_in_G and restricted_in_S and s_not_equal_training)

def increase_specializations(G_value,training_record,G,S):
    '''Returns all possible specializations of a given 'G_value'''
    get_attr_val = lambda g_attr,attr: S[attr] if g_attr == attr else G_value[g_attr]
    get_specs = lambda attr: {g_attr: get_attr_val(g_attr,attr) for g_attr in G_value}
    new_specs = [get_specs(attr) for attr in G_value if can_restrict_G_attr(attr,training_record,G,S)]

    return new_specs

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

    '''get_new_spec_list = lambda spec: increase_specializations(spec,training_record,new_G,S)
    all_new_specs = [spec for spec in get_new_spec_list(bad_spec) for bad_spec in bad_specs]

    return new_G + all_new_specs'''

    for spec in bad_specs:
        new_specs = increase_specializations(spec,training_record,new_G,S)
        for new_spec in new_specs:
            new_G.append(new_spec)
    return new_G

def S_and_G_equal(S,G):
    '''Check if S and G have converged'''
    num_unequal_attrs = len([attr for attr in G[0] if G[0][attr] != S[attr]])
    return len(G) == 1 and (num_unequal_attrs == 0)

def candidate_elimination_learner(training_data,class_label_col,positive_val,negative_val):
    '''Perform candidate elimination algorithm'''
    attr_cols = [col for col in training_data[0] if col != class_label_col]
    num_attr_cols = len(attr_cols)

    positive = 'Enjoy Sport'
    negative = 'Do Not Enjoy'

    G = [{attr_cols[i]: True for i in range(num_attr_cols)}]
    S = {attr_cols[i]: False for i in range(num_attr_cols)}

    i = 0
    S_set = False
    while not S_and_G_equal(S,G) and i < len(training_data):
        record = training_data[i]
        if record[class_label_col] == positive:
            G = prune_G(record,G)
            S = generalize_S(record,S)
            S_set = True
        elif S_set:
            G = specialize_G(record,G,S)
        i += 1

    return S

def classify_test_data(test_data,hypothesis,class_label_col,positive_val,negative_val):
    '''Returns decimal accuracy of hypothesis returned by classifier'''
    correct = 0
    total = 0
    for record in test_data:
        actual_class = record[class_label_col]
        record_classifier_data = {attr: record[attr] for attr in record if attr != class_label_col}
        estimate = positive_val if all_attributes_consistent(record_classifier_data,hypothesis) else negative_val
        if actual_class == estimate:
            correct += 1
        total += 1

    return correct/float(total)

def k_fold_cross_validation(training_data,k,class_label_col):
    '''Perform k_fold_cross_validation on dataset'''
    shuffle(training_data)
    subset_size = len(training_data)/k
    fold_accuracies = []
    positive = 'Enjoy Sport'
    negative = 'Do Not Enjoy'
    for i in range(k):
        test_data_start_index,test_data_end_index = i*subset_size, i*subset_size + subset_size
        is_in_fold_range = lambda i: i >= test_data_start_index and i < test_data_end_index
        fold_test_data = [training_data[i] for i in range(len(training_data)) if is_in_fold_range(i)]
        fold_training_data = [training_data[i] for i in range(len(training_data)) if not is_in_fold_range(i)]
        hypothesis = candidate_elimination_learner(fold_training_data,class_label_col,positive,negative)
        fold_accuracy = classify_test_data(fold_test_data,hypothesis,class_label_col,positive,negative)
        fold_accuracies.append(fold_accuracy)

    return 'Cross Validation Classification accuracy: %f percent' % (sum(fold_accuracies)/len(fold_accuracies)*100)
