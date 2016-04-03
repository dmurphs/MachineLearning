from math import factorial as fac
from operator import itemgetter
from collections import Counter
from itertools import permutations

def get_dim_measurements(data,dimension):
    '''Get number of positive and negative measurements for dimension'''
    num_child_pos = len(data[data[dimension] == 1])
    num_child_neg = len(data[data[dimension] == 0])
    return {'positive': num_child_pos, 'negative': num_child_neg}

def random_match_prob(N,r):
    return fac(r-1)/float(fac(N+r-1))

def upward_adjustment(num_pos,num_neg):
    return fac(num_pos)*fac(num_neg)

def calc_p_dim_prob(data,c_dim,p_dim,dim_val,r):
    '''Gets probability for a given parent dimension'''
    filtered_data = data[data[p_dim] == dim_val]
    matching_children_measurments = get_dim_measurements(filtered_data,c_dim)
    rand_match_prob = random_match_prob(len(filtered_data),r)
    upward_adj = upward_adjustment(matching_children_measurments['positive'],matching_children_measurments['negative'])
    return rand_match_prob*upward_adj

def g(data,dimension,parent_dimensions=[]):
    '''Calculates the parent probability'''
    num_unique_instatiations = len(Counter(data[dimension]))
    r = num_unique_instatiations
    if len(parent_dimensions) == 0:
        dim_measurments = get_dim_measurements(data,dimension)
        rand_match_prob = random_match_prob(len(data),r)
        upward_adj = upward_adjustment(dim_measurments['positive'],dim_measurments['negative'])
        return rand_match_prob*upward_adj
    else:
        p_dim_prob_calc = lambda p_dim,dim_val: calc_p_dim_prob(data,dimension,p_dim,dim_val,r)
        all_products = [p_dim_prob_calc(p_dim,1)*p_dim_prob_calc(p_dim,0) for p_dim in parent_dimensions];
        return reduce(lambda x,y: x*y,all_products)

def k2(data,u,dim_ordering):
    '''Find belief network structure for given ordering'''
    dimensions = data.columns
    dim_parents = {}
    for dim in dimensions:
        pi_i = []
        p_old = g(data,dim,pi_i)
        ok = True
        while ok and len(pi_i) < u:
            is_pred = lambda p_dim: dim_ordering.index(p_dim) < dim_ordering.index(dim) and p_dim not in pi_i
            preds = [p_dim for p_dim in dim_ordering if is_pred(p_dim)]
            new_p_vals = {p_dim: g(data,dim,pi_i + [p_dim]) for p_dim in preds}
            p_new_dim,p_new = max(new_p_vals.iteritems(),key=itemgetter(1)) if len(new_p_vals) > 0 else (None,0)
            if p_new > p_old:
                p_old = p_new
                pi_i += [p_new_dim]
            else:
                ok = False
        dim_parents[dim] = pi_i
    return dim_parents

def k2_all_orderings(data,dimensions):
    perms = permutations(dimensions)
    structures = [k2(data,3,perm) for perm in perms]
    return structures

def find_all_parent_val_combinations(num_dims,possible_values,all_combs=[]):
    if len(all_combs) == 2**num_dims or num_dims == 0:
        return all_combs

    new_combs_with_val = lambda val: [comb + [val] for comb in all_combs]
    new_combs = [[val] for val in possible_values] if len(all_combs) == 0 else reduce(lambda x,y: x + y,[new_combs_with_val(val) for val in possible_values])
    return find_all_parent_val_combinations(num_dims,possible_values,new_combs)

def find_matching_data(data,dims,value_combination):
    '''Get data from data that matches the combination of values from the given dimensions'''
    matching_data = data
    for dim in dims:
        matching_data = matching_data[matching_data[dim] == value_combination[dim]]
    return matching_data

def get_prob_table(data,dim,p_dims):
    '''Construct dictionary to represent table of conditional probability of dimension values given parent values'''
    dim_counts = Counter(data[dim])
    possible_values = dim_counts.keys()
    get_prob = lambda data,dim_val: len(data[data[dim] == dim_val])/float(len(data))
    if len(p_dims) == 0:
        return {dim_val: [(get_prob(data,dim_val),{})] for dim_val in possible_values}
    parent_val_combs = find_all_parent_val_combinations(len(p_dims),possible_values)
    get_mapped_comb = lambda comb: {p_dims[i]: comb[i] for i in range(len(p_dims))}
    mapped_combs = [get_mapped_comb(comb) for comb in parent_val_combs]
    get_all_probs = lambda dim_val: [(get_prob(find_matching_data(data,p_dims,parent_comb),dim_val),parent_comb) for parent_comb in mapped_combs]
    return {dim_val: get_all_probs(dim_val) for dim_val in possible_values}

def classify_record(data,test_record,network,target_class,possible_values):
    '''Classify a record'''
    for dim in network:
        p_dims = network[dim]
        p_dim_comb = {p_dim: test_record[p_dim] for p_dim in p_dims}
        table = get_prob_table(data,dim,p_dims)
