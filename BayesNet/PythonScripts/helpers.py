from math import factorial,log as fac,log
from operator import itemgetter
from collections import Counter
from itertools import permutations

def get_matching_records(data,dim,val):
    return [rec for rec in data if rec[dim] == val]

def get_dimension_data(data,dim):
    return [rec[dim] for rec in data]

def get_dim_measurements(data,dimension):
    '''Get number of positive and negative measurements for dimension'''
    num_pos = len(get_matching_records(data,dimension,1))
    num_neg = len(get_matching_records(data,dimension,0))
    return {'positive': num_pos, 'negative': num_neg}

def log_fac(n):
    return sum([log(i) for i in range(1,n+1)])

def random_match_prob(N,r):
    return log_fac(r-1) - log_fac(N+r-1)

def upward_adjustment(num_pos,num_neg):
    return log_fac(num_pos) + log_fac(num_neg)

def calc_p_dim_prob(data,c_dim,p_dim,dim_val,r):
    '''Gets probability for a given parent dimension'''
    filtered_data = get_matching_records(data,p_dim,dim_val)
    matching_children_measurments = get_dim_measurements(filtered_data,c_dim)
    rand_match_prob = random_match_prob(len(filtered_data),r)
    upward_adj = upward_adjustment(matching_children_measurments['positive'],matching_children_measurments['negative'])
    return rand_match_prob+upward_adj

def g(data,dimension,parent_dimensions=[]):
    '''Calculates the parent probability'''
    num_unique_instatiations = len(Counter(get_dimension_data(data,dimension)))
    r = num_unique_instatiations
    if len(parent_dimensions) == 0:
        dim_measurments = get_dim_measurements(data,dimension)
        rand_match_prob = random_match_prob(len(data),r)
        upward_adj = upward_adjustment(dim_measurments['positive'],dim_measurments['negative'])
        return rand_match_prob + upward_adj
    else:
        p_dim_prob_calc = lambda p_dim,dim_val: calc_p_dim_prob(data,dimension,p_dim,dim_val,r)
        all_products = [p_dim_prob_calc(p_dim,1)+p_dim_prob_calc(p_dim,0) for p_dim in parent_dimensions];
        return reduce(lambda x,y: x+y,all_products)

def k2(data,u,dim_ordering):
    '''Find belief network structure for given ordering'''
    dim_parents = {}
    for dim in dim_ordering:
        pi_i = []
        p_old = g(data,dim,pi_i)
        ok = True
        while ok and len(pi_i) < u:
            is_pred = lambda p_dim: dim_ordering.index(p_dim) < dim_ordering.index(dim) and p_dim not in pi_i
            preds = [p_dim for p_dim in dim_ordering if is_pred(p_dim)]
            new_p_vals = {p_dim: g(data,dim,pi_i + [p_dim]) for p_dim in preds}
            p_new_dim,p_new = max(new_p_vals.iteritems(),key=itemgetter(1)) if len(new_p_vals) > 0 else (None,None)
            if p_new != None and p_new > p_old:
                p_old = p_new
                pi_i += [p_new_dim]
            else:
                ok = False
        dim_parents[dim] = pi_i
    return dim_parents

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
        matching_data = get_matching_records(data,dim,value_combination[dim])
    return matching_data

def get_data_table(train_set,network,class_values):
    all_data = {}
    for cv in class_values:
        class_data = get_matching_records(train_set,'Class',cv)
        all_data[cv] = {}
        for dim in network:
            p_dims = network[dim]
            count_table = get_count_table(class_data,dim,p_dims)
            all_data[cv][dim] = count_table

    return all_data

def get_count_table(data,dim,p_dims):
    '''Construct dictionary to represent table of counts of dimension values given parent values'''
    dim_counts = Counter(get_dimension_data(data,dim))
    possible_values = dim_counts.keys()
    get_count = lambda data,dim_val: len(get_matching_records(data,dim,dim_val))
    if len(p_dims) == 0:
        return {dim_val: [(get_count(data,dim_val),{})] for dim_val in possible_values}
    parent_val_combs = find_all_parent_val_combinations(len(p_dims),possible_values)
    get_mapped_comb = lambda comb: {p_dims[i]: comb[i] for i in range(len(p_dims))}
    mapped_combs = [get_mapped_comb(comb) for comb in parent_val_combs]
    get_all_counts = lambda dim_val: [(get_count(find_matching_data(data,p_dims,parent_comb),dim_val),parent_comb) for parent_comb in mapped_combs]
    return {dim_val: get_all_counts(dim_val) for dim_val in possible_values}
