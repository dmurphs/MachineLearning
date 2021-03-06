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
    '''Find belief network structure for given ordering along with network score'''
    dim_parents = {}
    all_probs = []
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
        all_probs.append(p_old)
        dim_parents[dim] = pi_i
    return dim_parents,sum(all_probs)

def find_all_parent_val_combinations(num_dims,possible_values,all_combs=[]):
    '''Recursively get all possible combinations for parents'''
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

def find_class_prob(record,train_set,network,target_dim,class_val,dims):
    '''Find the probability that a record belongs to a class'''
    probs = []
    for dim in dims:
        #find probability dim value indicates class_val given its parents
        p_dims = network[dim]
        p_vals = {d: record[d] for d in p_dims}
        matching_parents = find_matching_data(train_set,p_dims,p_vals)
        matching_class_and_parents = [rec for rec in matching_parents if rec[target_dim] == class_val and rec[dim] == record[dim]]
        prob = (1 + len(matching_class_and_parents))/float(1000 + len(matching_parents))
        probs.append(prob)
    return reduce(lambda x,y: x*y,probs)

def get_best_network(data,u,orderings):
    '''Get the best network from several different orderings'''
    networks = [k2(data,u,ordering) for ordering in orderings]
    return max(networks,key=itemgetter(1))[0]
