from math import factorial as fac
from operator import itemgetter

def get_dim_measurements(records,dimension):
    '''Get number of positive and negative measurements for dimension'''
    num_child_pos = len([rec for rec in records if rec[dimension] == 1])
    num_child_neg = len([rec for rec in records if rec[dimension] == 0])
    return {'positive': num_child_pos, 'negative': num_child_neg}

def random_match_prob(N,r):
    return fac(r-1)/float(fac(N+r-1))

def upward_adjustment(num_pos,num_neg):
    return fac(num_pos)*fac(num_neg)

def calc_p_dim_prob(data,c_dim,p_dim,dim_val,r):
    '''Gets probability for a given parent dimension'''
    filtered_data = [rec for rec in data if rec[p_dim] == dim_val]
    matching_children_measurments = get_dim_measurements(filtered_data,c_dim)
    rand_match_prob = random_match_prob(len(filtered_data),r)
    upward_adj = upward_adjustment(matching_children_measurments['positive'],matching_children_measurments['negative'])
    return rand_match_prob*upward_adj

def g(data,dimension,parent_dimensions=[]):
    '''Calculates the parent probability'''
    num_unique_instatiations = 2
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

def k2(data,u):
    '''Find the most probable parents for a given dimension'''
    dimensions = [dim for dim in data[0]]
    dim_parents = {}
    for dim in dimensions:
        pi_i = []
        p_old = g(data,dim)
        ok = True
        while ok and len(pi_i) < u:
            potential_parents = [p_dim for p_dim in dimensions if p_dim != dim and p_dim not in pi_i]
            new_p_vals = {p_dim: g(data,dim,pi_i + [p_dim]) for p_dim in potential_parents}
            p_new,p_new_val = max(new_p_vals.iteritems(),key=itemgetter(1))
            if p_new_val > p_old:
                p_old = p_new_val
                pi_i += [p_new]
            else:
                ok = False
        dim_parents[dim] = pi_i
    return dim_parents
