from math import factorial as fac

def get_dim_measurements(records,dimension):
    num_child_pos = len([rec for rec in records if rec[dimension] == 1])
    num_child_neg = len([rec for rec in records if rec[dimension] == 0])
    return {'positive': num_child_pos, 'negative': num_child_neg}

def random_match_prob(N,r):
    return fac(r-1)/float(fac(N+r-1))

def upward_adjustment(num_pos,num_neg):
    return fac(num_pos)*fac(num_neg)

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
        all_products = [];
        print parent_dimensions
        for p_dim in parent_dimensions:
            positive_data = [rec for rec in data if rec[p_dim] == 1]
            positive_children_measurments = get_dim_measurements(positive_data,dimension)
            pos_rand_match_prob = random_match_prob(len(positive_data),r)
            pos_upward_adj = upward_adjustment(positive_children_measurments['positive'],positive_children_measurments['negative'])
            pos_prob = pos_rand_match_prob*pos_upward_adj

            negative_data = [rec for rec in data if rec[p_dim] == 0]
            negative_children_measurments = get_dim_measurements(negative_data,dimension)
            neg_rand_match_prob = random_match_prob(len(negative_data),r)
            neg_upward_adj = upward_adjustment(negative_children_measurments['positive'],negative_children_measurments['negative'])
            neg_prob = neg_rand_match_prob*neg_upward_adj

            all_products.append(pos_prob*neg_prob)
        return reduce(lambda x,y: x*y,all_products)
