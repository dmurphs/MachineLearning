from math import factorial as fac

num_dim_values = 2
positive = 1
negative = 0

def get_dim_measurements(records,dimension):
    num_child_pos = len([rec for rec in records if rec[dimension] == positive])
    num_child_neg = len([rec for rec in records if rec[dimension] == negative])
    return {'positive': num_child_pos, 'negative': num_child_neg}

def g(data,dimension,parent_dimensions=[]):
    '''Calculates the parent probability'''

    random_match_prob = lambda x: fac(num_dim_values - 1)/float(fac(x + num_dim_values - 1))
    upward_adjustment = lambda x: fac(x['positive'])*fac(x['negative'])

    if len(parent_dimensions) == 0:
        measurements = get_dim_measurements(data,dimension)
        return random_match_prob(len(data))*upward_adjustment(measurements)

    else:
        parent_child_match = lambda record,desired_val: record[dimension] == desired_val and record[parent_dimension] == desired_val

        positive_parents = [rec for rec in data if rec[parent_dimension] == positive]
        parent_positive = get_dim_measurements(positive_parents,dimension)

        negative_parents = [rec for rec in data if rec[parent_dimension] == negative]
        parent_negative = get_dim_measurements(negative_parents,dimension)


        p1 = random_match_prob(len(positive_parents))*upward_adjustment(parent_positive)
        p2 = random_match_prob(len(negative_parents))*upward_adjustment(parent_negative)

        return p1*p2
