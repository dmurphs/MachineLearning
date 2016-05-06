import numpy as np
import pandas as pd
import random
from operator import itemgetter

def get_random_bit_vector():
    return np.random.choice([True,False],size=(num_items,))

def get_fitness(sequence,data):
    positive_indices = [i for i in range(len(sequence)) if sequence[i]]
    positive_values = data.iloc[positive_indices]
    total_weight = positive_values['Weight'].sum()
    total_price = positive_values['Price'].sum()
    if total_weight <= max_weight:
        return total_weight
    else:
        amount_over_weight = total_weight - max_weight
        scale = 1/float(amount_over_weight + 1)
        return max_weight*scale

def get_n_most_fit_indices(n,fitnesses):
    fitnesses_with_indices = [(i,fitnesses[i]) for i in range(len(fitnesses))]
    top_n = sorted(fitnesses_with_indices,key=itemgetter(1),reverse=True)[:n]
    return [i for i,seq in top_n]

def roulette_selection(selection_candidates):
    prob = random.random()
    num_candidates = len(selection_candidates)
    candidates_with_lower_prob = [sc[0] for sc in selection_candidates if sc[1] < prob]
    if len(candidates_with_lower_prob) > 0:
        return candidates_with_lower_prob[-1]
    else:
        return selection_candidates[0][0]


item_data = pd.read_csv('../Data/items.csv')
num_items = len(item_data)
population_size = 200
max_weight = 200
num_gen_parents = 20
n_most_fit_to_keep = 20
n_survivors_per_gen = 100

population = [get_random_bit_vector() for i in range(population_size)]

survivors = get_survivors(population,item_data,n_most_fit_to_keep,n_survivors_per_gen)
