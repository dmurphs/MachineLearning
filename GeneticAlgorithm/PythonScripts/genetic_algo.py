import numpy as np
import pandas as pd
import random
from operator import itemgetter

def get_random_bit_vector():
    return np.random.choice([True,False],size=(num_items,))

def get_fitness(sequence,data,max_weight):
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

def split_n_most_fit(n,pop_with_fitness):
    sorted_pop_with_fitness = sorted(pop_with_fitness,key=itemgetter(1),reverse=True)
    print sorted_pop_with_fitness
    n_most_fit = sorted_pop_with_fitness[:n]
    candidates = sorted_pop_with_fitness[n:]
    print n_most_fit
    print candidates
    return n_most_fit,candidates

def roulette_selection(selection_candidates):
    prob = random.random()
    num_candidates = len(selection_candidates)
    candidates_with_lower_prob = [sc[0] for sc in selection_candidates if sc[1] < prob]
    if len(candidates_with_lower_prob) > 0:
        return candidates_with_lower_prob[-1]
    else:
        return selection_candidates[0][0]

def get_survivors(population,item_data,top_n_proportion,max_weight):
    num_most_fit = int(len(population)*top_n_proportion)
    pop_with_fitness = [(seq,get_fitness(seq,item_data,max_weight)) for seq in population]

item_data = pd.read_csv('../Data/items.csv')
num_items = len(item_data)
population_size = 200
max_weight = 200
top_n_proportion = 0.1

population = [get_random_bit_vector() for i in range(population_size)]
pop_with_fitness = [(seq,get_fitness(seq,item_data,max_weight)) for seq in population]
split_n_most_fit(10,pop_with_fitness)

survivors = get_survivors(population,item_data,n_most_fit_to_keep,n_survivors_per_gen)
