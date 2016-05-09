import numpy as np
import pandas as pd
from operator import itemgetter
import random

def get_random_bit_vector(num_items):
    '''Returns a random bit vector (True/False) with given_number of items'''
    return np.random.choice([True,False],size=(num_items,))

def get_fitness(sequence,data,max_weight):
    '''Get the fitness of a bit vector sequence'''
    positive_indices = [i for i in range(len(sequence)) if sequence[i]]
    positive_values = data.iloc[positive_indices]
    total_weight = positive_values['Weight'].sum()
    total_price = positive_values['Price'].sum()
    if total_weight <= max_weight:
        return total_price
    else:
        return total_price*(max_weight/float(total_weight))

def split_n_most_fit(n,pop_with_fitness):
    '''Split up n most fit in population and remaining population'''
    sorted_pop_with_fitness = sorted(pop_with_fitness,key=itemgetter(1),reverse=True)
    most_fit = sorted_pop_with_fitness[:n]
    candidates = sorted_pop_with_fitness[n:]
    return most_fit,candidates
    
def select_candidate(cands_with_info):
    '''Return a candidate to survive and the remaining candidates'''
    rand_num = random.random()
    get_interval = lambda i: (cands_with_info[i][3],cands_with_info[i+1][3])
    intervals = [(0,cands_with_info[0][3])] + [get_interval(i) for i in range(len(cands_with_info)-1)]
    intervals[-1] = (intervals[-1][0],1.0)
    is_selected_interval = lambda interval: rand_num >= interval[0] and rand_num < interval[1]
    selected_index = [i for i in range(len(intervals)) if is_selected_interval(intervals[i])][0]
    remaining_candidates = cands_with_info[:selected_index] + cands_with_info[selected_index+1:]
    return cands_with_info[selected_index],remaining_candidates
    
def roullette_selection(cands_with_info,num_selections):
    '''Select given number candidates for survival'''
    selections = []
    remaining_cands = cands_with_info
    while len(selections) < num_selections:
        selected_cand,remaining_cands = select_candidate(remaining_cands)
        selections.append(selected_cand)
    return selections

def get_survivors(population,item_data,top_n_proportion,max_weight):
    '''Get survivors from roullette selection'''
    num_most_fit = int(len(population)*top_n_proportion)
    pop_with_fitness = [(seq,get_fitness(seq,item_data,max_weight)) for seq in population]
    most_fit,candidates = split_n_most_fit(num_most_fit,pop_with_fitness)

    sum_cand_fitness = sum([ft for seq,ft in candidates])
    cand_probs = [ft/float(sum_cand_fitness) for seq,ft in candidates]

    get_cumul_prob = lambda i: sum([cand_probs[j] for j in range(i+1)])
    cand_cumul_probs = [get_cumul_prob(i) for i in range(len(cand_probs))]
    
    cand_seqs = [c[0] for c in candidates]
    cand_fitnesses = [c[1] for c in candidates]     
     
    cands_with_info = zip(cand_seqs,cand_fitnesses,cand_probs,cand_cumul_probs)
    
    selected_members = roullette_selection(cands_with_info,100 - len(most_fit))
    return [mf[0] for mf in most_fit] + [sp[0] for sp in selected_members]
    
def get_parents(population,item_data,tournament_size,num_parents,max_weight):
    '''Select parents for generation by tournament selection'''
    selected = []
    #keep track of candidate indexes so selected individuals can be removed
    candidates = [(i,population[i]) for i in range(len(population))]
    while len(selected) < num_parents:
        tournament_participants = random.sample(candidates,tournament_size)
        participant_fitness = [(tp[0],tp[1],get_fitness(tp[1],item_data,max_weight)) for tp in tournament_participants]
        winner = sorted(participant_fitness,key=itemgetter(2),reverse=True)[0]
        selected.append(winner[1])
        winner_index = winner[0]
        candidates = [c for c in candidates if c[0] != winner_index]
    return selected
    
def reproduce(parents,num_children):
    '''Recombine parents to produce specified number of new children'''
    children = []
    for _ in range(num_children):
        couple = random.sample(parents,2)
        num_items = len(couple[0])
        child = np.concatenate((couple[0][:num_items/2],couple[1][num_items/2:]),axis=0)
        children.append(child)
    return children
    
def mutate(individual):
    '''Create mutations on an individual'''
    mutated_individual = individual
    mutation_index = np.random.geometric(1/float(len(mutated_individual)))
    while mutation_index < len(individual):
        mutated_individual[mutation_index] = not mutated_individual[mutation_index]
        mutation_index += np.random.geometric(1/float(len(mutated_individual)))
    return mutated_individual

item_data = pd.read_csv('../Data/items.csv')
num_items = len(item_data)
population_size = 100
max_weight = 200
top_n_proportion = 0.1
tournament_size = 30

population = [get_random_bit_vector(num_items) for i in range(population_size)]
fitnesses = [get_fitness(seq,item_data,max_weight) for seq in population]
avg_fitness = sum(fitnesses)/float(len(fitnesses))

average_fitnesses = [avg_fitness]
max_fitnesses = [max(fitnesses)]
for generation in range(500):
    if generation % 100 == 0:
        print generation
    parents = get_parents(population,item_data,tournament_size,50,max_weight)
    
    new_children = reproduce(parents,100)
    new_pop = population + new_children
    
    new_pop_with_mutations = [mutate(seq) for seq in new_pop]
    
    population = get_survivors(new_pop_with_mutations,item_data,top_n_proportion,max_weight)
    
    new_fitnesses = [get_fitness(seq,item_data,max_weight) for seq in population]
    avg_fitness = sum(new_fitnesses)/float(len(new_fitnesses))
    average_fitnesses.append(avg_fitness)
    max_fitnesses.append(max(new_fitnesses))