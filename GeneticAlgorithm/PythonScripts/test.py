from genetic_algo import get_fitness,next_generation
import pandas as pd

item_data = pd.read_csv('../Data/items.csv').iloc[[1,2,3,4,5]]
num_items = len(item_data)
population_size = 200

pop = [[True,True,False,True,False],[False,True,False,True,True]]
next_generation(pop,item_data)
