from helpers import g,k2,find_all_parent_val_combinations
import pandas as pd

test_records = [
    {'x1': 1, 'x2': 0, 'x3': 0},
    {'x1': 1, 'x2': 1, 'x3': 1},
    {'x1': 0, 'x2': 0, 'x3': 1},
    {'x1': 1, 'x2': 1, 'x3': 1},
    {'x1': 0, 'x2': 0, 'x3': 0},
    {'x1': 0, 'x2': 1, 'x3': 1},
    {'x1': 1, 'x2': 1, 'x3': 1},
    {'x1': 0, 'x2': 0, 'x3': 0},
    {'x1': 1, 'x2': 1, 'x3': 1},
    {'x1': 0, 'x2': 0, 'x3': 0}
]

columns = ['x1','x2','x3']

print 'Expect about .005555'
print g(test_records,'x3',['x2'])
print '\n'

print 'Expect about .001111'
print g(test_records,'x2',['x1'])
print '\n'

print 'Expect about .0003607'
print g(test_records,'x1')
print '\n'

print 'Checking k2 with order x1,x2,x3'
print k2(test_records,3,list(columns))
print '\n'

print 'Testing find all combinations'
print find_all_parent_val_combinations(len(columns),[0,1])
print '\n'

'''print 'Testing probability table construction'
print get_count_table(test_records,'x3',['x2'])
print '\n'

print 'Testing classification of test record'
print classify_record(test_records,{'x1': 0, 'x2': 0, 'x3': 1},{'x2': ['x1'], 'x3': ['x2'], 'x1': []},'x1',[0,1])'''
