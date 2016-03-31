from helpers import g,k2
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

test_df = pd.DataFrame(test_records)

print 'Expect about .005555'
print g(test_df,'x3',['x2'])
print '\n'

print 'Expect about .001111'
print g(test_df,'x2',['x1'])
print '\n'

print 'Expect about .0003607'
print g(test_df,'x1')
print '\n'

print k2(test_df,3)
