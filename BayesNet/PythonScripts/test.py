from helpers import g

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

print 'Expect about .005555'
print g(test_records,'x3',['x2'])
print '\n'

print 'Expect about .0003607'
print g(test_records,'x1')
