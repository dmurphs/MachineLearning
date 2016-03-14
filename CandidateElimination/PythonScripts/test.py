from csv import DictReader
from helpers import generalize_S,prune_G,specialize_G

S = {'Origin': 'Japan', 'Manufacturer': 'Honda', 'Color': 'Blue', 'Year': '1980', 'Type': 'Economy'}
test_record = {'Origin': 'Japan', 'Manufacturer': 'Toyota', 'Color': 'Blue', 'Year': '1990', 'Type': 'Economy'}
expected_output = {'Origin': 'Japan', 'Color': 'Blue', 'Manufacturer': True, 'Type': 'Economy', 'Year': True}

print 'Testing generalize_S'
print generalize_S(test_record,S)
print generalize_S(test_record,S) == expected_output
print '\n'


G = [{'Origin': True, 'Manufacturer': 'Honda', 'Color': True, 'Year': True, 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': 'Blue', 'Year': True, 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': '1980', 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': True, 'Type': 'Economy'}]
test_record = {'Origin': 'Japan', 'Manufacturer': 'Toyota', 'Color': 'Blue', 'Year': '1990', 'Type': 'Economy'}
expected_output = [{'Origin': True, 'Color': 'Blue', 'Year': True, 'Type': True, 'Manufacturer': True},
                    {'Origin': True, 'Color': True, 'Year': True, 'Type': 'Economy', 'Manufacturer': True}]

print 'Testing prune_G'
print prune_G(test_record,G) == expected_output
print '\n'

G = [{'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': True, 'Type': True}]
S = {'Origin': 'Japan', 'Manufacturer': 'Honda', 'Color': 'Blue', 'Year': '1980', 'Type': 'Economy'}
test_record = {'Origin': 'Japan', 'Manufacturer': 'Toyota', 'Color': 'Green', 'Year': '1970', 'Type': 'Sports'}
expected_output = [{'Origin': True, 'Manufacturer': 'Honda', 'Color': True, 'Year': True, 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': 'Blue', 'Year': True, 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': '1980', 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': True, 'Type': 'Economy'}]


print 'Testing specialize_G'
print sorted(specialize_G(test_record,G,S)) == sorted(expected_output)

G = [{'Origin': True, 'Manufacturer': True, 'Color': 'Blue', 'Year': True, 'Type': True},
    {'Origin': True, 'Manufacturer': True, 'Color': True, 'Year': True, 'Type': 'Economy'}]
S = {'Origin': 'Japan', 'Manufacturer': True, 'Color': 'Blue', 'Year': True, 'Type': 'Economy'}
test_record = {'Origin': 'USA', 'Manufacturer': 'Chrysler', 'Color': 'Red', 'Year': '1980', 'Type': 'Economy'}
[{'Origin': 'Japan', 'Color': True, 'Manufacturer': True, 'Type': True, 'Year': True},
{'Origin': True, 'Color': 'Blue', 'Year': True, 'Type': True, 'Manufacturer': True}]
expected_output = [{'Origin': True, 'Manufacturer': True, 'Color': 'Blue', 'Year': True, 'Type': True},
    {'Origin': 'Japan', 'Manufacturer': True, 'Color': True, 'Year': True, 'Type': 'Economy'}]

print 'Testing specialize_G again'
print sorted(specialize_G(test_record,G,S)) == sorted(expected_output)
