Daniel Murphy
Spring 2016
CSCI 547
Naive Bayes Implementation (Using Raw Count Method)
Grad Student

USAGE:
  run: 'pip install -r requirements.txt'
  from PythonScripts directory, run: 'python naive_bayes.py [nbins]'
  example: run 'python naive_bayes.py 10' for 10 bins

APPROACH:
  I first created a function that will create a dictionary to hold the bins and the associated counts in each bin.
  I then began creating distributions from the training data.  My strategy involved using nested dictionaries
  in order to store the distributions for each attribute in each class.  The structure of the dictionary is
  as follows: {class_name: {attribute: {bin: count}}}  I did this in such a way that the code is not too
  dependent on the data.  Once I had the creation of distributions working correctly, i began writing code
  to classify a test case.  I started by just testing one record in the test data where a probability was
  calculated for each class given the value of the attribute.  My 'Data Cube' is a dictionary of dictionaries
  with the following structure: {class: {attr: probability}}.  I then did a multiplication among probabilities
  for attributes in each class and picked the highest one for the classification of the test case. Once I had this
  working I began using the entire test data set to classify. I wasn't using an m-estimator at first and was getting
  ok results with a low number of bins but for higher number of bins I was getting pretty low accuracy.
  I implemented an m-estimator by adding 1 to the numerator and 1000 to the denominator in the probability estimate.
  After this I began getting very good results.  In order to handle test values that are outside of the bin, I
  had a default value of 0 for the frequency.

POSSIBLE IMPROVEMENTS:
  I could attempt to fit each discrete set of bins into a continuous distribution.

ACCURACY:
  10 bins: 96%
  20 bins: 98%
  50 bins: 93%
  100 bins: 93%
