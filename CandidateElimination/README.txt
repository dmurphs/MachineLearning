Daniel Murphy
CSCI 547
Grad Student

USAGE:
  to run classification on test data, run 'python cross_validation.py' from PythonScripts directory.

VERSION SPACE DETAILS:
  The maximally general and maximally specific hypotheses are represented by boolean values True and False respectively
  Python's dictionary object was used to store hypotheses in the version space (i.e. {'AirTemp': 'Warm', 'Sky': True, 'Forecast': True, 'Water': True, 'Humidity': 'Normal', 'Wind': 'Weak'})
  G is stored as a list of dictionaries while S is a single dictionary.

APPROACH
  I began by working on core functionality such as pruning G to fit a positive record, generalizing S to fit a positive record, and specializing G to exclude a negative record.
  I used test data with known outputs to test that each of these functions worked correctly.  Once I had these working I began putting them together into a candidate elimination
  classification algorithm.  I had some trouble with dealing with maximally specific values at first when dealing with specializing G against negative records.  I took care of this
  by writing a function that detects if there is a false value for an attribute in S and then gets more specific hypotheses based on all possible values for that attribute not
  counting the value in the current training record.  I then wrote a function to get all hypotheses between G and S for when it completes running through the training records
  which allowed my estimate to attain the greatest accuracy.

RESULTS
  99.8% (up to 100%!) accuracy with 10 fold cross validation
  Confusion Matrix:  {'Enjoy Sport': {'Enjoy Sport': 477, 'Do Not Enjoy': 2}, 'Do Not Enjoy': {'Enjoy Sport': 0, 'Do Not Enjoy': 521}}
  The structure for the confusion matrix is: {actual_class: {predicted_class: count}}. In the example above there are 477 true positives,
  2 false negatives, 0 false positives, and 521 true negatives
