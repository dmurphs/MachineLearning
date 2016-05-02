Daniel Murphy
Grad Student

Run `pip install -r requirements.txt` to install requirements.
Run `python multilayer_ann.py` from PythonScripts directory to run program.

NOTE: if attempting to run error_plot.py, make sure to have matplotlib installed in your python environment
      as well as a backend

NOTE: Extra credit was completed (example output in network_parameters.txt)

APPROACH:
  I wrote the code for ANN classifier using matrices to hold the values of the test data and parameters.
  My first steps were creating representations of each connection weight using matrices and checking that
  forward calculations through the network worked.  Once I had this functionality in place I began working
  on the backpropogation functionality.  I started by using only one hidden layer and performing
  gradient updates.  I then introduced the momentum term for updating parameters.  Once I had all of this
  functionality in place I began to fine tune the learning rate and momentum parameters until I was
  achieving optimal results.  I began to translate the 1 hidden layer model into numerous hidden layers.
  Once I had the program working with multiple hidden layers I began to implement 10 fold cross validation
  and an error tracker to plot errors.

ISSUES:
  Using more than 1 hidden layer seems to yield unstable results.  This seems like it could have a lot to do
  with the momentum and learning rate parameters needing to be tuned differently for more layers.

RESULTS:

  Configuration: [3, 3, 3]
  Average Accuracy: 96.00 percent
  Confusion Matrix:
  [[ 50.   0.   0.]
  [  0.  46.   2.]
  [  0.   4.  48.]]

  Configuration: [4, 4, 3]
  Average Accuracy: 96.67 percent
  Confusion Matrix:
  [[ 50.   0.   0.]
   [  0.  45.   0.]
   [  0.   5.  50.]]

  Configuration: [3, 3, 3, 3]
  Average Accuracy: 89.33 percent
  Confusion Matrix:
  [[ 50.   0.   0.]
   [  0.  34.   0.]
   [  0.  16.  50.]]

  Configuration: [4, 12, 3]
  Average Accuracy: 96.67 percent
  Confusion Matrix:
  [[ 50.   0.   0.]
   [  0.  45.   0.]
   [  0.   5.  50.]]
