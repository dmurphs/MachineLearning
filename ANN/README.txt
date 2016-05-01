Daniel Murphy
Grad Student

Run `pip install -r requirements.txt` to install requirements.
Run `python ann.py` from PythonScripts directory to run program.

APPROACH:
  I wrote the code for ANN classifier using matrices to hold the values of the test data and parameters.
  My first steps were creating representations of each connection weight using matrices and checking that
  forward calculations through the network worked.  Once I had this functionality in place I began working
  on the backpropogation functionality.  I started by using only one hidden layer and performing
  gradient updates.  I then introduced the momentum term for updating parameters.  Once I had all of this
  functionality in place I began to fine tune the learning rate and momentum parameters until I was
  achieving optimal results.  I began to translate the 1 hidden layer model into numerous hidden layers.
