import pandas as pd
from collections import Counter
from helpers import *
from operator import itemgetter
import numpy as np
from argparse import ArgumentParser

# Get 'nbins' argument from command line and store it
parser = ArgumentParser()
program_desc = 'Perform Naive Bayes Algorithm on Dataset'
parser = ArgumentParser(description=program_desc)
parser.add_argument('nbins', type=int, help='Number of bins for Naive Bayes Algorithm')
args = parser.parse_args()
nbins = args.nbins

if nbins <= 0:
    print 'Invalid Argument'

else:
    # Read in data
    training_df = pd.read_csv('../Data/fruit.csv')
    test_df = pd.read_csv('../Data/testFruit.csv')

    #Get some precursor information about data
    classes = set(training_df['Class'])
    metric_cols = [col for col in training_df.columns if training_df[col].dtype == float]
    total_training_records = len(training_df)
    class_totals = {c:len(training_df[training_df['Class'] == c]) for c in classes}

    # Training stage here, create distributions for each attribute in each class
    class_attribute_distributions = create_distributions(classes,training_df,metric_cols,nbins)

    num_correct = 0
    test_data_rows = [row[1] for row in test_df.iterrows()]

    for record in test_data_rows:
        #this 'class_probs' variable will be the data cube
        class_probs = {}
        for c in classes:
            class_probs[c] = {}
            attribute_distributions = class_attribute_distributions[c]
            for attr in attribute_distributions:
                 distribution = attribute_distributions[attr]
                 print distribution
                 record_attr_val = record[attr]
                 record_attr_val_count = get_dist_frequency(distribution,record_attr_val)
                 num_in_attr_bin = get_num_with_attr_val(attr,record_attr_val,class_attribute_distributions)

                 #use m-estimator to handle 0 probabilities
                 pr_class_given_attr = (record_attr_val_count+1)/float(num_in_attr_bin + 1000)

                 class_probs[c][attr] = pr_class_given_attr

        # perform multiplication on probabilities and pick the highest one as class
        class_total_probs = []
        for c in class_probs:
            attr_probs = class_probs[c]
            mult = lambda x,y: x*y
            prob_product = reduce(mult,[attr_probs[p] for p in attr_probs])
            class_total_probs.append((c,prob_product))
        vote = max(class_total_probs,key=itemgetter(1))[0]

        if vote == record.Class:
            num_correct += 1

    total = len(test_data_rows)
    print '%i correct out of %i' %(num_correct,total)
    print '%.2f percent correct' %(num_correct/float(total)*100)
