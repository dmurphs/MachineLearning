import numpy as np

def get_num_with_attr_val(attr,record_attr_val,class_attribute_distributions):
    '''Get the total number of records for an attribute that fall into same bin'''
    record_attr_val_count = 0
    for c in class_attribute_distributions:
        attribute_distribution = class_attribute_distributions[c][attr]
        for b in attribute_distribution:
            if record_attr_val >= b[0] and record_attr_val < b[1]:
                record_attr_val_count += attribute_distribution[b]

    return record_attr_val_count

def create_bins(training_df,metric_cols,nbins):
    '''Creates bins given data and a number of bins'''
    attr_bins = {}
    for col in metric_cols:
        col_vals = training_df[col]
        max_val = max(col_vals)
        min_val = min(col_vals)
        bin_ends = bin_ends = np.linspace(min(col_vals),max(col_vals),nbins+1)
        bins = [(bin_ends[i],bin_ends[i+1]) for i in range(len(bin_ends)-1)]
        attr_bins[col] = bins
    return attr_bins

def get_attr_distribution(attr_vals,attr_bins):
    '''Get counts for each bin given a list of attribute values and bins.'''
    field_distribution = {b:0 for b in attr_bins}
    for val in attr_vals:
        for b in attr_bins:
            if val >= b[0] and val < b[1]:
                field_distribution[b] += 1
        if val == attr_bins[-1][1]:
            field_distribution[attr_bins[-1]] += 1
    return field_distribution

def get_summary_stats(data_rows):
    '''Get Summary statistics to use for gaussian fit, returns mean,stdev'''
    mean = np.mean(data_rows)
    stdev = np.std(data_rows)
    return mean,stdev

def create_distributions(classes,training_df,metric_cols,nbins):
    '''Create distributions for classifier to use to place test data into bins'''
    class_attribute_distributions = {}
    bins = create_bins(training_df,metric_cols,nbins)
    for c in classes:
        class_records = training_df[training_df['Class'] == c]
        col_distributions = {}
        for col in metric_cols:
            col_vals = class_records[col]
            attr_bins = bins[col]
            col_distributions[col] = get_attr_distribution(col_vals,attr_bins)
        class_attribute_distributions[c] = col_distributions
    return class_attribute_distributions

def get_dist_frequency(distribution,attr_val):
    '''Returns frequency for a test value in a given distribution'''
    record_attr_val_count = 0
    for b in distribution:
        if attr_val >= b[0] and attr_val < b[1]:
            record_attr_val_count = distribution[b]
    return record_attr_val_count
