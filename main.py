#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################

from experiments_weighted_centers import ExperimentsWeightedCenters
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

#ExperimentsWeightedCenters.error_vs_coreset_size_streaming_2(experiment_type='synthetic')
#ExperimentsWeightedCenters.error_vs_coreset_size_streaming(experiment_type='dataset_1')
#ExperimentsWeightedCenters.error_vs_coreset_size_streaming(experiment_type='dataset_2')
ExperimentsWeightedCenters.error_vs_coreset_size_streaming(experiment_type='dataset_3')







