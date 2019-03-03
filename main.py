#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################
from outlier_detector import OutliersDetector
import numpy as np

your_data = np.random.rand(2000,5)
C_cost, coreset_total_time, C_outliers, coreset_means, data_without_outliers, C = OutliersDetector.detect(your_data)
data_without_outliers_points = data_without_outliers.points
coreset = C.points







