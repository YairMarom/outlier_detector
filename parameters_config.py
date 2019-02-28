import numpy as np

class ParameterConfig:

    def __init__(self):
        # experiment  parameters
        header_indexes = [4]
        self.dim = len(header_indexes)
        self.points_number = 500000
        self.sample_sizes = [100]  # [100,200,500,700,1000,2000,3000,4000,5000,20000]
        self.inner_iterations = 5
        self.ground_truth_mean = np.array([[0.025, 0.025, 0.025]])
        self.centers_number = 5
        self.outliers_number = 6
        self.input_points_file_name = 'datasets/HIGGS/HIGGS_sub.csv'
        self.outliers_trashold_value = 300

        # coreset parameters
        self.median_sample_size = 1
        self.closest_to_median_rate = 0.5
        self.number_of_remains_multiply_factor = 2
        self.max_sensitivity_multiply_factor = 2

        # iterations
        self.RANSAC_iterations = 1
        self.RANSAC_EM_ITERATIONS = 10
        self.RANSAC_loops_number = 10
        self.coreset_iterations = 1
        self.ground_truth_iterations = (np.asarray(self.sample_sizes) * 10).tolist()



