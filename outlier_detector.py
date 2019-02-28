#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################


from __future__ import division

import csv
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from coreset_streamer import CoresetStreamer
from set_of_points import SetOfPoints
from parameters_config import ParameterConfig



class OutliersDetector:
    parameters_config = None
    current_iteration = None
    @staticmethod
    def EM_estimator_k_means_robust(P, is_ground_truth = True, EM_iterations = np.nan):
        parameters_config = OutliersDetector.parameters_config
        centers_number = parameters_config.centers_number
        outliers_number = parameters_config.outliers_number
        if is_ground_truth:
            EM_iterations = P.get_size() * 100
        size = P.get_size()
        k_centers = P.get_sample_of_points(centers_number)
        closest = P.get_closest_points_to_set_of_points(k_centers, size - outliers_number, type="by number")
        min_cost = closest.get_sum_of_distances_to_set_of_points(k_centers)
        min_k_centers = k_centers
        for i in range(EM_iterations):
            k_centers = P.get_sample_of_points(centers_number)
            closest = P.get_closest_points_to_set_of_points(k_centers, size - outliers_number, type="by number")
            current_cost = closest.get_sum_of_distances_to_set_of_points(k_centers)
            if current_cost < min_cost:
                min_cost = current_cost
                min_k_centers = k_centers
        outliers = P.get_farthest_points_to_set_of_points(min_k_centers, outliers_number, type="by number")
        return [min_k_centers, outliers]

    @staticmethod
    def RANSAC(P, sample_size):

        parameters_config = OutliersDetector.parameters_config
        outliers_number = parameters_config.outliers_number
        min_cost = np.infty
        P_size = P.get_size()
        RANSAC_steps = int(np.sqrt(sample_size))
        counter = 0
        while True:
            R = P.get_sample_of_points(sample_size)
            R.weights = R.weights.reshape(-1)
            [centers, outliers] = OutliersDetector.EM_estimator_k_means_robust(R, is_ground_truth=False, EM_iterations = 1)
            closest = P.get_closest_points_to_set_of_points(centers,P_size-outliers_number,"by number")
            current_cost = closest.get_sum_of_distances_to_set_of_points(centers)
            if min_cost > current_cost:
                min_cost = current_cost
                R_min = R
                counter = 0
            counter += 1
            if counter == RANSAC_steps:
                break

        return R_min

    @staticmethod
    def run_corset_2(P, sample_size):
        parameters_config = OutliersDetector.parameters_config
        points_number = parameters_config.points_number
        k = parameters_config.centers_number + parameters_config.outliers_number
        min_coreset_cost = np.infty
        coreset_total_time = 0
        coreset_starting_time = time.time()
        C = CoresetStreamer(sample_size=sample_size, points_number=points_number, k=k, parameters_config = parameters_config).stream(P)
        coreset_ending_time = time.time()
        coreset_total_time += coreset_ending_time - coreset_starting_time
        [coreset_means, coreset_outliers] = OutliersDetector.EM_estimator_k_means_robust(C)
        coreset_means.set_all_weights_to_specific_value(1.0)
        coreset_outliers.set_all_weights_to_specific_value(1.0)
        print("coreset_means.points: \n", coreset_means.points)
        print("coreset_outliers.points: \n", coreset_outliers.points)
        corset_cost = P.get_cost_to_center_without_outliers(coreset_means, coreset_outliers)

        RANSAC_total_time = 0
        RANSAC_starting_time = time.time()
        R = OutliersDetector.RANSAC(P, sample_size)
        RANSAC_ending_time = time.time()
        RANSAC_total_time += RANSAC_ending_time - RANSAC_starting_time
        R.weights = R.weights.reshape(-1)
        [random_sample_means, random_sample_outliers] = OutliersDetector.EM_estimator_k_means_robust(R)
        print("random_sample_means.points: \n", random_sample_means.points)
        print("random_sample_outliers.points: \n", random_sample_outliers.points)
        random_sample_cost = P.get_cost_to_center_without_outliers(random_sample_means, random_sample_outliers)
        return [corset_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, coreset_outliers.points, random_sample_outliers.points]

    @staticmethod
    def get_points_from_file(points_file_name):
        """
        TODO: complete
        :param points_file_name:
        :return:
        """
        parameters_config = OutliersDetector.parameters_config
        print("Start reading points from file..")
        points_number = parameters_config.points_number
        points = []
        with open(points_file_name, 'rt') as csvfile:
            spamreader = csv.reader(csvfile)
            i = 0
            for row in spamreader:
                if row == []:
                    continue
                row_arr = np.asarray(row)
                row_arr = row_arr[parameters_config.headers_indixes]
                points.append(row_arr)
                i += 1
                if i == points_number:
                    break
        print("Finish reading points from file.")
        points = [[float(entry) for entry in point] for point in points]
        points = np.array(points)
        P = SetOfPoints(points)
        return P

    @staticmethod
    def init_parameter_config():
        parameters_config = ParameterConfig()
        # main parameters
        parameters_config.points_number = 4000
        parameters_config.headers_indixes = [0,1,2,3]
        # experiment  parameters
        parameters_config.sample_sizes = [100]  # [2000, 4000, 7000, 10000] #[100,200,500,700,1000,2000,3000,4000,5000,20000]
        parameters_config.inner_iterations = 3
        parameters_config.centers_number = 5
        parameters_config.outliers_number = 3
        parameters_config.outliers_trashold_value = 300000
        # coreset parameters
        parameters_config.median_sample_size = 1
        parameters_config.closest_to_median_rate = 0.5
        parameters_config.number_of_remains_multiply_factor = 1
        parameters_config.max_sensitivity_multiply_factor = 2
        # iterations
        parameters_config.RANSAC_iterations = 1
        parameters_config.coreset_iterations = 1
        parameters_config.RANSAC_EM_ITERATIONS = 10
        parameters_config.RANSAC_loops_number = 10
        # files
        parameters_config.input_points_file_name = 'datasets/bengin_traffic.csv'
        OutliersDetector.parameters_config = parameters_config

    @staticmethod
    def error_vs_coreset_size_streaming():
        """
        In this experiment, we get an uncompleted matrix D with one randomly missing entry in each observation in D.
        We are running state of the art matrix completion algorithms on our coreset sample, and on a sample we get from
        RANSAC as well, and compare the solution accuracy versus the size of sample.
        :return:
        """
        # parameters
        OutliersDetector.init_parameter_config()
        parameters_config= OutliersDetector.parameters_config
        input_points_file_name = parameters_config.input_points_file_name
        points_number = parameters_config.points_number
        coreset_iterations = parameters_config.coreset_iterations
        RANSAC_iterations = parameters_config.RANSAC_iterations
        inner_iterations = parameters_config.inner_iterations
        #containers for statistics
        C_error_totals_final = []  # coreset total average error at each iteration
        C_error_totals_var = []  # coreset total variance for each sample
        random_sample_error_totals_final = []  # RANSAC total average error at each iteration
        random_sample_error_totals_var = []  # RANSAC total variance for each sample
        coreset_totals_time = []  # RANSAC total variance for each sample
        RANSAC_totals_time = []  # RANSAC total variance for each sample
        C_outliers_number_totals = []  # RANSAC total variance for each sample
        RANSAC_ouliers_number_totals = []  # RANSAC total variance for each sample
        sample_sizes = []
        points_numbers = []
        ground_truth_cost = 1

        P = OutliersDetector.get_points_from_file(input_points_file_name)
        for u in range(len(parameters_config.sample_sizes)):
            current_iteration = u
            sample_size = parameters_config.sample_sizes[u]
            print("iteration number ", u)
            C_error_total = []
            random_sample_error_total = []
            coreset_total_time_inner = []
            RANSAC_total_time_inner = []
            C_outliers_number_total = []
            RANSAC_ouliers_number_total = []
            for t in range(inner_iterations):
                print("inner iteration number ", t)
                [C_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, C_outliers, RANSAC_outliers] = OutliersDetector.run_corset_2(P=P, sample_size=sample_size)
                C_error = C_cost / ground_truth_cost
                random_sample_error = random_sample_cost / ground_truth_cost
                C_error_total.append(C_error)
                random_sample_error_total.append(random_sample_error)
                coreset_total_time_inner.append(coreset_total_time)
                RANSAC_total_time_inner.append(RANSAC_total_time)
                C_ouliers_number = len(C_outliers[C_outliers>parameters_config.outliers_trashold_value])
                RANSAC_ouliers_number = len(RANSAC_outliers[RANSAC_outliers>parameters_config.outliers_trashold_value])
                C_outliers_number_total.append(C_ouliers_number)
                RANSAC_ouliers_number_total.append(RANSAC_ouliers_number)
            # avgs
            C_error_total_avg = np.mean(C_error_total)
            C_error_total_var = np.var(C_error_total) / C_error_total_avg
            random_sample_error_total_avg = np.mean(random_sample_error_total)
            random_sample_error_total_var = np.var(random_sample_error_total) / random_sample_error_total_avg
            coreset_total_time_inner_avg = np.mean(coreset_total_time_inner)
            RANSAC_total_time_inner_avg = np.mean(RANSAC_total_time_inner)
            C_outliers_number_total_avg = np.mean(C_outliers_number_total)
            RANSAC_ouliers_number_total_avg = np.mean(RANSAC_ouliers_number_total)
            C_error_totals_final.append(C_error_total_avg)
            C_error_totals_var.append(C_error_total_var)
            random_sample_error_totals_final.append(random_sample_error_total_avg)
            random_sample_error_totals_var.append(random_sample_error_total_var)
            coreset_totals_time.append(coreset_total_time_inner_avg)
            RANSAC_totals_time.append(RANSAC_total_time_inner_avg)
            C_outliers_number_totals.append(C_outliers_number_total_avg)
            RANSAC_ouliers_number_totals.append(RANSAC_ouliers_number_total_avg)
            sample_sizes.append(sample_size)
            points_numbers.append(points_number)
            # information printing
            print("points_numbers = ", points_numbers)
            print("sample_sizes = ", sample_sizes)
            print("C_error_totals_final = ", C_error_totals_final)
            print("random_sample_error_totals_final = ", random_sample_error_totals_final)
            print("C_error_totals_var =  ", C_error_totals_var)
            print("random_sample_error_totals_var = ", random_sample_error_totals_var)
            print("coreset_totals_time = ", coreset_totals_time)
            print("RANSAC_totals_time = ", RANSAC_totals_time)
            print("coreset_to_RANSAC_time_rate = ",np.asarray(coreset_totals_time) / np.asarray(RANSAC_totals_time))
            print("C_outliers_number_totals = ",C_outliers_number_totals)
            print("RANSAC_ouliers_number_totals = ",RANSAC_ouliers_number_totals)
        fig = plt.figure(1)
        ax_errors = fig.add_subplot(111)
        linestyle = {"linestyle": "-", "linewidth": 1, "markeredgewidth": 1, "elinewidth": 1, "capsize": 4}
        ax_errors.errorbar(sample_sizes, C_error_totals_final, color='b', fmt='o', **linestyle)
        ax_errors.errorbar(sample_sizes, random_sample_error_totals_final, color='r', fmt='o', **linestyle)
        plt.ylabel('error')
        plt.xlabel('#sample size')
        plt.show()


    @staticmethod
    def run_corset(P, sample_size, RANSAC=False):
        parameters_config = OutliersDetector.parameters_config
        points_number = parameters_config.points_number
        k = parameters_config.centers_number + parameters_config.outliers_number
        min_coreset_cost = np.infty
        coreset_total_time = 0
        coreset_starting_time = time.time()
        C = CoresetStreamer(sample_size=sample_size, points_number=points_number, k=k,
                            parameters_config=parameters_config).stream(P)
        coreset_ending_time = time.time()
        coreset_total_time += coreset_ending_time - coreset_starting_time
        [coreset_means, coreset_outliers] = OutliersDetector.EM_estimator_k_means_robust(C)
        coreset_means.set_all_weights_to_specific_value(1.0)
        coreset_outliers.set_all_weights_to_specific_value(1.0)
        # print("coreset_means.points: \n", coreset_means.points)
        # print("coreset_outliers.points: \n", coreset_outliers.points)
        corset_cost = P.get_cost_to_center_without_outliers(coreset_means, coreset_outliers)

        random_sample_cost = RANSAC_total_time = random_sample_outliers = None
        if RANSAC:
            RANSAC_total_time = 0
            RANSAC_starting_time = time.time()
            R = OutliersDetector.RANSAC(P, sample_size)
            RANSAC_ending_time = time.time()
            RANSAC_total_time += RANSAC_ending_time - RANSAC_starting_time
            R.weights = R.weights.reshape(-1)
            [random_sample_means, random_sample_outliers] = OutliersDetector.EM_estimator_k_means_robust(R)
            print("random_sample_means.points: \n", random_sample_means.points)
            print("random_sample_outliers.points: \n", random_sample_outliers.points)
            random_sample_cost = P.get_cost_to_center_without_outliers(random_sample_means, random_sample_outliers)
            return [corset_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, coreset_outliers,
                    random_sample_outliers]
        return [corset_cost, 0, coreset_total_time, 0, coreset_outliers, np.asarray([]), coreset_means, np.asarray([])]

    @staticmethod
    def detect():
        OutliersDetector.init_parameter_config()
        parameters_config = OutliersDetector.parameters_config
        P = OutliersDetector.get_points_from_file(parameters_config.input_points_file_name)
        sample_size = parameters_config.sample_sizes[0]
        [C_cost, random_sample_cost, coreset_total_time, RANSAC_total_time, C_outliers, RANSAC_outliers,
         coreset_means, RANSAC_means] = OutliersDetector.run_corset(P=P, sample_size=sample_size)
        return C_cost, coreset_total_time, C_outliers, coreset_means


C_cost, coreset_total_time, C_outliers, coreset_means = OutliersDetector.detect()
print("C_cost: ", C_cost)
print("coreset_total_time: ", coreset_total_time)
print("C_outliers: ", C_outliers.points)
print("coreset_means: ", coreset_means.points)
