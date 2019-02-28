"""
    @staticmethod
    def run_corset_and_RANSAC_offline(P, k, m, coreset_iterations, RANSAC_iterations):

        In this experiment, we get a set P consist of weighted points with outliers, sample randomly a set S_1
        of m points from P, and sample a coreset S_2 consist of m points from P. Run the the recursive median on S_1 to
        find a median q_1, and run the recursive median on S_2 to find a median q_2, and compare the robust cost from q_1
        to P (ignoring the k farthest points) versus the robust cost from q_2 to P (ignoring the k farthest points).
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            m (int) : size of sample
        Returns:
            [float, float, float] : the coreset total cost, the RANSAC total cost, and the ground truth total cost



        assert P.get_size() != 0, "Q size is zero"
        assert k >= 0, "k is negative"

        coreset = CoresetForWeightedCenters()
        for i in range(coreset_iterations):
            if i == 0:
                C = coreset.coreset(P, k + 1, m)
                sum_of_weights = np.sum(C.weights)
                [C_median, C_outliers] = ExperimentsWeightedCenters.EM_estimator_k_means_robust(C, k)
                min_C_median_cost = P.get_cost_to_point_without_outliers(C_median, C_outliers)
                continue
            C = coreset.coreset(P, k + 1, m)
            sum_of_weights = np.sum(C.weights)
            [C_median, C_outliers] = ExperimentsWeightedCenters.EM_estimator_k_means_robust(C, k)
            current_C_median_cost = P.get_cost_to_point_without_outliers(C_median, C_outliers)
            if min_C_median_cost > current_C_median_cost:
                min_C_median_cost = current_C_median_cost
        C_median_cost = min_C_median_cost

        for i in range(RANSAC_iterations):
            if i == 0:
                random_sample = P.get_sample_of_points(m)
                [random_sample_median, random_sample_outliers] = ExperimentsWeightedCenters.EM_estimator_k_means_robust(random_sample, k, EM_iterations=parameters_config.RANSAC_EM_ITERATIONS)
                min_cost = P.get_cost_to_point_without_outliers(random_sample_median, random_sample_outliers)
                continue
            random_sample = P.get_sample_of_points(m)
            [random_sample_median, random_sample_outliers] = ExperimentsWeightedCenters.EM_estimator_k_means_robust(random_sample, k, EM_iterations=parameters_config.RANSAC_EM_ITERATIONS)
            current_sample_median_cost = P.get_cost_to_point_without_outliers(random_sample_median,random_sample_outliers)
            if min_cost > current_sample_median_cost:
                min_cost = current_sample_median_cost
        random_sample_median_cost = min_cost

        P_without_outliers = P.get_points_at_indices(0,P.get_size() - k)  # this brings us the points withput outlirs since we built P that way the outlirs in the end of the list
        ground_truth_median = np.array([0, 0])
        ground_truth_cost = P_without_outliers.get_sum_of_distances_to_point(ground_truth_median)

        return [C_median_cost, random_sample_median_cost, ground_truth_cost]

    ######################################################################
    """

"""
    def stream_old(self):

        The method start to get in a streaming points from the file st as required


        batch_size = self.sample_size*2
        current_batch = []
        dim = parameters_config.dim
        total = 0
        with open(self.file_name, 'rt') as csvfile:
            spamreader = csv.reader(csvfile)
            i = 0
            for row in spamreader:
                if row == []:
                    continue
                if total % int(self.points_number/10) == 0:
                    print("Points read so far: ", total)
                    #sum_of_weights = 0
                    #for t in range(len(self.stack)):
                    #    sum_of_weights += np.sum(self.stack[t].points.weights)
                    #print("Sum of weights so far: ", sum_of_weights)
                    #print(" ")
                if total == self.points_number:
                    break
                row_arr = np.asarray(row)
                current_batch.append(row_arr)
                i += 1
                if i % int(batch_size) == 0:
                    current_batch = [[float(entry) for entry in point] for point in current_batch]
                    current_batch = np.array(current_batch)
                    P = SetOfPoints(current_batch)
                    self.add_to_tree(P)
                    current_batch = []
                    i=0
                total += 1
            if  i > 0:
                current_batch = [[float(entry) for entry in point] for point in current_batch]
                current_batch = np.array(current_batch)
                P = SetOfPoints(current_batch)
                self.add_to_tree(P)

        while len(self.stack) > 1:
            node1 = self.stack.pop()
            node2 = self.stack.pop()
            new_node = self.merge_two_nodes(node1, node2)
            self.stack.append(new_node)
        C = self.stack[0].points
        print("coreset sum of weights: ", np.sum(C.weights))
        return C
    """



"""
flag = False
if flag:
    l1 = np.arange(0,250).reshape(-1,1)
    l2 = np.arange(250,500).reshape(-1,1)
    P1 = SetOfPoints(l1)
    P2 = SetOfPoints(l2)
    print("np.sum(P1.weights): ", np.sum(P1.weights))
    print("np.sum(P2.weights): ", np.sum(P2.weights))
    P3 = SetOfPoints()
    P3.add_set_of_points(P1)
    P3.add_set_of_points(P2)
    print("np.sum(P3.weights): ", np.sum(P3.weights))
    C = CoresetForWeightedCenters().coreset(P3, 2, 250)
    print("np.sum(C.weights): ", np.sum(C.weights))

    l1_2 = np.arange(500,750).reshape(-1,1)
    l2_2 = np.arange(750,1000).reshape(-1,1)
    P1_2 = SetOfPoints(l1_2)
    P2_2 = SetOfPoints(l2_2)
    print("np.sum(P1_2.weights): ", np.sum(P1_2.weights))
    print("np.sum(P2_2.weights): ", np.sum(P2_2.weights))
    P3_2 = SetOfPoints()
    P3_2.add_set_of_points(P1)
    P3_2.add_set_of_points(P2)
    print("np.sum(P3_2.weights): ", np.sum(P3_2.weights))
    C_2 = CoresetForWeightedCenters().coreset(P3_2, 2, 250)
    print("np.sum(C_2.weights): ", np.sum(C_2.weights))

    P_final = SetOfPoints()
    P_final.add_set_of_points(C)
    P_final.add_set_of_points(C_2)
    print("np.sum(P_final.weights): ", np.sum(P_final.weights))
    C_final = CoresetForWeightedCenters().coreset(P_final, 2, 250)
    print("np.sum(C_final.weights): ", np.sum(C_final.weights))
"""


#P1 = SetOfPoints([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
#P2 = SetOfPoints([[0,0,0],[1,1,1]])
#ans = P1.get_sum_of_distances_to_set_of_points(P2)


"""
    def coreset_return_sensitivities(self, P, k, m, sample_size_rate=parameters_config.median_sample_size, closest_rate1=parameters_config.median_closest_rate, closest_rate2=parameters_config.recursive_median_closest_to_median_rate):

        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            sample_size_rate (float) : parameter for the recursive median
            closest_rate1 (float) : parameter for the recursive median
            closest_rate2 (float) : parameter for the recursive median

        Returns:
            SetOfPoints: the coreset of P for k weighted centers. See Alg. 2 in the paper;


        assert k > 0, "k is not a positive integer"
        assert m > 0, "m is not a positive integer"
        assert P.get_size() != 0, "Q size is zero"
        minimum_number_of_points_in_iteration = int(math.log(P.get_size()))
        Q = copy.deepcopy(P)
        temp_set = SetOfPoints()
        max_sensitivity = -1
        while True:
            [q_k, Q_k] = self.recursive_robust_median(Q, k, sample_size_rate, closest_rate1, closest_rate2) #get the recursive median q_k and its closest points Q_k
            Q_k_size = Q_k.get_size()
            if Q_k_size == 0:
                x=2
            Q_k.set_sensitivities(k) # sets all the sensitivities in Q_k as described in line 5 in main alg.
            current_sensitivity = Q_k.get_arbitrary_sensitivity()
            if current_sensitivity > max_sensitivity:
                max_sensitivity = current_sensitivity #we save the maximum sensitivity in order to give the highest sensitivity to the points that remains in Q after this loop ends
            temp_set.add_set_of_points(Q_k) #since we remove Q_k from Q each time, we still wan to save every thing in order to go over the entire points after this loop ends and select from them and etc., so we save everything in temp_set
            x = temp_set.get_sum_of_weights()
            Q.remove_from_set(Q_k)
            size = Q.get_size()
            Q_k_weigted_size = Q_k.get_sum_of_weights()
            if size <= minimum_number_of_points_in_iteration or Q_k_weigted_size == 0: # stop conditions
                break
        if size != 0:
            Q.set_all_sensitivities(max_sensitivity * 2) # here we set the sensitivities of the points who left to the highest - since they are outliers with a very high probability
            temp_set.add_set_of_points(Q) #and now temp_set is all the points we began woth - just with updated sensitivities
        x = temp_set.get_sum_of_weights()
        T = temp_set.get_sum_of_sensitivities()
        temp_set.set_weights(T, m)
        return temp_set.sensitivities, temp_set.weights

        """

"""
######################################################################

@staticmethod
def error_vs_coreset_size_offline():
The
method
generate
a
set
of
points(synthetic or real
data
from a file, see

generate_weighted_points
function
and in each
iteration, a
sample(increasing
from iteration to

iteration) was
taken
by
coreset and by
the
competitor
RANSAC, and the
performance
for that sample was measured.The performance of the
algorithms is being tested at their level of error in relation to the overall price function that we want to
achieve to a minimum.

Returns:
    ~

    k = ParameterConfig.k
    m = ParameterConfig.sample_size_starting
    points_number = ParameterConfig.points_number
    coreset_iterations = ParameterConfig.coreset_iterations
    RANSAC_iterations = ParameterConfig.RANSAC_iterations
    C_error_totals_final = []  # coreset total average error at each iteration
    random_sample_error_totals_final = []  # RANSAC total average error at each iteration
    ground_truth_cost_total = []  # the ground truth error - which is 1 at each iteration
    sample_sizes = []
    points_numbers = []
    [P, weights] = ExperimentsWeightedCenters.generate_weighted_points(points_number)
    P = SetOfPoints(P, weights)
    iterations = ParameterConfig.iterations
    inner_iterations = ParameterConfig.inner_iterations
    for u in range(iterations):
        m += ParameterConfig.coreset_size_jumps
        print("iteration number ", u)
        C_error_total = []
        random_sample_error_total = []
        for t in range(inner_iterations):
            print("inner iteration number ", t)
            [C_cost, random_sample_cost, ground_truth_cost] = ExperimentsWeightedCenters.run_corset_and_RANSAC_offline(P, k, m, coreset_iterations, RANSAC_iterations)
            C_error = C_cost / ground_truth_cost
            random_sample_error = random_sample_cost / ground_truth_cost
            C_error_total.append(C_error)
            random_sample_error_total.append(random_sample_error)

        # avgs
        C_error_total_avg = sum(C_error_total) / len(C_error_total)
        random_sample_error_total_avg = sum(random_sample_error_total) / len(random_sample_error_total)
        print("coreset error in this iteration: ", C_error_total_avg)
        print("random sample error in this iteration: ", random_sample_error_total_avg)
        # fill final arrays
        C_error_totals_final.append(C_error_total_avg)
        random_sample_error_totals_final.append(random_sample_error_total_avg)
        ground_truth_cost_total.append(1)
        sample_sizes.append(m)
        points_numbers.append(points_number)

    # information printing
    print("points number: ", points_numbers)
    print("sample size: ", sample_sizes)
    print("coreset error: ", C_error_totals_final)
    print("ransac error: ", random_sample_error_totals_final)
    print("coreset error variance: ", np.var(C_error_totals_final))
    print("ransac error variance: ", np.var(random_sample_error_totals_final))

    plt.figure(1)
    plt.plot(sample_sizes, C_error_totals_final, 'b', sample_sizes, random_sample_error_totals_final, 'r',
             sample_sizes, ground_truth_cost_total, 'y')
    plt.ylabel('error')
    plt.xlabel('#sample size')
    plt.show()
"""