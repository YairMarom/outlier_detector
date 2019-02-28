#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################




from __future__ import division

import csv
import numpy as np
from coreset_node import CoresetNode
from coreset_for_weighted_centers import CoresetForWeightedCenters
from set_of_points import SetOfPoints
from parameters_config import ParameterConfig
import copy

"""
Class that performs the full streaming operation. Each step read m points from a file, comprase it and add it to the
coreset tree.
Attributes:
    stack (list): A list that simulates the streaming comprassion tree operations
    st (string): The path to the file
    m (int): size of chunk
    co (int): flag/number of points for read
    eps (float): error parameter
    delta (float): failure probability
"""


class CoresetStreamer:

    def __init__(self, sample_size, points_number, k, parameters_config):

        self.stack = []
        self.k = k
        self.file_name = parameters_config.input_points_file_name
        self.sample_size = sample_size
        self.points_number = points_number #if points_number == -1 then it will read until EOF
        self.parameters_config = parameters_config

    ######################################################################

    def stream(self, P):
        """
        The method start to get in a streaming points from the file st as required
        TODO: complete parameteres
        """
        points_number = self.points_number
        batch_size = self.sample_size*2
        current_batch = []
        starting_index = 0
        number_of_points_read_so_far = 0
        Q = copy.deepcopy(P)
        while True:
            if number_of_points_read_so_far == self.points_number:
                break
            if number_of_points_read_so_far % int(self.points_number / 10) == 0:
                print("Points read so far: ", number_of_points_read_so_far)
                #sum_of_weights = 0
                #for t in range(len(self.stack)):
                #    sum_of_weights += np.sum(self.stack[t].points.weights)
                #print("Sum of weights so far: ", sum_of_weights)
                #print(" ")
            if batch_size > Q.get_size():
                self.add_to_tree(Q)
                break
            current_batch = Q.get_points_at_indices(starting_index, starting_index+batch_size)
            Q.remove_points_at_indexes(starting_index, starting_index+batch_size)
            self.add_to_tree(current_batch)
            number_of_points_read_so_far += batch_size
        while len(self.stack) > 1:
            node1 = self.stack.pop()
            node2 = self.stack.pop()
            new_node = self.merge_two_nodes(node1, node2)
            self.stack.append(new_node)
        C = self.stack[0].points
        print("coreset sum of weights: ", np.sum(C.weights))
        return C

    ######################################################################

    def add_to_tree(self, P):
        if P.get_size() > self.sample_size:
            coreset = CoresetForWeightedCenters(self.parameters_config).coreset(P=P, k=self.k, m=self.sample_size)
            x = np.sum(coreset.weights)
            current_node = CoresetNode(coreset)
        else:
            current_node = CoresetNode(P)

        if len(self.stack) == 0:
            self.stack.append(current_node)
            return

        stack_top_node = self.stack[-1]
        if stack_top_node.rank != current_node.rank:
            self.stack.append(current_node)
            return
        else:
            while stack_top_node.rank == current_node.rank: #TODO: take care for the case they are not equal, currently the node deosn't appanded to the tree
                self.stack.pop()
                current_node_sum_of_weights = np.sum(current_node.points.weights)
                top_node_sum_of_weights = np.sum(stack_top_node.points.weights)
                current_node = self.merge_two_nodes(current_node, stack_top_node)
                current_node_sum_of_weights = np.sum(current_node.points.weights)
                if len(self.stack) == 0:
                    self.stack.append(current_node)
                    return
                stack_top_node = self.stack[-1]
                if stack_top_node.rank != current_node.rank:
                    self.stack.append(current_node)
                    return

    ######################################################################

    def merge_two_nodes(self, node1, node2):
        """
        The method gets two nodes of the corset tree, merge them, and return the corset of the merged nodes
        :param current_node: CoresetNode
        :param stack_top_node: CoresetNode
        """
        P1 = node1.points
        L1_sum_of_weights = np.sum(P1.weights)
        P2 = node2.points
        L2_sum_of_weights = np.sum(P2.weights)
        P1.add_set_of_points(P2)
        coreset = CoresetForWeightedCenters(self.parameters_config).coreset(P=P1, k=self.k, m=self.sample_size)
        coreset_sum_of_weights = np.sum(coreset.weights)
        return CoresetNode(coreset, node1.rank+1)

    ######################################################################

    def create_synthetic_points(file_name, points):
        with open(file_name, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for point in points:
                writer.writerow(point)
