#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################


from __future__ import division

from set_of_points import SetOfPoints


"""
Class that represent a node in the coreset tree
Attributes:
    lines (SetOfLines): The set of lines
    rank (integer): The rank of the node in the tree
"""


class CoresetNode:

    def __init__(self, P = SetOfPoints(), rank = 1):
        self.points = P
        self.rank = rank