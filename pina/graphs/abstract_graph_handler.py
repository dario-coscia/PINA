"""Module for Graph Handler"""

from abc import ABCMeta, abstractmethod


class GraphHandlerInterface(metaclass=ABCMeta):
    """
    Base class for handling graphs in PINA. A graph is used to model
    pairwise relations (edges) between objects (nodes).
    This class handles the construction of the graph.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_graph(self):
        """
        Building the graph topology.
        """
        pass


class DynamicGraphHandlerInterface(GraphHandlerInterface):
    """
    Base class for handling graphs in PINA which change the topology.
    A graph is used to model pairwise relations (edges) between objects (nodes).
    This class handles the construction of the graph, and its dyniamic update
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_topology(self):
        """
        Update the topology of the graph.
        """
        pass