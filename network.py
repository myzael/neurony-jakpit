import random
import sys

__author__ = 'Pita'

def get_output(layer):
    return [l.output for l in layer]

class Node:
    """
    Represents each computational node.
    Each node has information about network and layer in which it's in -
    it simplifies computation process.
    """
    def __init__(self, layer,network,weights=[],activ_f=None):
        self.weights = weights
        self.output = None
        self.layer = layer
        self.network = network
        self.activ_f = activ_f

    def compute(self):
        # multiply weights by output of previous layer
        self.output = sum(x*y for x,y in zip(self.weights,get_output(self.network.layers[self.layer - 1])))
        # apply activation function
        if self.activ_f is not None:
            self.output = self.activ_f(self.output)

    def __repr__(self):
        return "w: " + str(self.weights) + ", output:" + str(self.output) + " layer:" + str(self.layer)


class Network:
    """
    Represents neural network.
    Layers are kept in list, each layer is represented by list of nodes.
    """
    def __init__(self):
        self.layers = []

    def compute(self,input):
        """
        Compute whole network, based on given input.
        :param input: Input vector for network computation
        """
        # set first layer
        for index,value in enumerate(input):
            self.layers[0][index].output = value
        # compute other layers
        for layer in self.layers[1:]:
            [l.compute() for l in layer]

    def result(self):
        """
        Returns final vector with network answer
        """
        return get_output(self.layers[len(self.layers)-1])

    def create_from_file(self,filename):
        """
        Initialize neural network based on given file
        """
        with open(filename) as f:
            # create first layer
            first_layer_nodes = len(f.readline().split())
            first_layer = [ Node(0,self) for _ in xrange(first_layer_nodes)]
            self.layers.append(first_layer)
            f.seek(0)

            #create all other layers
            line = f.readline()
            actual_l = 1
            while line:
                self.layers.append([])
                while line and "---" not in line:
                    #TODO Active fun!
                    self.layers[actual_l].append( Node(actual_l,self,[float(l) for l in line.split()]) )
                    line = f.readline()
                actual_l+=1
                line = f.readline()


    def create_random(self,arguments):
        """
        Initialize neural network with random weights
        param: arguments: quantity of nodes in each layer
        """
        # create first layer
        first_layer = [ Node(0,self) for _ in xrange(arguments[0])]
        self.layers.append(first_layer)

        # create all other layers
        for index,a in enumerate(arguments[1:]):
            layer = [ Node(index+1,self,[random.random() for _ in xrange(arguments[index])]) for _ in xrange(a) ]
            self.layers.append(layer)


