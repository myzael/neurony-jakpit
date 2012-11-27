import random
from math import exp, sqrt

__author__ = 'Pita'

def get_output(layer):
    return [l.output for l in layer]

def get_with_conscious(net):
    if net.conscious:
        return [(i,node) for i,node in enumerate(net.layers[1]) if node.conscious > net.conscious_min]
    else:
        return net.layers[1]

def step_f(input):
    """
    Activation function: step function
    """
    return 0.0 if input <= 0.0 else 1.0

def log_f(input):
    """
    Activation function: sigmoid function
    """
    return 1.0 / (1.0 + exp(-1.0 * input))

def euclidean(x,y):
    sumSq=0.0

    #add up the squared differences
    for i in range(len(x)):
        sumSq+=(x[i]-y[i])**2

    #take the square root of the result
    return sumSq**0.5

def compare_euclidean(net,x,y):
    return cmp(euclidean(x,get_output(net.layers[0])),euclidean(y,get_output(net.layers[0])))

class Node:
    """
    Represents each computational node.
    Each node has information about network and layer in which it's in -
    it simplifies computation process.
    """
    def __init__(self,layer,network,weights=[],activ_f=None,bias=0.0):
        self.weights = weights
        self.layer = layer
        self.output = None
        self.bias = bias
        self.network = network
        self.activ_f = activ_f


    def compute(self):
        # multiply weights by output of previous layer adding bias at the end
        self.output = sum(x*y for x,y in zip(self.weights,get_output(self.network.layers[self.layer - 1]))) - self.bias
        # apply activation function
        if self.activ_f is not None:
            self.output = self.activ_f(self.output)

    def __repr__(self):
        if self.activ_f:
            fun_name = self.activ_f.__name__
        else:
            fun_name = "Brak"
        return "w: %s, bias:%s, output:%s, fun:%s" % (str(self.weights), str(self.bias), str(self.output), fun_name)


class Network:
    """
    Represents neural network.
    Layers are kept in list, each layer is represented by list of nodes.
    """
    def __init__(self):
        self.layers = []

    def __repr__(self):
        s = ""
        for index,l in enumerate(self.layers):
            s= "".join([s,"layer ",str(index),":\n",str(l),"\n"])
        return s

    def set_input(self, input):
        # normalize input
        suma = sum([x*x for x in input])
        input = [x / suma for x in input]
        for index, value in enumerate(input):
            self.layers[0][index].output = value

    def compute(self,input):
        """
        Compute whole network, based on given input.
        :param input: Input vector for network computation
        """
        # set first layer
        self.set_input(input)
        # compute other layers
        for layer in self.layers[1:]:
            [l.compute() for l in layer]

    def result(self):
        """
        Returns final vector with network answer
        """
        return get_output(self.layers[len(self.layers)-1])

class KohonenNetwork(Network):
    """
    This network has 2 layers - input and kohonen layer.
    """

    def learn(self,step,steps,learning_set):
        """
        Represents one step in learning process
        """
        for data_set in learning_set:
            self.set_input(data_set)
            n = self.findWinner()
            self.modifyConscious(n)
            neighbors = self.findNeighbors(n)
            self.modifyWeights([(n,1)] + neighbors)
        self.reduceFactors(step,steps)

    def findWinner(self):
        '''
        Retrieves nr of node which is closest to input vector
        '''
        weights = [(i, x.weights) for i,x in get_with_conscious(self)]
        # sort due to distance from input vector
        weights.sort(cmp=lambda x,y : compare_euclidean(self,x[1],y[1]))
        # return index of winning node
        return weights[0][0]

    def findNeighbors1D(self, n):
        """
        Returns list of tuples with two elements - first is the nr of neighbor node, second is distance
        """
        neighbors = []
        for i in xrange(1,self.radius+1):
            if n-i >= 0:
                neighbors.append( (n-i,i+1) )
            if n+i < len(self.layers[1]):
                neighbors.append( (n+i,i+1) )
        return neighbors

    def findNeighbors2D(self, n):
        """
        Returns list of tuples with two elements - first is the nr of neighbor node, second is distance
        """
        rows = sqrt(len(self.layers[1]))
        neighbors = []
        for i in xrange(len(self.layers[1])):
            if n != i:
                dist = abs(i // rows - n // rows) + abs(i % rows - n % rows)
                if dist <= self.radius:
                    neighbors.append( (i,int(dist)+1) )
        return neighbors

    def modifyWeights(self, neighbors):
        for node,radius in neighbors:
            for i,values in enumerate(zip(self.layers[1][node].weights,get_output(self.layers[0]))):
                self.layers[1][node].weights[i] += self.teta * (1 / radius) * (values[1] - values[0])

    def reduceFactors(self,step,steps):
        self.teta = self.max_teta - self.max_teta * sqrt(step/steps)

    def modifyConscious(self, n):
        self.layers[1][n].conscious -= self.conscious_min
        for i,node in enumerate(self.layers[1]):
            if i != n:
                new_val = node.conscious + 1.0 / len(self.layers[1])
                node.conscious = new_val if new_val <= 1.0 else 1.0

class NetworkFactory:
    """
    Provides method which allows to initialize neural network.
    Network can be created from filedata or with random weights and bias
    """

    def get_network(self):
        return Network()

    def get_kohonen_network(self):
        return KohonenNetwork()

    def build_from_file(self,net,filename,activ_f):
        """
        Initialize neural network based on given file
        """

        with open(filename) as f:
            # create first layer
            first_layer_nodes = len(f.readline().split())-2
            first_layer = [ Node(0,net) for _ in xrange(first_layer_nodes)]
            net.layers.append(first_layer)
            f.seek(0)

            #create all other layers
            line = f.readline()
            actual_l = 1
            while line:
                net.layers.append([])
                while line and "---" not in line:
                    params = line.split()
                    # in last line there's activation function, decide which one to use. If other - default
                    if params[-1] == "step":
                        activ_f = step_f
                    elif params[-1] == "log":
                        activ_f = log_f
                    net.layers[actual_l].append( Node(actual_l,net,[float(l) for l in params[:-2]],activ_f,float(params[-2])))
                    line = f.readline()
                actual_l+=1
                line = f.readline()
        return net


    def build_random(self,net,arguments,activ_f,init_zeros=False):
        """
        Initialize neural network with random weights
        param: arguments: quantity of nodes in each layer
        """
        # create first layer
        first_layer = [ Node(0,net) for _ in xrange(arguments[0])]
        net.layers.append(first_layer)

        # create all other layers
        for index,a in enumerate(arguments[1:]):
            if init_zeros:
                layer = [ Node(index+1,net,[0 for _ in xrange(arguments[index])],activ_f) for _ in xrange(a) ]
            else:
                layer = [ Node(index+1,net,[random.random()*2.0 - 1.0 for _ in xrange(arguments[index])],activ_f) for _ in xrange(a) ]
            net.layers.append(layer)
        return net