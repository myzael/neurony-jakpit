import argparse
from network import NetworkFactory, step_f,log_f

__author__ = 'pita'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random',nargs='*',type=int)
    parser.add_argument('--data',nargs=1)
    parser.add_argument('--learn_file',nargs=1)
    parser.add_argument('--steps',nargs=1,type=int)
    parser.add_argument('--fun',nargs=1)
    args = parser.parse_args()
    fun = None
    if args.fun and args.fun[0] == "step":
        fun = step_f
    elif args.fun and args.fun[0] == "log":
        fun = log_f

    net = NetworkFactory().get_kohonen_network()

    # init weights from file or randomly
    if args.random:
        net = NetworkFactory().build_random(net,args.random,fun,init_zeros=False)
    elif args.data:
        net= NetworkFactory().build_from_file(net,args.data[0],fun)

    # init specific kohonen properties
    net.findNeighbors = net.findNeighbors2D
    net.radius = 1
    net.teta = 0.06
    net.max_teta = net.teta
    net.conscious_min = 0.75

    # init conscious
    for node in net.layers[1]:
        node.conscious = 1.0
    net.conscious = True

    # teach network
    with open(args.learn_file[0]) as f:
        learning_set = [[float(a) for a in line.split()] for line in f]
    for step in xrange(args.steps[0]):
        net.learn(step,args.steps[0],learning_set)


    while True:
        for node in net.layers[1]:
            print [0 if a < 0.01 else 1 for a in node.weights ]
        var = raw_input("\n----\nEnter input vector:")
        input = [float(a) for a in var.split()]
        net.compute(input)
        print "\nNetwork after computation:"
        print net
        print "Network result is:"
        print net.result()
        print "Matched node:"
        print net.result().index(max(net.result()))

if __name__ == "__main__":
    main()
