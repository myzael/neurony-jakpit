import argparse
from network import NetworkFactory,step_f,log_f

__author__ = 'pita'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random',nargs='*',type=int)
    parser.add_argument('--data',nargs=1)
    parser.add_argument('--fun',nargs=1)
    args = parser.parse_args()
    fun = None
    if args.fun and args.fun[0] == "step":
        fun = step_f
    elif args.fun and args.fun[0] == "log":
        fun = log_f

    net = NetworkFactory().get_network()
    if args.random:
        net = NetworkFactory().build_random(net,args.random,fun)
    elif args.data:
        net= NetworkFactory().build_from_file(net,args.data[0],fun)

    while True:
        var = raw_input("\n----\nEnter input vector:")
        input = [float(a) for a in var.split()]
        net.compute(input)
        print "\nNetwork after computation:"
        print net
        print "Network result is:"
        print net.result()


if __name__ == "__main__":
    main()
