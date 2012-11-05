import argparse
import network

__author__ = 'pita'


def main():
    net = network.Network()
    parser = argparse.ArgumentParser()
    parser.add_argument('--random',nargs='*',type=int)
    parser.add_argument('--data',nargs=1)
    parser.add_argument('--fun',nargs=1)
    args = parser.parse_args()

    if args.fun == "step":
        args.fun = network.step_f
    elif args.fun == "log":
        args.fun == network.log_f
    if args.random:
        net.create_random(args.random,args.fun)
    elif args.data:
        net.create_from_file(args.data,args.fun)

    while True:
        var = raw_input("Enter input vector:")
        input = [float(a) for a in var.split()]
        net.compute(input)
        print "network after computation:"
        print net.layers
        print "network result is:"
        print net.result()


if __name__ == "__main__":
    main()
