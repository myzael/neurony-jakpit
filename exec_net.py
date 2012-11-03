import sys

__author__ = 'pita'

from network import Network

def main():
    net = Network()
    if sys.argv[1] == "-R":
        net.create_random([int(a) for a in sys.argv[2:]])

    elif sys.argv[2] == "-D":
        net.create_from_file(sys.argv[3])
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
