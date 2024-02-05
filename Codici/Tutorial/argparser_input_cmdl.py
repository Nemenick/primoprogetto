import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--position", type=int)
parser.add_argument("-s", "--sample", type=int)
parser.add_argument("-f", "--fname", type=str)

args = parser.parse_args()
print(args.fname)