import argparse

def parser():
    parser = argparse.ArgumentParser(description='Mobile Net argument parser')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--test-batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default = 10)
    parser.add_argument('--seed', type=int, default = 1)

    return parser