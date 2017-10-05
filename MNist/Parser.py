import argparse

def parser():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch sie for training (default = 1')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default = 1)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    return parser