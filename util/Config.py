import re
import argparse

__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # Log and save
    parser.add_argument('--print-freq', default=20, type=int,
                        metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--save-freq', default=0, type=int,
                        metavar='EPOCHS', help='checkpoint frequency(default: 0)')
    parser.add_argument('--save-dir', type=str, metavar='DIR')

    # Technical details
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--is-parallel', default=False, type=str2bool,
                        help='use data parallel', metavar='BOOL')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='gpu number (default: 0)')

    # Data
    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=128, type=int,
                        metavar='N', help='batch size for labeled data (default: 128)')
    parser.add_argument('--labels', type=str, default='', metavar='DIR')
    parser.add_argument('--train-subdir', type=str, metavar='DIR')
    parser.add_argument('--eval-subdir', type=str, metavar='DIR')
    parser.add_argument('--twice', default=False, type=str2bool,
                        help='use two data stream', metavar='BOOL')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='lenet')
    parser.add_argument('--model', metavar='MODEL', default='pi')

    # Optimization
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--loss', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'soft'])
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE',
                        choices=['sgd', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, 
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--cons-loss-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'])
    
    # LR schecular
    parser.add_argument('--lr-scheduler', default="cos", type=str, metavar='TYPE',
                        choices=['cos', 'multistep', 'exp-warmup', 'none'])
    parser.add_argument('--min-lr', '--minimum-learning-rate', default=1e-7, type=float,
                        metavar='LR', help='minimum learning rate')
    parser.add_argument('--steps', default="0,", 
                        type=lambda x: [int(s) for s in x.split(',')],
                        metavar='N', help='milestones')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='factor of learning rate decay')
    parser.add_argument('--rampup-length', default=30, type=int, 
                        metavar='EPOCHS', help='length of the ramp-up')
    parser.add_argument('--rampdown-length', default=30, type=int, 
                        metavar='EPOCHS', help='length of the ramp-down')
    
    # MeanTeacher and PI
    parser.add_argument('--ema-decay', default=1e-4, type=float,
                        metavar='W', help='ema weight decay (default:1e-4)')
    parser.add_argument('--cons-weight', default=1.0, type=float, metavar='M',
                        help='the upper of weight for consistency loss')
    parser.add_argument('--weight-rampup', default=30, type=int, metavar='EPOCH',
                        help='the length of rampup weight')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    print("Using these args: ", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
