import sys
import torch

import demo_cons
from util.Config import parse_dict_args

def parameters():
    defaults = {
        # Log and save
        'print_freq': 30,
        'save_freq': 30,
        'save_dir': 'checkpoints/',

        # Technical details
        'is_parallel': False,
        'workers': 2,
        'gpu': 3,

        # Data
        'dataset': 'cifar10',
        'base_batch_size': 128,
        'base_labeled_batch_size': 64,
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Architecture
        #'arch': 'lenet',
        #'arch': 'vgg19',
        #'arch': 'resnet18',
        #'arch': 'preact_resnet18',
        #'arch': 'densenet121',
        #'arch': 'resnext29_32x4d',
        #'arch': 'senet',
        #'arch': 'dpn92',
        #'arch': 'shuffleG3',
        #'arch': 'mobileV2',
        'arch': 'convlarge',
        'model': 'ema',

        # Optimization
        'loss': 'soft',
        'optim': 'sgd',#'adam',
        'epochs': 500,
        'base_lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,
        'cons_loss_type': 'mse',

        # lr_schedular
        'steps': '100,150,200,250,300,350,400,450,480',
        'gamma': 0.5,
        'lr_scheduler': 'exp-warmup',
        'min_lr': 1e-4,
        'rampup_length': 80,
        'rampdown_length': 80,
        
        # MeanTeacher
        'cons_weight': 100.0,
        'ema_decay': 0.97,
        'twice': True,
    }

    defaults['save_dir'] += '{}/'.format(defaults['model'])
    for n_labels in [1000]:
        for data_seed in range(10, 15):
            yield{
                **defaults,
                'n_labels': n_labels,
                'data_seed': data_seed
            }

def run(base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, is_parallel, **kwargs):
    if is_parallel and torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
    else:
        ngpu = 1
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr,
        'labels': 'data-local/labels/{}/{}_balanced_labels/{:02d}.txt'.format(kwargs['dataset'], n_labels, data_seed),
        'is_parallel': is_parallel,
    }
    args = parse_dict_args(**adapted_args, **kwargs)
    demo_cons.main(args)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
