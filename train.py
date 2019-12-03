#!python3
import argparse
import time

import torch
import yaml

from trainer import Trainer
# tagged yaml objects
from experiment import Structure, TrainSettings, ValidationSettings, Experiment
from concern.log import Logger
from data.data_loader import DataLoader
from data.image_dataset import ImageDataset
from training.checkpoint import Checkpoint
from training.model_saver import ModelSaver
from training.optimizer_scheduler import OptimizerScheduler
from concern.config import Configurable, Config


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--start_iter', type=int, help='Begin counting iterations starting from this value (should be used with resume)')
    parser.add_argument('--start_epoch', type=int, help='Begin counting epoch starting from this value (should be used with resume)')
    parser.add_argument('--max_size', type=int, help='max length of label')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, help='The optimizer want to use')
    parser.add_argument('--thresh', type=float, help='The threshold to replace it in the representers')
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    parser.add_argument('--visualize', action='store_true', help='visualize maps in tensorboard')
    parser.add_argument('--force_reload', action='store_true', dest='force_reload', help='Force reload data meta')
    parser.add_argument('--no-force_reload', action='store_false', dest='force_reload', help='Force reload data meta')
    parser.add_argument('--validate', action='store_true', dest='validate', help='Validate during training')
    parser.add_argument('--no-validate', action='store_false', dest='validate', help='Validate during training')
    parser.add_argument('--print-config-only', action='store_true', help='print config without actual training')
    parser.add_argument('--debug', action='store_true', dest='debug', help='Run with debug mode, which hacks dataset num_samples to toy number')
    parser.add_argument('--no-debug', action='store_false', dest='debug', help='Run without debug mode')
    parser.add_argument('--benchmark', action='store_true', dest='benchmark', help='Open cudnn benchmark mode')
    parser.add_argument('--no-benchmark', action='store_false', dest='benchmark', help='Turn cudnn benchmark mode off')
    parser.add_argument('-d', '--distributed', action='store_true', dest='distributed', help='Use distributed training')
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    parser.add_argument('-g', '--num_gpus', dest='num_gpus', default=4, type=int, help='The number of accessible gpus')
    parser.set_defaults(debug=False)
    parser.set_defaults(benchmark=True)

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    if args['distributed']:
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    if not args['print_config_only']:
        torch.backends.cudnn.benchmark = args['benchmark']
        trainer = Trainer(experiment)
        trainer.train()

if __name__ == '__main__':
    main()

