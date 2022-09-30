import argparse

from bonfire_benchmark.datasets import dataset_names, get_dataset_clz
from bonfire_benchmark.models import model_names, get_model_clz
from bonfire.train.trainer import Trainer
from bonfire.util import get_device
from bonfire.util.config_util import parse_yaml_config
from bonfire_benchmark import get_config_path

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_name', choices=model_names, help='The model to train.')
    parser.add_argument('-r', '--n_repeats', default=5, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.n_repeats


def run_training():
    # Parse args
    dataset_name, model_name, n_repeats = parse_args()

    # Parse wandb config and get training config for this model
    config_path = get_config_path(dataset_name)
    config = parse_yaml_config(config_path, model_name)
    print(config)

    # Create trainer
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)
    trainer = Trainer(device, model_clz, dataset_clz)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(config, n_repeats=n_repeats)


if __name__ == "__main__":
    run_training()
