import argparse

from bonfire import dataset_names
from bonfire import model_names
from bonfire import create_benchmark_tuner
from bonfire import get_device

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL tuning script.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_name', choices=model_names, help='The model to tune.')
    parser.add_argument('-n', '--n_trials', default=100, type=int, help='The number of trials to run when tuning.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.n_trials


def run_tuning():
    dataset_name, model_name, n_trials = parse_args()

    # Create tuner
    study_name = 'Tune_{:s}_{:s}'.format(dataset_name, model_name)
    tuner = create_benchmark_tuner(device, model_name, dataset_name, study_name, n_trials)

    # Log
    print('Starting {:s} tuning'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using dataset {:}'.format(dataset_name))
    print('  Using device {:}'.format(device))
    print('  Running study with {:d} trials'.format(n_trials))

    # Start run
    tuner.run_study()


if __name__ == "__main__":
    run_tuning()
