import csv

import torch
from sklearn.model_selection import train_test_split

from bonfire.main.data.mil_dataset import MilDataset
from abc import ABC, abstractmethod


class MuskDataset(MilDataset, ABC):

    d_in = 166
    n_expected_dims = 2  # i * f
    n_classes = 2

    def __init__(self, bag_names, bags, targets):
        super().__init__(bags, targets, None)
        self.bag_names = bag_names

    # TODO do this as a mixin? Tef datasets are also using csv
    @classmethod
    @property
    @abstractmethod
    def csv_path(cls):
        pass

    @classmethod
    def create_datasets(cls, random_state=12):
        parsed_data = parse_data(cls.csv_path)
        bag_names, bags, targets = parsed_data

        bags = normalise(bags)

        splits = train_test_split(bag_names, bags, targets, train_size=0.7, stratify=targets, random_state=random_state)

        train_bag_names, train_bags, train_targets = [splits[i] for i in [0, 2, 4]]
        test_bag_names, test_bags, test_targets = [splits[i] for i in [1, 3, 5]]

        splits = train_test_split(test_bag_names, test_bags, test_targets, train_size=0.5, stratify=test_targets,
                                  random_state=random_state)

        val_bag_names, val_bags, val_targets = [splits[i] for i in [0, 2, 4]]
        test_bag_names, test_bags, test_targets = [splits[i] for i in [1, 3, 5]]

        train_dataset = MuskDataset(train_bag_names, train_bags, train_targets)
        val_dataset = MuskDataset(val_bag_names, val_bags, val_targets)
        test_dataset = MuskDataset(test_bag_names, test_bags, test_targets)

        return train_dataset, val_dataset, test_dataset


class Musk1Dataset(MuskDataset):

    name = "Musk1"
    csv_path = "./data/MUSK/clean1.data"


class Musk2Dataset(MuskDataset):

    name = "Musk2"
    csv_path = "./data/MUSK/clean2.data"


def parse_data(csv_path):
    bag_data = {}
    bag_targets = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            molecule_name, instance_vector, instance_target = parse_line(line)
            if molecule_name not in bag_data:
                bag_data[molecule_name] = []
                bag_targets[molecule_name] = 0
            bag_data[molecule_name].append(instance_vector)
            if instance_target != 0:
                bag_targets[molecule_name] = 1

    bag_names = []
    bags = []
    targets = []
    for bag_name in bag_data.keys():
        bag_names.append(bag_name)
        bag = bag_data[bag_name]
        bag = torch.stack(bag)
        bags.append(bag)
        target = bag_targets[bag_name]
        targets.append(target)

    return bag_names, bags, targets


def parse_line(line):
    molecule_name = line[0]
    instance_vector = line[2:-1]
    instance_target = int(float(line[-1]))
    features = torch.as_tensor([float(f) for f in instance_vector])
    return molecule_name, features, instance_target


# TODO as method in abstract base class?
def normalise(bags):
    all_instances = torch.cat(bags)
    dataset_mean = torch.mean(all_instances, dim=0)
    dataset_std = torch.std(all_instances, dim=0)
    norm_bags = []
    for bag in bags:
        norm_bag = (bag - dataset_mean) / dataset_std
        norm_bags.append(norm_bag)
    return norm_bags
