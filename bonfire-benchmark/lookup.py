def create_trainer_from_names(device, model_name, dataset_name, project_name=None):
    # Parse model and dataset classes
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)

    # Create the trainer
    return create_trainer_from_clzs(device, model_clz, dataset_clz, project_name=project_name)


def create_tuner_from_config(device, model_name, dataset_name, study_name, n_trials):
    # Get model and dataset classes
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)

    # Load training and tuning configs
    config = parse_yaml_benchmark_config(dataset_name)
    training_config = parse_training_config(config['training'], model_name)
    tuning_config = parse_tuning_config(config['tuning'], model_name)

    # Create tuner
    return Tuner(device, model_clz, dataset_clz, study_name, training_config, tuning_config, n_trials)