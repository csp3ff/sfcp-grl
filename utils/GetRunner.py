
def return_runner(config_file):
    """
    Choose the appropriate runner based on the model specified in the config file.
    :param config_file: Configuration dictionary loaded from YAML.
    :return: An instance of the chosen runner class.
    """

    if config_file['model'] not in ["GCN_PPO"]:
        raise ValueError(f'Unsupported model type. Please choose {["GCN_PPO"]}')
    
    if config_file['model'] == "GCN_PPO":
        from evolution.GCN_PPO import GCN_PPO
        runner = GCN_PPO(config_file)
    return runner
