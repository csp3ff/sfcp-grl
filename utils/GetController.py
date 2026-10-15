from controller.GCN_PPO import GCN_PPO_Controller
from controller.BaseController import BaseController


def return_controller(config):
    """
    Choose the appropriate controller based on the model specified in the config.
    :param config: Configuration dictionary loaded from YAML.
    :return: An instance of the chosen controller class.
    """
    if config['model'] not in ["GCN_PPO"]:
        raise ValueError(f'Unsupported model type. Please choose {["GCN_PPO"]}')
    
    if config['model'] == "GCN_PPO":
        return GCN_PPO_Controller(config)

    return BaseController(config)