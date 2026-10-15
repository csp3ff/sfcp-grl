import copy
from datetime import datetime
import sys

from utils.GetRunner import return_runner
import os
import yaml
config_path = "train.yml"
with open(config_path, 'r', encoding='utf-8') as f:
    config_file = yaml.safe_load(f)
data_path = config_file['data_path']
config_file['data_path'] = os.path.join(data_path, "train")
config_file['train'] = 1
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print("Start at: ", timestamp)
config_file['train_dir'] = f"results/{config_file['model']}/{timestamp}/train"
config_file['timestamp'] = timestamp
runner = return_runner(config_file)
runner.train()

