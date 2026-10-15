import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from utils.GetController import return_controller
from evolution.Recorder import Recorder

class RunBase:
    def __init__(self, config):
        self.config = config
        self.sdn = return_controller(config)
        self.recorder = Recorder(config)

    def gen_sfcr(self, total_time_steps, arrival_rate):
        from utils.Generator import gen_sfcr_random
        reqs = []
        if self.config['train'] == 1 or self.config['test_type'] == 1:
            reqs = gen_sfcr_random(start=1, end=total_time_steps, arrival_rate=arrival_rate)
        else:
            start = 1
            end = 500
            arr_rate = arrival_rate
            while end <= total_time_steps:
                reqs.extend(gen_sfcr_random(start=start, end=end, arrival_rate=arr_rate))
                start += 500
                end += 500
                arr_rate += 0.05
        if not reqs:
            raise ValueError("Generated SFCRs are empty. Please check the generation logic.")
        return reqs
    
    def save_model(self):
        trained_dir = self.config['train_dir'].replace('train', 'model')
        if not os.path.exists(trained_dir):
            os.makedirs(trained_dir)
        self.sdn.agent.save(os.path.join(trained_dir, 'actor.pth'), os.path.join(trained_dir, 'critic.pth'))
        print("Model saved to:", trained_dir)