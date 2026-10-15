
import os
import pandas as pd


class Evaluator:
    def __init__(self, config):
        self.success_sfc = 0
        self.total_sfc = 0
        self.delay = 0
        self.revenue = 0
        self.cost = 0
        self.LRCR = 0
        self.model = config['model']
    
    def reset(self):
        self.success_sfc = 0
        self.total_sfc = 0
        self.delay = 0
        self.revenue = 0
        self.cost = 0
        self.LRCR = 0

    def step(self, status, delay, revenue, cost, reward):
        if status:
            self.success_sfc += 1
        self.total_sfc += 1
        self.delay += delay
        self.revenue += revenue
        self.cost += cost
        self.LRCR += reward

    def save(self, path):
        new_data = pd.DataFrame({
            f"acc_{self.model}": [self.success_sfc / self.total_sfc],
            f"delay_{self.model}": [self.delay / self.success_sfc],
            f"ave rev_{self.model}": [self.revenue / 500],
            f"ave cost_{self.model}": [self.cost / 500],
            f"ave LRCR_{self.model}": [self.LRCR / 500],
        })
        print('\n', new_data)
        if os.path.exists(path):
            old_data = pd.read_csv(path)
            new_data = pd.concat([old_data, new_data], ignore_index=True)
        new_data.to_csv(path, index=False)
