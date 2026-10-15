import json
import os

import numpy as np
from pandas import DataFrame
import yaml


class Recorder:
    def __init__(self, config):
        self.delay = []
        self.revenue = []
        self.cost = []
        self.reward = []
        self.revenue_cost_ratio = []
        self.success_sfc = []
        self.total_sfc = []
        self.processing_time = []
        self.time_flag = 0
        self.config = config
        self.loss = []
        
        # 新增：每500时间单位的统计
        self.interval_stats = []  # 存储每个时间段的统计信息
        self.interval_size = 500  # 时间间隔大小
        self.current_interval = 0  # 当前时间间隔索引
        self.current_interval_data = self._init_interval_data()  # 当前间隔的数据

    def _init_interval_data(self):
        """初始化单个时间间隔的数据结构"""
        return {
            'start_time': self.current_interval * self.interval_size + 1,
            'end_time': (self.current_interval + 1) * self.interval_size,
            'delay': 0,
            'revenue': 0,
            'cost': 0,
            'reward': 0,
            'revenue_cost_ratio': 0,
            'success_sfc': 0,
            'total_sfc': 0,
            'arrival_rate': 0.1 + self.current_interval * 0.05  # 根据你的到达率逻辑
        }

    def init(self):
        """
        Initialize the recorder with default values.
        This is typically called at the beginning of a new episode or test run.
        """
        self.delay.append(0)
        self.revenue.append(0)
        self.cost.append(0)
        self.reward.append(0)
        self.revenue_cost_ratio.append(0)
        self.success_sfc.append(0)
        self.total_sfc.append(0)
        self.processing_time.append(0)
        self.loss.append({})
        self.time_flag = 0

    def record_loss(self, episode, loss):
        """
        Record the loss for the current episode.
        :param episode: The current episode number.
        :param loss: The loss value to record.
        """
        for key in loss.keys():
            if key not in self.loss[episode]:
                self.loss[episode][key] = []
            self.loss[episode][key].append(loss[key])

    def set_time(self, episode, time):
        if self.time_flag == 0:
            self.time_flag = time
        else:
            self.processing_time[episode] = time - self.time_flag
            self.time_flag = 0

    def return_delay(self, episode):
        assert self.success_sfc[episode] > 0, "Success SFC count must be greater than zero to calculate average delay."
        return self.delay[episode] / self.success_sfc[episode]
    
    def return_reward(self, episode):
        return self.reward[episode]

    def return_lrcr(self, episode):
        assert self.config['total_time_steps'] > 0, "Total time steps must be greater than zero to calculate average revenue/cost ratio."
        return self.revenue_cost_ratio[episode] / self.config['total_time_steps']
    
    def return_acc(self, episode):
        assert self.total_sfc[episode] > 0, "Total SFC count must be greater than zero to calculate success rate."
        return self.success_sfc[episode] / self.total_sfc[episode]

    def return_info(self, episode):
        """
        Return a formatted string with the recorded information.
        :return: A string containing the success rate, average delay, and average revenue/cost ratio.
        """
        info = f"Success Rate: {self.return_acc(episode):.4f}, ({self.success_sfc[episode]}/{self.total_sfc[episode]}), "
        info += f"Average Delay: {self.return_delay(episode):.4f}, "
        info += f"Average Revenue/Cost Ratio: {self.return_lrcr(episode):.4f}"
        info += f", Average Reward: {self.return_reward(episode):.4f}"
        info += f", Total Processing Time: {self.processing_time[episode]:.4f} seconds"
        mean_loss = {key: sum(value) if value else 0 for key, value in self.loss[episode].items()}
        info += f", Loss: {json.dumps(mean_loss)}"
        return info
    

    def record(self, episode, result):
        """
        Record the results of processing an SFC request.
        :param episode: The current episode number.
        :param result: A dictionary containing the results of the SFC processing.
                       Expected keys: 'status', 'delay', 'revenue', 'cost', 'revenue_cost_ratio', 'rewards'.
        """
        status = result['status']
        delay = result['delay'] 
        revenue = sum(result['revenue'])
        cost = sum(result['cost'])
        revenue_cost_ratio = sum(result['revenue_cost_ratio'])
        reward = sum(result['rewards'])
        
        if status:
            self.success_sfc[episode] += 1
            self.delay[episode] += delay
            self.revenue[episode] += revenue
            self.cost[episode] += cost
            self.revenue_cost_ratio[episode] += revenue_cost_ratio
        self.reward[episode] += reward
        self.total_sfc[episode] += 1

        # 每500时间单位统计一次
        if (episode + 1) % self.interval_size == 0:
            self.current_interval += 1
            self.interval_stats.append(self._calculate_interval_stats())

    def _calculate_interval_stats(self):
        """计算当前时间间隔的统计信息"""
        stats = {
            'interval': self.current_interval,
            'start_time': self.current_interval_data['start_time'],
            'end_time': self.current_interval_data['end_time'],
            'total_delay': self.delay[-self.interval_size:],
            'total_revenue': self.revenue[-self.interval_size:],
            'total_cost': self.cost[-self.interval_size:],
            'total_reward': self.reward[-self.interval_size:],
            'total_revenue_cost_ratio': self.revenue_cost_ratio[-self.interval_size:],
            'total_success_sfc': self.success_sfc[-self.interval_size:],
            'total_sfc': self.total_sfc[-self.interval_size:],
        }
        # 更新当前间隔的数据
        self.current_interval_data = self._init_interval_data()
        return stats

    def init_interval_recording(self):
        """初始化间隔记录模式"""
        self.interval_stats = []
        self.current_interval = 0
        self.current_interval_data = self._init_interval_data()

    def record_interval(self, timeslot, result):
        """
        按时间间隔记录结果
        :param timeslot: 当前时间槽
        :param result: SFC处理结果
        """
        # 检查是否需要切换到下一个时间间隔
        if timeslot > self.current_interval_data['end_time']:
            # 保存当前间隔的统计信息
            self._finalize_current_interval()
            
            # 切换到下一个间隔
            self.current_interval += 1
            self.current_interval_data = self._init_interval_data()
        
        # 记录当前结果到当前间隔
        status = result['status']
        delay = result['delay'] 
        revenue = sum(result['revenue'])
        cost = sum(result['cost'])
        revenue_cost_ratio = sum(result['revenue_cost_ratio'])
        reward = sum(result['rewards'])
        
        if status:
            self.current_interval_data['success_sfc'] += 1
            self.current_interval_data['delay'] += delay
            self.current_interval_data['revenue'] += revenue
            self.current_interval_data['cost'] += cost
            self.current_interval_data['revenue_cost_ratio'] += revenue_cost_ratio
        
        self.current_interval_data['reward'] += reward
        self.current_interval_data['total_sfc'] += 1

    def _finalize_current_interval(self):
        """完成当前时间间隔的统计并保存"""
        if self.current_interval_data['total_sfc'] > 0:
            # 计算平均值
            interval_result = {
                'interval': self.current_interval,
                'start_time': self.current_interval_data['start_time'],
                'end_time': self.current_interval_data['end_time'],
                'arrival_rate': self.current_interval_data['arrival_rate'],
                'success_rate': self.current_interval_data['success_sfc'] / self.current_interval_data['total_sfc'],
                'total_requests': self.current_interval_data['total_sfc'],
                'successful_requests': self.current_interval_data['success_sfc'],
                'average_delay': self.current_interval_data['delay'] / max(self.current_interval_data['success_sfc'], 1),
                'total_revenue': self.current_interval_data['revenue'],
                'total_cost': self.current_interval_data['cost'],
                'average_revenue_cost_ratio': self.current_interval_data['revenue_cost_ratio'] / self.interval_size,
                'total_reward': self.current_interval_data['reward']
            }
            self.interval_stats.append(interval_result)

    def finalize_interval_recording(self):
        """完成所有间隔的记录"""
        # 处理最后一个间隔
        if self.current_interval_data['total_sfc'] > 0:
            self._finalize_current_interval()

    def save_interval_results(self):
        """保存间隔统计结果到CSV文件"""
        if not self.interval_stats:
            print("No interval statistics to save.")
            return
        
        # 确保输出目录存在
        output_dir = self.config['train_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建DataFrame
        data = DataFrame(self.interval_stats)
        
        # 保存到CSV文件
        results_file = os.path.join(output_dir, f"interval_results_{self.config.get('timestamp', 'test')}.csv")
        data.to_csv(results_file, index=False)
        print(f"Interval results saved to {results_file}")
        
        # 打印统计摘要
        print(f"\nInterval Statistics Summary:")
        print(f"Total intervals: {len(self.interval_stats)}")
        for stat in self.interval_stats:
            print(f"  Interval {stat['interval']} (t={stat['start_time']}-{stat['end_time']}): "
                  f"Rate={stat['arrival_rate']:.2f}, Success={stat['success_rate']:.4f}, "
                  f"Delay={stat['average_delay']:.4f}")

    def save_results(self):
        """
        Save the recorded results to a csv file.
        :param episode: The current episode number.
        :param train_dir: The directory where the results will be saved.
        """
        if not os.path.exists(self.config['train_dir']):
            os.makedirs(self.config['train_dir'], exist_ok=True)

        data = DataFrame({
            'Episode': list(range(self.config['num_episode'] if self.config['train'] == 1 else 1)),
            'ACC': [self.return_acc(i) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)],
            'Average Delay': [self.return_delay(i) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)],
            'Long Term Revenue/Cost Ratio': [self.return_lrcr(i) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)],
            'Average Reward': self.reward,
            'Processing Time': self.processing_time
        })

        # 如果有loss信息，则添加到DataFrame中
        if self.loss[0] != {}:
            data['Loss_actor'] = [np.sum(self.loss[i]['actor_loss']) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)]
            data['Loss_critic'] = [np.sum(self.loss[i]['critic_loss']) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)]
            data['Loss_episode'] = [np.sum(self.loss[i]['total_loss']) for i in range(self.config['num_episode'] if self.config['train'] == 1 else 1)]


        if self.config['train'] == 1:
            train_dir = os.path.join(self.config['train_dir'])
            if not os.path.exists(train_dir):
                os.makedirs(train_dir, exist_ok=True)
            results_file = os.path.join(train_dir, f"{self.config['timestamp']}.csv")
            data.to_csv(results_file, index=False)
            # 写入配置文件,yml格式
            config_file_path = os.path.join(os.path.dirname(self.config['train_dir']), 'config.yml')
            with open(config_file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True)
            print(f"Train results saved to {results_file}")
        else:
            test_dir = os.path.join(self.config['train_dir'].replace('train', 'test'))
            if not os.path.exists(test_dir):
                os.makedirs(test_dir, exist_ok=True)
            results_file = os.path.join(test_dir, f"{self.config['timestamp']}.csv")
            data.to_csv(results_file, index=False)
            print(f"Test results saved to {results_file}")



