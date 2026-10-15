import sys
import time

from utils.GetController import return_controller
from evolution.RunBase import RunBase
import os

class GCN_PPO(RunBase):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        print(">> Train model")
        if not os.path.exists(self.config['train_dir']):
            os.makedirs(self.config['train_dir'], exist_ok=True)
        print("Results will be saved to:", self.config['train_dir'])

        max_r = -1e9
        for episode in range(self.config['num_episode']):
            self.sdn.reset()
            self.recorder.init()
            self.recorder.set_time(episode, time.time())
            arrival_rate = self.config['arrival_rate']
            timeline = self.gen_sfcr(total_time_steps=self.config['total_time_steps'], arrival_rate=arrival_rate)
            log_info = " " + self.config['model'] + ": "
            index = 0
            for timeslot in range(self.config['total_time_steps']):
                self.sdn.clear_sfcr(timeslot)
                while index < len(timeline) and (timeline[index].start_t <= timeslot):
                    sfcr = timeline[index]
                    result = self.sdn.learn(sfcr, timeslot)
                    self.recorder.record(episode, result)
                    index += 1
                # updata the policy
                loss = self.sdn.agent.update_policy()
                if loss:
                    print(loss)
                    self.recorder.record_loss(episode, loss)
                if (episode + 1) % 1000 == 0:
                    print('\r' + f"Episode: {episode + 1} / {self.config['num_episode']}, "
                            + f"{timeslot + 1} / {self.config['total_time_steps']}", end='', flush=True)
            assert self.recorder.success_sfc[episode] != 0, "No success sfc"
            self.recorder.set_time(episode, time.time())
            log_info += self.recorder.return_info(episode)
            ep_r = self.recorder.return_reward(episode)
            if ep_r > max_r:
                max_r = ep_r
                self.save_model()
            print(log_info)
        self.recorder.save_results()
        # trained_dir = self.config['train_dir'].replace('train', 'model')
        # if not os.path.exists(trained_dir):
        #     os.makedirs(trained_dir)
        # self.sdn.agent.save(os.path.join(trained_dir, 'actor.pth'), os.path.join(trained_dir, 'critic.pth'))
        # print("Train completed, result will be saved to: ", trained_dir)
    
    def test(self, trained_dir):
        print(">> Test model")
        assert trained_dir is not None, "Test directory must be provided."
        father_dirname = os.path.dirname(trained_dir).split('/')[-1]
        test_dir = trained_dir.replace('train', 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)
        test_results_file = os.path.join(test_dir, f'{father_dirname}.csv')
        print("Results will be saved to:", test_results_file)
        

        model_dir = trained_dir.replace('train', 'model')
        self.sdn.agent.load(os.path.join(model_dir, 'actor.pth'), os.path.join(model_dir,'critic.pth'))
        timeline = self.gen_sfcr(total_time_steps=self.config['total_time_steps'], arrival_rate=self.config['arrival_rate'])
        
        index = 0
        self.recorder.init()
        self.recorder.set_time(0, time.time())
        for timeslot in range(self.config['total_time_steps']):
            self.sdn.clear_sfcr(timeslot)
            while index < len(timeline) and (timeline[index].start_t <= timeslot):
                sfcr = timeline[index]
                result = self.sdn.solve(sfcr, timeslot)
                self.recorder.record(0, result)
                index += 1
            print('\r' + f"Testing at timeslot {timeslot + 1} / {self.config['total_time_steps']}", end='', flush=True)
        print("\n >> Test completed.")
        assert self.recorder.success_sfc[0] != 0, "No success sfc"
        log_info = self.config['model'] + ": "
        self.recorder.set_time(0, time.time())
        log_info += self.recorder.return_info(0)
        print(log_info)
        self.recorder.save_results()

    def test_per_500(self, trained_dir):
        """
        每500个时间单位统计一次指标的测试函数
        """
        print(">> Test model with 500-interval statistics")
        assert trained_dir is not None, "Test directory must be provided."
        father_dirname = os.path.dirname(trained_dir).split('/')[-1]
        test_dir = trained_dir.replace('train', 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir, exist_ok=True)
        
        # 更新配置以使用新的测试目录
        self.config['train_dir'] = test_dir
        self.config['timestamp'] = father_dirname
        
        print("Results will be saved to:", test_dir)

        # 加载模型
        model_dir = trained_dir.replace('train', 'model')
        self.sdn.agent.load(os.path.join(model_dir, 'actor.pth'), os.path.join(model_dir,'critic.pth'))
        
        # 生成SFC请求时间线
        timeline = self.gen_sfcr(total_time_steps=self.config['total_time_steps'], arrival_rate=self.config['arrival_rate'])
        
        # 初始化间隔记录
        self.recorder.init_interval_recording()
        
        index = 0
        print(f"Starting test with {self.config['total_time_steps']} time slots...")
        
        for timeslot in range(self.config['total_time_steps']):
            self.sdn.clear_sfcr(timeslot)
            
            # 处理当前时间槽的所有请求
            while index < len(timeline) and (timeline[index].start_t <= timeslot):
                sfcr = timeline[index]
                result = self.sdn.solve(sfcr, timeslot)
                # 使用间隔记录方法
                self.recorder.record_interval(timeslot, result)
                index += 1
            
            # 每100个时间槽显示一次进度
            if (timeslot + 1) % 100 == 0 or (timeslot + 1) == self.config['total_time_steps']:
                print(f'\rTesting progress: {timeslot + 1} / {self.config["total_time_steps"]} '
                      f'({((timeslot + 1) / self.config["total_time_steps"] * 100):.1f}%)', end='', flush=True)
        
        print("\n >> Test completed.")
        
        # 完成间隔记录并保存结果
        self.recorder.finalize_interval_recording()
        self.recorder.save_interval_results()
        
        print(f"Test completed successfully. Results saved to: {test_dir}")
