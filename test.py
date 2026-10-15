import argparse
import sys

from utils.GetRunner import return_runner
import yaml

parser = argparse.ArgumentParser(description="参数")
parser.add_argument('--p', required=True, help='测试路径')
parser.add_argument('--a', required=False, help='到达率')
parser.add_argument('--t', type=int, default=1, choices=[1, 2], 
                    help='测试类型: 1=常规测试(test), 2=每500时间单位统计测试(test_per_500)')
args = parser.parse_args()
config_path = args.p.split('train')[0] + 'config.yml' 
with open(config_path, 'r', encoding='utf-8') as f:
    config_file = yaml.safe_load(f)
data_path = config_file['data_path']
config_file['train'] = 0
config_file['train_dir'] = args.p.replace('train', 'test')
if args.a:
    config_file['arrival_rate'] = float(args.a)

if args.t == 2:
    config_file['test_type'] = 0  
    if not args.a:
        config_file['arrival_rate'] = 0.1  
    config_file['total_time_steps'] = 10000  
    print(f"使用测试类型2: 每500时间单位统计测试")
    print(f"初始到达率: {config_file['arrival_rate']}")
    print(f"总时间步数: {config_file['total_time_steps']}")
else:
    config_file['test_type'] = 1  
    print(f"使用测试类型1: 常规测试")
    if args.a:
        print(f"到达率: {config_file['arrival_rate']}")

runner = return_runner(config_file)

if args.t == 2:
    runner.test_per_500(args.p)
else:
    runner.test(args.p)
