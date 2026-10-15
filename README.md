# Adaptive Orchestration of Service Function Chains in SAGIN-MEC via Graph Reinforcement Learning

**简体中文** | [English](README_EN.md)

本仓库是论文 **“Adaptive Orchestration of Service Function Chains in SAGIN-MEC via Graph Reinforcement Learning”** 的代码实现。项目结合图强化学习（GRL）和近端策略优化（PPO），在空天地一体化网络（SAGIN）场景中进行服务功能链（SFC）编排。

## 环境依赖

- Python>=3.10
- PyTorch、NumPy、NetworkX、pandas、PyYAML、Matplotlib

以下命令均在项目根目录执行。物理网络拓扑位于 `dataset/topology/network.gpickle`；SFC 请求在运行时随机生成。

## 训练

按需修改 `train.yml`，然后运行：

```bash
python train.py
```

主要配置项如下：

| 配置项 | 说明 |
| --- | --- |
| `num_episode` | 训练轮数 |
| `total_time_steps` | 每轮的时间槽数 |
| `arrival_rate` | 每个时间槽生成 SFC 请求的概率 |
| `net_path` | 物理网络拓扑文件路径 |
| `model` | 模型名称，目前支持 `GCN_PPO` |
| `GCN_PPO` | 学习率、批量大小等 PPO 参数 |

训练输出保存在 `results/GCN_PPO/<timestamp>/`：

```text
results/GCN_PPO/<timestamp>/
├── config.yml                 # 本次运行的配置
├── model/
│   ├── actor.pth              # 按每轮奖励选出的最佳策略网络
│   └── critic.pth             # 对应的价值网络
└── train/
    └── <timestamp>.csv        # 每轮训练指标
```

## 测试

将训练产生的时间戳替换到下方命令中的 `YYYYMMDD_HHMMSS`：

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train
```

默认测试使用该次训练保存在 `config.yml` 中的请求到达率和时间槽数。通过 `--a` 可指定测试到达率：

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train --a 0.3
```

通过 `--t 2` 可启用每 500 个时间槽统计一次的测试模式：

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train --t 2
```

该模式运行 10,000 个时间槽；请求到达率默认从 `0.1` 开始，也可用 `--a` 设置初始值，之后每 500 个时间槽增加 `0.05`。测试结果保存在 `results/GCN_PPO/<timestamp>/test/`：常规测试生成 `<timestamp>.csv`，分段统计模式生成 `interval_results_<timestamp>.csv`。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `model/` | GCN 策略网络、价值网络及 PPO 实现 |
| `controller/` | SFC 节点映射、链路路由和资源约束处理 |
| `environment/` | 物理网络、虚拟网络、仿真环境和评估 |
| `evolution/` | 训练、测试流程及指标记录 |
| `utils/` | SFC 请求生成和组件选择 |
| `dataset/topology/` | 仿真使用的物理网络拓扑 |
| `train.py`、`test.py`、`train.yml` | 运行入口及训练配置 |
