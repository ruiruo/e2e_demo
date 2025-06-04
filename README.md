# Trajectory Generator

一个基于 Transformer 与 拓扑编码器（topology-encoder） 的端到端轨迹生成框架demo。
框架同时支持 监督学习（离散化航点上的 token-level 交叉熵）以及使用 近端策略优化（Proximal Policy Optimisation, PPO） 的
on-policy 微调。

---

## Features

* **轨迹离散化Token化** – 将二维航点映射到紧凑词表，使问题转化为语言模型式的序列生成。
* **上下文感知编码器** – 将自车状态、静态地图与动态邻居信息通过可学习编码器融合。
* **Transformer解码器** – 自回归地在已编码上下文基础上预测下一个token。
* **额外评估指标** – L2误差、Hausdorff距离、傅里叶距离、带宽/面积等。
* **TeacherForcing&ScheduledSampling** – 训练阶段可在强制教学和逐步放宽之间平滑切换。

---

## Quick Start

### 1. Installation

```bash
# create env
conda create -n trajgen python=3.10 -y
conda activate trajgen

# install core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Data preparation

* Data: example data here:  XXXX
* Create tokenisation lookup tables:

```bash
python utils/trajectory_utils.py --build-vocab \
       --data_dir DATA_ROOT \
       --out_dir  vocab/
```

* Update the config file

### 3. Supervised training

```bash
python train.py --config configs/training.yaml
```

### 4. PPO fine‑tuning

```bash
python train.py --config configs/ppo.yaml
```

This uses `TrajectoryGeneratorPPOModule` and plugs in reward signals defined in
`utils/rewards.py`.

### 5. Inference

```bash
python predict.py --config configs/predict.yaml --ckpt path/to/best.ckpt
```

---

## Evaluation

```bash
python -m utils.eval \
    --pred outputs/sample.jsonl \
    --gt   data/val.jsonl
```

Yields metrics identical to those used during validation:

| 指标                 | 含义             |
|--------------------|----------------|
| `l2`               | 点对点L2距离均值（米）   |
| `hausdorff`        | Hausdorff距离（米） |
| `fourier`          | 傅里叶距离（米）       |
| `strip_area`       | 轨迹覆盖带面积（m²）    |
| `strip_width_mean` | 覆盖带平均宽度（米）     |

---

## Reference

| Module                                         | Paper                                                                                                                           | Notes                                        |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| Transformer architecture & positional encoding | Vaswani et al., 2017, *Attention Is All You Need*                                                                               | Backbone & sinusoidal positions              |
| BERT encoder                                   | Devlin et al., 2019, *BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding*                         | Bidirectional pre‑training inspiration       |
| Autoregressive decoding / multitask            | Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*                                                     | BOS→…→EOS sampling paradigm                  |
| Conditional normalisation (FiLM)               | Pérez et al., 2018, *FiLM: Visual Reasoning with a General Conditioning Layer*                                                  | Dynamic feature scaling/shifting             |
| Tokenised trajectories                         | Lee et al., 2024, *TGT: Tokenized Generative Trajectory Prediction*                                                             | Continuous → discrete waypoint tokens        |
| Trajectory Transformer                         | Sun et al., 2022, *Trajectory Transformer for Autonomous Driving*                                                               | Social attention & autoregressive generation |
| Context & goal encoder                         | Shah et al., 2024, *ViNT: A Foundation Model for Visual Navigation*                                                             | Vision/map/goal fusion                       |
| Scaling laws                                   | Kaplan et al., 2020, *Scaling Laws for Neural Language Models*                                                                  | Predictability across model/data/compute     |
| Top‑p / Top‑k sampling                         | Holtzman et al., 2020, *The Curious Case of Neural Text Degeneration*; Fan et al., 2018, *Hierarchical Neural Story Generation* | Sampling truncation/filtration               |
| Scheduled sampling                             | Bengio et al., 2015, *Scheduled Sampling for Sequence‑to‑Sequence Learning*                                                     | Gradual teacher‑forcing annealing            |
| Fourier trajectory distance                    | Lamb et al., 2019, *Fourier Distance for Trajectories*                                                                          | Frequency‑domain similarity metric           |
| Simulator — highway‑env                        | Leurent, 2019, *An Environment for Autonomous Driving Decision‑Making*                                                          | Gym‑based driving environment                |
| PPO optimisation                               | Schulman et al., 2017, *Proximal Policy Optimization Algorithms*                                                                | On‑policy updates                            |
| Distributed RL framework                       | Liang & Moritz et al., 2018, *RLlib: Abstractions for Distributed Reinforcement Learning*                                       | Scalable RL on Ray                           |
| Soft Actor‑Critic                              | Haarnoja et al., 2018, *Soft Actor‑Critic: Off‑Policy Maximum‑Entropy RL*                                                       | Off‑policy actor‑critic                      |
| Prioritised Experience Replay                  | Schaul et al., 2016, *Prioritized Experience Replay*                                                                            | TD‑error‑based sampling                      |
| Hindsight Experience Replay                    | Andrychowicz et al., 2017, *Hindsight Experience Replay*                                                                        | Goal relabelling                             |
| Decision Transformer                           | Chen et al., 2021, *Decision Transformer: Offline RL with Sequence Models*                                                      | Return‑to‑go conditioning                    |
| Gymnasium interface                            | Brockman et al., 2016, *The OpenAI Gym*                                                                                         | Standard RL API                              |