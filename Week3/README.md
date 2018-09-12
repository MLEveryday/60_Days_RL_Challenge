# DQN，Double Q-learning，Deuling Networks，Multi-step learning和Noisy Nets在Pong的应用

本周将Deep Q-Networks (DQN)应用到[Pong](https://gym.openai.com/envs/Pong-v0/).

![Pong Gif](imgs/pong_gif.gif)

我大部分按照[Mnih et al.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)进行DQN实现和超参数选取。（最后一页有个表格汇总所有超参数。）

为了使项目更有趣，我改进了基本DQN, 实现了一些变型，如 **Double Q-learning**, **Dueling networks**, **Multi-step learning** and **Noisy Nets**。可以在[Hessel et al.](https://arxiv.org/pdf/1710.02298.pdf)找到他们。

### [学习理论](../README.md)

---

### Double Q-learning - [论文](https://arxiv.org/pdf/1509.06461.pdf)

使传统Q-learning的过高估计偏差最小化。

<img src="imgs/double_Qlearning_formula.png" alt="drawing" width="400"/>

为使用它，在*main.py*进行如下设置：
```python
DQN_HYPERPARAMS = {
    'double_DQN': True,
    ...
}
```

---

### 竞争网络结构(Dueling networks) - [论文](http://proceedings.mlr.press/v48/wangf16.pdf)

它使用2个不同的神经网络：一个输出状态值(the value of the state)，另一个输出每个动作的优势(the advantage of each action)。 
这2个网络共享卷积编码器(convolutional encoder)。

<img src="imgs/Dueling_img.png" alt="drawing" width="400"/>

为使用它，在*main.py*进行如下设置：
```python
DQN_HYPERPARAMS = {
    'dueling': True,
    ...
}
```

---

### 嘈杂网络(NoisyNet) - [论文](https://arxiv.org/pdf/1706.10295.pdf)

为克服ε-greedy limitations，引入噪声线性层(noise linear layers)。网络会管理噪声流(noise stream)，平衡 The network will manage the noise stream to balance the exploration.

<img src="imgs/noisenet_formula.png" alt="drawing" width="400"/>

为使用它，在*main.py*进行如下设置：
```python
DQN_HYPERPARAMS = {
    'noisy_net': True,
    ...
}
```

---

### Multi-step

Introduce a forward-view multi-step. Similar to TD(λ)

<img src="imgs/multistep_formula.png" alt="drawing" width="350"/>


To use it, in *main.py*, set
```python
DQN_HYPERPARAMS = {
    'n_multi_step': 2, # or 3
    ...
}
```

NB: From today's on, because we will train deep neural networks, I suggest to run the code on GPUs. If you don't have it, you can use [Google Colab](https://colab.research.google.com/).
Also, to track the networks' results, we'll use [TensorboardX](https://github.com/lanpa/tensorboardX) (tensorboard for PyTorch). In case you use Google Colab to run TensorBoard on your pc, execute the commands in the section below.

NB: If you use GPUs remember to change DEVICE from 'cpu' to 'cuda' in *main.py*.


## 为使代码更清晰，分成6个文件:
 - **main.py** 包含主题。它创建代理，环境，并操作游戏。每一股，更新代理。contains the main body. It creates the agent, the environment and plays N games. For each step, it updates the agent
 - **agents.py** 包含代理类，负责管理核心控制，重复操作缓冲和基本的功能。has the Agent class that control the central control, the replay buffer and basic functions
 - **central_control.py** 包含核心控制类，负责初始化DQN（包括它的变型），优化，计算损失等。，contains CentralControl class. It is responsible to instantiate the DQN (or its variants), optimize it, calculate the loss ecc..
 - **buffers.py** 包含类，保存代理记忆的双端序列，并从中采样。contains the ReplayBuffer class to keep the agent's memories inside a deque list and sample from it.
 - **neural_net.py** 包含DQN等代理的深度神经网络。contains the deep neural nets for the agent namely DQN, DuelingDQN and a NoisyLinear Layer for the Noisy DQN.
 - **atari_wrappers.py** 包含atari封装。include some Atari wrappers. https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
 - **utils.py**, 包含测试方程。for now, contains only a testing function.


## Results

In the image below are shown the rewards mean of the last 10 games and the last 40 games for three different DQN variations.
The x-axis is the number of games. You can see that only 120 games are enough to learn the game pretty well.

![results](imgs/DQN_variations.png)

- ![#00ac77](https://placehold.it/15/00ac77/000000?text=+) `Basic DQN`
- ![#628ced](https://placehold.it/15/628ced/000000?text=+) `2-step DQN`
- ![#df1515](https://placehold.it/15/df1515/000000?text=+) `2-step Dueling DQN`

May seem strange that 2-step Dueling DQN performs worst than 2-step DQN but it's important to keep in mind that the NNs are stochastic and that I tested only on one game. The authors of the DuelingDQN paper, reported better results when applied to other games.


## Install

```
!pip install gym
!pip install torch torchvision
!pip install tensorboardX
!apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

Install gym
```
!git clone https://github.com/openai/gym.git
import os
os.chdir('gym')
!ls
!pip install -e .
os.chdir('..')
```

Install gym
```
!pip install gym[atari]
```


## To run TensorBoard in Google Colab

Instructions from https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/

Download and install ngrok
```
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
```

run ngrok and tensorboard
```
LOG_DIR = 'content/runs'

get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```
