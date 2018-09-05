![](images/logo5.png)

# 强化学习60天
[英文地址](https://github.com/andri27-ts/60_Days_RL_Challenge)

### 我为了你我设计这个挑战：在这60天里深入学习“深度强化学习”。
你肯定听说过 [Deepmind with AlphaGo Zero](https://www.bilibili.com/video/av29385179) 和
[OpenAI in Dota 2](https://www.bilibili.com/video/av29385428) 取得的惊人成绩！
你难道不想知道他们是如何工作的吗？现在正是你我最终学会“深度强化学习”，并应用到已有项目的时机。

> 终极目标是使用这些多功能的技术，并应用他们到各种重要的真实世界问题中。**Demis Hassabis**

这个项目引导你完成从最基本的到高级的AlphaGo Zero深度强化学习算法。你可以发现**按周组织的主题**和**建议学习资源**。
同时，每周我会提供用Python实现的**应用实例**，帮助你更好地消化理论。

这是原作者的第一个此类型项目，有任何想法，建议或改进都可以联系作者andrea.lonza@gmail.com。

在整个挑战期间，作者将持续更新此项目，请保持关注。

**MLEveryday提示**：以下资源尽可能换成国内可访问网站，并用标签`中文`，`英文字幕`，`英文`等区别。如果有找到中文版，请通过issue反馈。

### 必备知识
* 了解Python和PyTorch
* 了解机器学习
* 了解深度学习（MLP，CNN和RNN）

## 项目（待定）
 - **Q-learning**
 - **DQN**
 - **AC2**
 - **ES**
 - **AlphaGo Zero**

## 第一周 - 强化学习介绍

 - #### `中文`|`bilibili`[强化学习简介(An introduction to Reinforcement Learning)](https://www.bilibili.com/video/av30055826) by Arxiv Insights
 - #### `英文字幕`|`bilibili`[强化学习课程CS294(Introduction and course overview)](https://www.bilibili.com/video/av20957290) by Levine
 - #### `中文`[强化学习：像素乒乓大战(Deep Reinforcement Learning: Pong from Pixels)](http://ju.outofmemory.cn/entry/319445) by Karpathy
 - #### `中文`|`优酷`[强化学习简介(Introduction to Reinforcement Learning)](https://v.youku.com/v_show/id_XMjcwMDQyOTcxMg==.html?spm=a2h0j.11185381.listitem_page1.5!4~A&&f=49376145) - RL by David Silver

## 第二周 - 强化学习基础：马尔可夫决策过程，动态规划与无模型控制

> 忘记过去的人，终将重蹈覆辙。 - **George Santayana**

在这一周，我们将会学习基本的强化学习内容，我们将通过评估和优化表示策略和状态的函数去定义现实世界的各类问题。
----

### 理论材料

 - #### `中文`|`优酷`[马尔科夫决策过程(Markov Decision Process)](https://v.youku.com/v_show/id_XMjcwMDU5ODEyOA==.html?spm=a2h0j.11185381.listitem_page1.5!3~A&&f=49376145) - RL by David Silver
   马尔科夫决策过程定义强化学习问题
   - 马尔科夫过程
   - 马尔科夫决策过程

 - #### `中文`|`优酷`[动态规划设计(Planning by Dynamic Programming)](https://v.youku.com/v_show/id_XMjcwMDY1MDI1Mg==.html?spm=a2h0j.11185381.listitem_page1.5!2~A&&f=49376145) - RL by David Silver
   如何解决马尔科夫决策问题
   - 策略迭代
   - 价值迭代

 - #### `英文`|`优酷`[无模型预测(Model-Free Prediction)](https://v.youku.com/v_show/id_XMjcwMDc2NjYwOA==.html?spm=a2h0j.11185381.listitem_page1.5~A&&f=49376145) - RL by David Silver
   评估无模型马尔科夫决策过程的价值函数
   - 蒙特卡罗学习
   - 时间差分学习
   - TD(λ)

 - #### `英文`|`优酷`[无模型约束(Model-Free Control)](https://v.youku.com/v_show/id_XMjcwNDA5NzIwOA==.html?spm=a2h0j.11185381.listitem_page1.5!5~A&&f=49376145) - RL by David Silver
   优化无模型卡尔科夫决策过程价值函数
   - Ɛ贪婪策略迭代
   - GLIE蒙特卡罗搜索
   - SARSA
   - 重要性采样

----

### 本周项目

[Q-learning解决冰冻湖问题](Week2/frozenlake_Qlearning.ipynb). 在本练习中，你将学会使用SARSA或者Q-learning.

----

#### 想知道更多
- 阅读该书的第3,4,5,6,7章节 [Reinforcement Learning An Introduction - Sutton, Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)


## Week 3 - Value Function Approximation and DQN

## Week 4 - A2C and A3C

## Week 5 - RL in continous space - TRPO/PPO

## Week 6 - Evolution Strategies and Genetic Algorithms

## Week 7 - I2A

## Week 8 - AlphaGoZero + Bonus

## Last 4 days - Review + sharing


## 强化学习论文

## 强化学习资源

:tv: `英文`|`youtube`[Deep Reinforcement Learning](https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3) - UC Berkeley class by Levine, check [here](http://rail.eecs.berkeley.edu/deeprlcourse/) their site.

:tv: `英文`|`youtube`[Reinforcement Learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - by David Silver, DeepMind. Great introductory lectures by Silver, a lead researcher on AlphaGo. They follow the book Reinforcement Learning by Sutton & Barto.

:notebook: [Reinforcement Learning: An Introduction](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262193981/ref=sr_1_2?s=books&ie=UTF8&qid=1535898372&sr=1-2&keywords=reinforcement+learning+sutton) - by Sutton & Barto. The "Bible" of reinforcement learning. [Here](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) you can find the PDF draft of the second version.


## 额外的资源

:books: [Awesome Reinforcement Learning](https://github.com/aikorea/awesome-rl). 强化学习专用资源列表
