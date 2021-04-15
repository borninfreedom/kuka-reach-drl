# kuka-reach-drl
Train kuka robot reach a point with deep rl in pybullet.
<!-- 
![The train process](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/train.gif)![The evaluate process](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/eval.gif)![The average episode reward](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/train_results.png) -->


The train process with mlp|The evaluate process with mlp|train plot
:---------------:|:------------------:|:-------------------------:
![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_train_with_mlp.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_result_with_mlp.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_mlp_train_process.png)

The train process with cnn|The evaluate process with cnn|train plot
:---------------:|:------------------:|:-------------------------:
![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_train_with_cnn.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_result_with_cnn.gif)|


# Installation guide (Now only support linux and macos)
**I strongly recommend using Conda to install the env, because you will possible encounter the mpi4py error with pip.**

The spinningup rl library is the necessary lib.
first, you should install miniconda or anaconda.
second, install some dev dependencies.

```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
sudo apt install libgl1-mesa-glx
```
third, create a conda virtual environment
```bash
conda create -n spinningup python=3.6   #python 3.6 is recommended
```


```bash
#activate the env
conda activate spinningup
```

then, install spiningup,is contains almost dependencies
```bash
# clone my version, I made some changes.
git clone https://github.com/borninfreedom/spinningup.git
cd spinningup
pip install -e .
```

last, install torch and torchvision.

if you have a gpu, please run this (conda will install a correct version of cudatoolkit and cudnn in the virtual env, so don't care which version you have installed in your machine.)
```bash
# CUDA 10.1
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

if you only have a cpu, please run this,
```bash
# CPU Only
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
```

## Alternative installation method
Or, you can create the virtual environment directly through
```bash
conda create --name spinningup --file requirements.txt
```
but I can not ensure this method can success.


# Run instruction

if you want to train the kuka with coordition env, whose input to policy is the coordition of the target pos, and the actor critic framework is based on mlp, please run
```bash
python train_with_mlp.py --is_render  --is_good_view  --cpu 5 --epochs 100
```
if you don't want to view the scene, just train it, run
```bash
python train_with_mlp.py  --cpu 5 --epochs 100
```

if you want to train kuka with image input and cnn model,run
```bash
python train_with_cnn.py --is_render  --is_good_view  --cpu 5 --epochs 500
```
if you don't want to view the scene, just train it, run
```bash
python train_with_cnn.py  --cpu 5 --epochs 500
```


if you want to train kuka with image input and lstm model,run
```bash
python train_with_lstm.py --is_render  --is_good_view  --cpu 5 --epochs 500
```
if you don't want to view the scene, just train it, run
```bash
python train_with_lstm.py --cpu 5 --epochs 500
```




# Files guide
the train.py file is the main train file, you can directly run it or through `python train.py --cpu 6` to run it in terminal. Please notice the parameters.

eval.py file is the evaluate trained model file, the model is in the logs directory named model.pt. In the eval file, pybullet render is open default. **When you want to evaluate my trained model, please change the source code `ac=torch.load("logs/ppo-kuka-reach/ppo-kuka-reach_s0/pyt_save/model.pt")` to `ac=torch.load("saved_model/model.pt")` in `eval.py`**

ppo directory is the main algorithms about ppo.

env directory is the main pybullet env.

## view the train results through plot
```bash
python -m spinup.run plot ./logs
``` 
More detailed information please visit [plotting results](https://spinningup.openai.com/en/latest/user/plotting.html)


# some relative blogs and articles.

1. [spinningup docs](https://spinningup.openai.com/en/latest/user/installation.html)
2. [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6)(do not carefully read now.)
3. [some ray/rllib and other rl problems' blogs](https://www.datahubbs.com/)
4. [Action Masking with RLlib](https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505)
5. [This AI designs beautiful Forest Landscapes for Games!](https://medium.com/deepgamingai/this-ai-designs-beautiful-forest-landscapes-for-games-8675e053636e)
6. [Chintan Trivedi's homepage](https://medium.com/@chintan.t93), he writes many blogs about AI and games. It's very recommended.
7. [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://twitter.com/ericwen5986/status/1374361315100172289)
8. [Proximal Policy Optimization Tutorial (Part 2/2: GAE and PPO loss)](https://twitter.com/ericwen5986/status/1374361470859767809)
9. [Antonin Raffin](https://araffin.github.io/), he is the member of stable baseline3 project.
10. [robotics-rl-srl](https://github.com/araffin/robotics-rl-srl), S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics. In this project, there are CNN policy and instructions how to connect a real robot using deep rl.
11. [zenetio/DeepRL-Robotic](https://github.com/zenetio/DeepRL-Robotic), a deep rl project using gazebo.
12. [robotology-playground/pybullet-robot-envs](https://github.com/robotology-playground/pybullet-robot-envs), a deep rl project using pybullet, it is built by a company, there are a lot can study from their codes. But their envs do not introduce images.
13. [mahyaret/kuka_rl](https://github.com/mahyaret/kuka_rl), a tutorial tells you how to implement DQN and ppo algorithms to kuka robot grasping.
14. [AutodeskRoboticsLab/RLRoboticAssembly](https://github.com/AutodeskRoboticsLab/RLRoboticAssembly), a deep rl robot assembly project build by autodesk, it uses rllib and ray.
15. [MorvanZhou/train-robot-arm-from-scratch](https://github.com/MorvanZhou/train-robot-arm-from-scratch), a deep rl robot project build by Morvan.
16. [BarisYazici/deep-rl-grasping](https://github.com/BarisYazici/deep-rl-grasping), a deep rl robot grasping project built by a student in Technical University of Munich. He also released his degree's paper, we can learn a lot from his paper.
17. [spinningup using in pybullet envs](https://www.etedal.net/2020/04/pybullet-panda_3.html), this is a blog about how to use spinningup to pybullet envs and use the image as the observation.
18. [mahyaret/gym-panda](https://github.com/mahyaret/gym-panda), this is a pybullet panda environment for deep rl. In the codes, author makes the image as the observation.
19. [gaoxiaos/Supermariobros-PPO-pytorch](https://github.com/gaoxiaos/Supermariobros-PPO-pytorch), a tutorial about how to implement deep rl to super mario game, the algorithms are modified from spiningup, and the observation is image. So the code is very suitable for image based deep rl.
20. [MrSyee/pg-is-all-you-need](https://github.com/MrSyee/pg-is-all-you-need)
21. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), this is a good blog introducing lstm.
22. [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction), this is the python version code of the book reinforcement learning an introduction second edition, the full book and other resources can be found here [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).


# Machine learning and reinforcement learning knowledges
* [The Logit and Sigmoid Functions](https://nathanbrixius.wordpress.com/2016/06/04/functions-i-have-known-logit-and-sigmoid/)
* [Generalized Advantage Estimator](https://zhuanlan.zhihu.com/p/139097326)

# Blogs about deep rl written by me
1. [Ubuntu助手 — 一键自动安装软件，一键进行系统配置](https://www.guyuehome.com/15107)
2. [深度强化学习专栏 —— 1.研究现状](https://www.guyuehome.com/21403)
3. [深度强化学习专栏 —— 2.手撕DQN算法实现CartPole控制](https://www.guyuehome.com/22329)
4. [深度强化学习专栏 —— 3.实现一阶倒立摆](https://www.guyuehome.com/23439)
5. [深度强化学习专栏 —— 4. 使用ray做分布式计算](https://www.guyuehome.com/?p=26243)
6. [深度强化学习专栏 —— 5. 使用ray的tune组件优化强化学习算法的超参数](https://www.guyuehome.com/?p=26247)
7. [深度强化学习专栏 —— 6. 使用RLLib和ray进行强化学习训练](https://www.guyuehome.com/26251)
8. [深度强化学习专栏 —— 7. 实现机械臂reach某点之PPO算法实现（一）](https://www.guyuehome.com/33650)
9. [深度强化学习专栏 —— 8. 实现机械臂reach某点之PPO算法实现（二）](https://www.guyuehome.com/33663)
10. [深度强化学习专栏 —— 9. 实现机械臂reach某点之PPO算法实现（三）](https://www.guyuehome.com/33682)
11. [深度强化学习专栏 —— 10. 实现机械臂reach某点之环境实现实现](https://www.guyuehome.com/33691)

# Blogs about pybullet written by me
1. [pybullet杂谈 ：使用深度学习拟合相机坐标系与世界坐标系坐标变换关系（一）](https://www.guyuehome.com/24528)
2. [pybullet杂谈 ：使用深度学习拟合相机坐标系与世界坐标系坐标变换关系（二）](https://www.guyuehome.com/26255)
3. [pybullet电机控制总结](https://blog.csdn.net/bornfree5511/article/details/108188632)
4. [Part 1 - 自定义gym环境](https://blog.csdn.net/bornfree5511/article/details/108212687)
5. [Part 1.1 - 注册自定义Gym环境](https://blog.csdn.net/bornfree5511/article/details/108212963)
6. [Part 1.2 - 实现一个井字棋游戏的gym环境](https://blog.csdn.net/bornfree5511/article/details/108214740)
7. [Part 1.3 - 熟悉PyBullet](https://blog.csdn.net/bornfree5511/article/details/108307638)
8. [Part 1.4 - 为PyBullet创建Gym环境](https://blog.csdn.net/bornfree5511/article/details/108326084)

# VSCode tricks
* [python extensions](https://zhuanlan.zhihu.com/p/361654489?utm_source=com.miui.notes&utm_medium=social&utm_oi=903420714332332032)
* Resolve a.py in A folder import b.py in B folder
Add the codes below at the top of a .py file
```python
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append('../')
```
* Add header template in .py files
```python
# Select FIle -> Preference -> User Snippets -> 选择python文件
# Add the codes below
{
	// Place your snippets for python here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	// "Print to console": {
	// 	"prefix": "log",
	// 	"body": [
	// 		"console.log('$1');",
	// 		"$2"
	// 	],
	// 	"description": "Log output to console"
	// }


	
	"HEADER":{
		"prefix": "header",
		"body": [
		"#!/usr/bin/env python3",
		"# -*- encoding: utf-8 -*-",
		"'''",
		"@File    :   $TM_FILENAME",
		"@Time    :   $CURRENT_YEAR/$CURRENT_MONTH/$CURRENT_DATE $CURRENT_HOUR:$CURRENT_MINUTE:$CURRENT_SECOND",
		"@Author  :   Yan Wen ",
		"@Version :   1.0",
		"@Contact :   z19040042@s.upc.edu.cn",
		"@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA",
		"@Desc    :   None",
		
		"'''",
		"",
		"# here put the import lib",
		"$1"
	],
	}
		
	
	
}
```
