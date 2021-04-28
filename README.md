# kuka-reach-drl
Train kuka robot reach a point with deep rl in pybullet.

* **NOTE: The main brach is trained with spinup, and there are some issues with gpu and multi core CPUs at the same time, so this brach will be deprecated in the future. The rllib branch is trained with ray/rllib, and this branch will be mainly used in the future.**
* **The main branch will not update for a while, the rllib brach is the newest**


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


# Resources about deep rl reach and grasp.
## Articles
* [spinningup docs](https://spinningup.openai.com/en/latest/user/installation.html)
* [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-1-actor-critic-method-d53f9afffbf6)(do not carefully read now.)
* [some ray/rllib and other rl problems' blogs](https://www.datahubbs.com/)
* [Action Masking with RLlib](https://towardsdatascience.com/action-masking-with-rllib-5e4bec5e7505)
* [This AI designs beautiful Forest Landscapes for Games!](https://medium.com/deepgamingai/this-ai-designs-beautiful-forest-landscapes-for-games-8675e053636e)
* [Chintan Trivedi's homepage](https://medium.com/@chintan.t93), he writes many blogs about AI and games. It's very recommended.
* [Proximal Policy Optimization Tutorial (Part 1/2: Actor-Critic Method)](https://twitter.com/ericwen5986/status/1374361315100172289)
* [Proximal Policy Optimization Tutorial (Part 2/2: GAE and PPO loss)](https://twitter.com/ericwen5986/status/1374361470859767809)
* [Antonin Raffin](https://araffin.github.io/), he is the member of stable baseline3 project.
* [spinningup using in pybullet envs](https://www.etedal.net/2020/04/pybullet-panda_3.html), this is a blog about how to use spinningup to pybullet envs and use the image as the observation.
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), this is a good blog introducing lstm.

## Source codes
* [robotics-rl-srl](https://github.com/araffin/robotics-rl-srl), S-RL Toolbox: Reinforcement Learning (RL) and State Representation Learning (SRL) for Robotics. In this project, there are CNN policy and instructions how to connect a real robot using deep rl.
* [zenetio/DeepRL-Robotic](https://github.com/zenetio/DeepRL-Robotic), a deep rl project using gazebo.
* [robotology-playground/pybullet-robot-envs](https://github.com/robotology-playground/pybullet-robot-envs), a deep rl project using pybullet, it is built by a company, there are a lot can study from their codes. But their envs do not introduce images.
* [mahyaret/kuka_rl](https://github.com/mahyaret/kuka_rl), a tutorial tells you how to implement DQN and ppo algorithms to kuka robot grasping.
* [AutodeskRoboticsLab/RLRoboticAssembly](https://github.com/AutodeskRoboticsLab/RLRoboticAssembly), a deep rl robot assembly project build by autodesk, it uses rllib and ray.
* [MorvanZhou/train-robot-arm-from-scratch](https://github.com/MorvanZhou/train-robot-arm-from-scratch), a deep rl robot project build by Morvan.
* **[BarisYazici/deep-rl-grasping](https://github.com/BarisYazici/deep-rl-grasping), a deep rl robot grasping project built by a student in Technical University of Munich. He also released his degree's paper, we can learn a lot from his paper.**
* [mahyaret/gym-panda](https://github.com/mahyaret/gym-panda), this is a pybullet panda environment for deep rl. In the codes, author makes the image as the observation.
* [gaoxiaos/Supermariobros-PPO-pytorch](https://github.com/gaoxiaos/Supermariobros-PPO-pytorch), a tutorial about how to implement deep rl to super mario game, the algorithms are modified from spiningup, and the observation is image. So the code is very suitable for image based deep rl.

* [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction), this is the python version code of the book reinforcement learning an introduction second edition, the full book and other resources can be found here [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html).


# Machine learning and reinforcement learning knowledges
* [The Logit and Sigmoid Functions](https://nathanbrixius.wordpress.com/2016/06/04/functions-i-have-known-logit-and-sigmoid/)
* [Generalized Advantage Estimator](https://zhuanlan.zhihu.com/p/139097326)
* [Python浮点算术：争议和限制](https://docs.python.org/zh-cn/3/tutorial/floatingpoint.html)
* [rainbow-is-all-you-need](https://github.com/Curt-Park/rainbow-is-all-you-need)
* [pg-is-all-you-need](https://github.com/MrSyee/pg-is-all-you-need)

# Robotics knowledge
* [回到基础——理解几何旋转与欧拉角](https://robodk.com/cn/blog/%E5%87%A0%E4%BD%95%E6%97%8B%E8%BD%AC%E4%B8%8E%E6%AC%A7%E6%8B%89%E8%A7%92/)

# Python Knowledge
* [Python中的作用域、global与nonlocal](https://note.qidong.name/2017/07/python-legb/)
* [Delgan/loguru](https://github.com/Delgan/loguru), this is a great python log module, it is much greater than python built in logging module.
* [wandb](https://wandb.ai/site),Developer tools for machine learning. Build better models faster with experiment tracking, dataset .
* logging usage
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename='./logs/client1-{}.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
    filemode='w')
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()

stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# in the codes.
# logger.info()
# logger.debug()
```

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

# Some resources about how to implement the RL to real robots
## Source codes
* [kindredresearch/SenseAct](https://github.com/kindredresearch/SenseAct)

## Papers
* [Setting up a Reinforcement Learning Task with a Real-World Robot](https://arxiv.org/pdf/1803.07067.pdf)
* [Real-World Human-Robot Collaborative Reinforcement Learning](https://arxiv.org/pdf/2003.01156.pdf)

## Blogs
* [入门机器人强化学习的一些坑：模拟环境篇](https://zhuanlan.zhihu.com/p/300541709)

## Some comments from Facebook groups
 0|1|2|3
--|--|--|--
![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/rosrl0.jpg)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/rosrl1.jpg)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/rosrl2.jpg)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/rosrl3.jpg)

# VSCode tricks
## [About python extensions](https://zhuanlan.zhihu.com/p/361654489?utm_source=com.miui.notes&utm_medium=social&utm_oi=903420714332332032)

## Resolve a.py in A folder import b.py in B folder
* Add the codes below at the top of a .py file
```python
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append('../')
```
## Add header template in .py files
* Select FIle -> Preference -> User Snippets -> 选择python文件
* Add the codes below
```python

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
		"@Desc    :   None",
		
		"'''",
		"",
		"# here put the import lib",
		"$1"
	],
	}	
}
```

# People in relative projects
* [Yang Guan](https://idthanm.github.io/)

# Details about RL
* 强化学习中的CNN一般没有池化层，池化层会让你获得平移不变性，即网络对图像中对象的位置变得不敏感。这对于 ImageNet 这样的分类任务来说是有意义的，但游戏中位置对潜在的奖励至关重要，我们不希望丢失这些信息。
* 经验回放的动机是：①深度神经网络作为有监督学习模型，要求数据满足独立同分布；②通过强化学习采集的数据之间存在着关联性，利用这些数据进行顺序训练，神经网络表现不稳定，而经验回放可以打破数据间的关联。