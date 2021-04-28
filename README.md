# kuka-reach-drl
Train kuka robot reach a point with deep rl in pybullet.

* **NOTE: The main brach is trained with spinup, and there are some issues with gpu and multi core CPUs at the same time, so this brach will be deprecated in the future. The rllib branch is trained with ray/rllib, and this branch will be mainly used in the future.**

The train process with mlp|The evaluate process with mlp|train plot
:---------------:|:------------------:|:-------------------------:
![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_train_with_mlp.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_result_with_mlp.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_mlp_train_process.png)

The train process with cnn|The evaluate process with cnn|train plot
:---------------:|:------------------:|:-------------------------:
![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_train_with_cnn.gif)|![](https://github.com/borninfreedom/kuka-reach-drl/blob/main/pictures/reach_result_with_cnn.gif)|


# Installation guide (Now only support linux and macos)
* I recommend using conda to install, because conda can help you install cudatoolkit and cudnn, so you don't need to care which version you have installed in your machine, conda helps you install them in an isolated environment.

```bash
pip install ray,ray[rllib],ray[tune]
conda install pytorch-gpu
pip install pybullet
```

# Run instruction



# Wiki
## [Reinforcement Learning](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/Reinforcement%20Learning.md)
## [pybullet](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/pybullet.md)
## [robotics](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/robotics.md)
## [deep learning](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/deep_learning.md)
## [python](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/python.md)
## [vscode](https://github.com/borninfreedom/kuka-reach-drl/blob/rllib/docs/vscode.md)




