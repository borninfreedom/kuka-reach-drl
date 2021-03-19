# kuka-reach-drl
Train kuka robot reach a point with deep rl in pybullet.

* **I strongly recommend using Conda to install the env, because you will possible encounter the mpi4py error with pip.**
* The spinningup rl library is the necessary lib.

### Installation guide (Now only support linux and macos)
first, you should install miniconda or anaconda.
second, install some dev dependencies.
```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
third, create a conda virtual environment
```bash
conda create -n spinningup python=3.6   #python 3.6 is recommended
```
```bash
#activate the env
conda activate spinningup
```

last, install spiningup,is contains almost dependencies
```bash
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```

#### alternative installation method
Or, you can create the virtual environment directly through
```bash
conda create --name spinningup --file requirements.txt
```
but I can not ensure this method can success.

### some relative blogs and articles.

1. [spinningup docs](https://spinningup.openai.com/en/latest/user/installation.html)
