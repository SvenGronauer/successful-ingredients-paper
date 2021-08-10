# Successful Ingredients of Policy Gradient Algorithms

This repository contains source code used to produce the results reported in:

```
The Successful Ingredients of Policy Gradient Algorithms
Sven Gronauer, Martin Gottwald, Klaus Diepold
Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021
```

For the technical appendix, please see:
[Technical Appendix PDF](https://github.com/SvenGronauer/successful-ingredients-paper/blob/master/ijcai_21_technical_appendix.pdf)

### Operating Systems

We tested this repository under MacOSX (Catalina) and Linux Ubuntu (20.04 LTS)
with Python 3.7 and 3.8.
We cannot guarantee that this repository works under deviating distributions or versions.

Major dependencies:
+ PyBullet (pybullet==3.0.6)
+ PyTorch (torch==1.6.0)

### Benchmarked Environments
Locomotion
+ HalfCheetahBulletEnv-v0
+ HopperBulletEnv-v0
+ AntBulletEnv-v0
+ Walker2DBulletEnv-v0
+ HumanoidBulletEnv-v0

Manipulation
+ ReacherBulletEnv-v0
+ PusherBulletEnv-v0
+ KukaBulletEnv-v0


### Comments on Hardware

The default setup is that each algorithm run is executed in a single thread 
(single learner setup). For each policy iteration, a batch of 32k transition 
samples is collected, which are then used for policy updates.
We use Threadripper 3970X (64 threads) and 3990X CPUs (128 threads) which offer high amount of parallel processing.


# Quick Start

### Installation

```
git clone https://github.com/SvenGronauer/successful-ingredients-paper.git
cd successful-ingredients-paper
pip install -e .
```

### Usage

```
    python -m sipga.train --alg ALG --env ENV
```

where for ALG you can choose between [iwpg, ppo, trpo, npg]
and for ENV any child class of OpenAI's gym.Env 