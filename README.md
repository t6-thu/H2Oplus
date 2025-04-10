# H2O+
[H2O+](https://arxiv.org/abs/2309.12716) is An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps, which offers great flexibility to bridge various choices of offline and online learning methods, while also accounting for dynamics gaps between the real and simulation environment. Compared to [H2O](https://arxiv.org/abs/2206.13464), H2O+ gets rid of over-conservative offline RL backbone and enjoys explorative benefits of simulation environments. Through extensive simulation and real-world robotics experiments, we demonstrate superior performance and flexibility over advanced cross-domain online and offline RL algorithms, details of which can be seen in our [webpage](https://sites.google.com/view/h2oplusauthors/).


## Installation and Setups
To install the dependencies, run the command:
```python
    pip install -r requirements.txt
```
Add this repo directory to your `PYTHONPATH` environment variable:
```
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Wheel-legged Robot Experiments with Issac Sim
Practitioner can rewrite scripts about the task (`SimpleSAC/wheel_legged_task.py`), environment (`SimpleSAC/envs.py`), and training algorithms for Issac Sim wheel-legged robot at their use, according to the examples in `SimpleSAC`. Note that some dependencies are in [D2C](https://github.com/AIR-DI/D2C) repository.

## Run Benchmark Experiments
We benchmark H2O+ and its baselines on MuJoCo simulation environment and D4RL datasets. To begin, enter the folder `SimpleSAC`:
```
    cd SimpleSAC
```
Then you can run H2O+ experiments using the following example commands.
### Simulated in HalfCheetah-v2 with 2x gravity and Medium Replay dataset
```python
    python drh2o_main.py \
        --env_list HalfCheetah-v2 \
        --data_source medium_replay \
        --unreal_dynamics gravity \
        --variety_list 2.0 
```
### Simulated in Walker-v2 with .3x friction and Medium Replay dataset
```python
    python drh2o_main.py \
        --env_list Walker-v2 \
        --data_source medium_replay \
        --unreal_dynamics friction \
        --variety_list 0.3 
```
### Simulated in HalfCheetah-v2 with joint noise N(0,1) and Medium dataset
```python
    python drh2o_main.py \
        --env_list HalfCheetah-v2 \
        --data_source medium \
        --variety_list 1.0 \
        --joint_noise_std 1.0 
```

## Visualization of Learning Curves
You can resort to [wandb](https://wandb.ai/site) to login your personal account with your wandb API key.
```
    export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run `wandb online` to turn on the online syncronization.

## Citation
If you are using H2O+ framework or code for your project development, please cite the following paper:
```
@inproceedings{
    niu2025h2o+,
    title={H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps},
    author={Haoyi Niu and Tianying Ji and Bingqi Liu and Haocheng Zhao and Xiangyu Zhu and Jianying Zheng and Pengfei Huang and Guyue Zhou and Jianming HU and Xianyuan Zhan},
    booktitle={IEEE International Conference on Robotics and Automation},
    year={2025}
}
```
