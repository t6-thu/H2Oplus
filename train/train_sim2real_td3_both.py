'''
delta_r on both real and sim data
'''

import argparse
import sys
import time, datetime

import numpy as np

sys.path.append("..")

from algos.TD3_sim2real_both import Sim2real_TD3
from utils.envs import *

import wandb


def main():
    wandb.init(project="when-to-trust-your-simulator", entity="t6-thu")
    nowTime = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # Parameters
    parser = argparse.ArgumentParser(description='Solve the Mujoco locomotion tasks with TD3')
    parser.add_argument('--current_time', default=nowTime, type=str, help='Current system time at the start.')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--env_list', default='HalfCheetah-v2', type=str, help='choose avaliable mujoco env, seperated by \';\'.')
    parser.add_argument('--data_source', default='medium_replay', help='where the fixed real dataset comes from')
    parser.add_argument('--start_steps', default=25e3, type=int)
    parser.add_argument('--unreal_dynamics', default="gravity", type=str, help="Customize env mismatch degree as you like.""If you want to serially run multiple tasks, separate them by \';\'.")
    parser.add_argument('--variety_list', default="2.0", type=str, help="Customize env mismatch degree as you like.""If you want to serially run multiple tasks, separate them by \';\'.")
    parser.add_argument('--alpha', default=1.0, type=float, help="reward correction coefficient")
    args = parser.parse_args()
    wandb.config.update(args)

    wandb.run.name = f"{args.env_list}_{args.data_source}_{args.unreal_dynamics}x{args.variety_list}_{args.current_time}"

    # different unreal dynamics properties: gravity; density; friction
    for unreal_dynamics in args.unreal_dynamics.split(";"):
        # different environment: Walker2d-v2, Hopper-v2, HalfCheetah-v2
        for env_name in args.env_list.split(";"):
            args.env_name = env_name
            real_env = get_new_gravity_env(1, args.env_name)
            # different varieties: 0.5, 1.5, 2.0
            for variety_degree in args.variety_list.split(";"):
                variety_degree = float(variety_degree)

                args.variety_degree = variety_degree

                if unreal_dynamics == "gravity":
                    sim_env = get_new_gravity_env(variety_degree, args.env_name)
                elif unreal_dynamics == "density":
                    sim_env = get_new_density_env(variety_degree, args.env_name)
                elif unreal_dynamics == "friction":
                    sim_env = get_new_friction_env(variety_degree, args.env_name)
                else:
                    raise RuntimeError("Got error unreal dynamics %s" % unreal_dynamics)
                    
                generate_log("\n-------------Env name: {}, variety: {}, unreal_dynamics: {}-------------".format(env_name, variety_degree, unreal_dynamics))

                agent_TD3 = Sim2real_TD3(sim_env=sim_env, 
                real_env_name=args.env_name,
                real_env=real_env, 
                data_source=args.data_source, 
                device=args.device, 
                start_steps=args.start_steps, 
                alpha=args.alpha)

                agent_TD3.learn(total_time_step=int(1e+6))


if __name__ == '__main__':
    main()
