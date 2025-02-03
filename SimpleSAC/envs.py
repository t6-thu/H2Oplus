'''
  Generate different type of dynamics mismatch.
  @python version : 3.6.4
'''

import gym
from gym.spaces import Box, Discrete, Tuple
# from SimpleSAC.utils import update_target_env_gravity, update_target_env_density, update_target_env_friction, update_source_env
from utils import update_target_env_gravity, update_target_env_density, update_target_env_friction, update_target_env_thigh_range, update_target_env_foot_shape, update_source_env, update_target_env_soft_foot, update_target_env_short_thigh, update_target_env_ellipsoid_limb, update_target_env_box_limb, update_target_env_head_size, update_target_env_torso_length, update_target_env_soft_limb, update_target_env_tendon_elasticity
try:
    from d2c.envs.external import IsaacGym
except:
    print(f'[Warning]: Please install d2c first!')


def get_new_gravity_env(variety, env_name):
    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)

    return env


def get_source_env(env_name="Walker2d-v2"):
    update_source_env(env_name)
    env = gym.make(env_name)

    return env


def get_new_density_env(variety, env_name):
    update_target_env_density(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_thigh_range_env(variety, env_name):
    update_target_env_thigh_range(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_friction_env(variety, env_name):
    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_foot_shape_env(env_name):
    update_target_env_foot_shape(env_name)
    env = gym.make(env_name)

    return env

def get_new_foot_stiffness_env(variety, env_name):
    update_target_env_soft_foot(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_limb_stiffness_env(variety, env_name):
    update_target_env_soft_limb(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_tendon_elasticity_env(variety, env_name):
    update_target_env_tendon_elasticity(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_thigh_size_env(variety, env_name):
    update_target_env_short_thigh(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_ellipsoid_limb_env(env_name):
    update_target_env_ellipsoid_limb(env_name)
    env = gym.make(env_name)

    return env

def get_new_box_limb_env(env_name):
    update_target_env_box_limb(env_name)
    env = gym.make(env_name)

    return env

def get_new_head_size_env(variety, env_name):
    update_target_env_head_size(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_torso_length_env(variety, env_name):
    update_target_env_torso_length(variety, env_name)
    env = gym.make(env_name)

    return env

def get_isaac_env(env_name: str, **env_args):
    if 'task_args' in env_args:
        task_args = env_args.pop('task_args')
        return IsaacGym(env_name, **env_args, **task_args)
    else:
        return IsaacGym(env_name, **env_args)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
