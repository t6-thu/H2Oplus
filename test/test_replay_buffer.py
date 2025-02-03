import numpy as np
import sys
sys.path.append('..')
from SimpleSAC.mixed_replay_buffer import NewReplayBuffer


def test_new_replay_buffer():
    file_path = 'temp/5.11.s'
    env_name = 'WheelLegged'
    reward_scale = 1.0
    reward_bias = 0.1
    clip_action = 1.0
    state_dim = 4
    action_dim = 1
    rb = NewReplayBuffer(
        file_path=file_path,
        env_name=env_name,
        reward_scale=reward_scale,
        reward_bias=reward_bias,
        clip_action=clip_action,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    _ = rb.sample(64)


if __name__ == '__main__':
    test_new_replay_buffer()
