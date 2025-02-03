import numpy as np
import sys
sys.path.append('..')
from SimpleSAC.envs import get_isaac_env


def test_isaac_env():
    env_name: str = 'WheelLegged'
    backend = 'torch'
    sim_params = {'use_gpu_pipeline': True}
    task_args = {'device': 'cuda'}
    env = get_isaac_env(
        env_name=env_name,
        backend=backend,
        sim_params=sim_params,
        task_args=task_args,
    )
    s = env.reset()
    print(s)
    assert s.shape == (1, 4)
    a = np.random.random((1,))
    s, _, _, _ = env.step(a)
    assert s.shape == (1, 4)
    env.close()


if __name__ == '__main__':
    test_isaac_env()