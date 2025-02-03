'''
Utils functions and some configs.
@python version : 3.6.8
'''

import os, re, copy, time, random, datetime, argparse
import numpy as np


# nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')
# parser = argparse.ArgumentParser(description="Process running arguments")

# # tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# # hype parameter for PPO training
# hype_parameters = {
#     "gamma": 0.99,
#     "lamda": 0.95,
#     "batch_size": 512,
#     "epoch_num": 10,
#     "clip_value": 0.2,
#     "c_1": 3,
#     "c_2": 0.001,
#     "init_lr": 3e-4,
#     "d_lr": 3e-4,
#     "lr_epsilon": 1e-6
# }

# # Algorithms running configuration parameters. Add argument if needed.
# parser = argparse.ArgumentParser(description='Solve the Mujoco locomotion tasks with TD3')
# parser.add_argument('--current_time', default=nowTime, type=str, help='Current system time at the start.')
# parser.add_argument('--device', default='cuda', help='cuda or cpu')
# parser.add_argument('--env_list', default='HalfCheetah-v2', type=str, help='choose avaliable mujoco env, seperated by \';\'.')
# parser.add_argument('--data_source', default='medium_replay', help='where the fixed real dataset comes from')
# parser.add_argument('--start_steps', default=25e3, type=int)
# parser.add_argument('--unreal_dynamics', default="gravity", type=str, help="Customize env mismatch degree as you like.""If you want to serially run multiple tasks, separate them by \';\'.")
# parser.add_argument('--variety_list', default="2.0", type=str, help="Customize env mismatch degree as you like.""If you want to serially run multiple tasks, separate them by \';\'.")
# parser.add_argument('--alpha', default=1.0, type=float, help="reward correction coefficient")
# parser.add_argument('--sim_only', default=0, type=int, help="Ablation Study: Mixed training data stream V.S. Simulated-only data stream")
# parser.add_argument('--penalize_sim', default=1, type=int, help="Ablation Study: Reward penalty on sim data V.S. No reward penalty on sim data")
# parser.add_argument('--sim_real_ratio', default=1.0, type=float, help="Ablation Study: Different data stream with sim/real ratio")
# parser.add_argument('--learning_steps', default=1e+6, type=int, help="Total learning iterations")
# parser.add_argument('--sim_warmup', default=0.0, type=float, help="The sim-only warmup phase occupies how much ratio of total learning steps. Attention:[0,1]")
# parser.add_argument('--vg',
#                     default="0", type=str,
#                     help='Visible gpus.')
# parser.add_argument('--log_index',
#                     default=nowTime, type=str,
#                     help='Current system time for creating files.')

# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.vg

# generate xml assets path: gym_xml_path
def generate_xml_path():
    import gym, os
    xml_path = os.path.join(gym.__file__[:-11], 'envs/mujoco/assets')

    assert os.path.exists(xml_path)
    print("gym_xml_path: ",xml_path)

    return xml_path


gym_xml_path = generate_xml_path()


def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileExistsError:
        pass

    return path


def update_xml(index, env_name):
    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    else:
        raise RuntimeError("No available environment named \'%s\'" % env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)

#TODO: gravity
def update_target_env_gravity(variety_degree, env_name):
    old_xml_name = parse_xml_name(env_name)
    # create new xml 
    xml_name = "{}_gravityx{}.xml".format(old_xml_name.split(".")[0], variety_degree)

    with open('../xml_path/source_file/{}'.format(old_xml_name), "r+") as f:

        new_f = open('../xml_path/target_file/{}'.format(xml_name), "w+")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                gravity_list = a[0].split(" ")
                new_gravity_list = []
                for num in gravity_list:
                    new_gravity_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_gravity_list)
                replace_num = "gravity=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    # replace the default gym env with newly-revised env
    os.system(
        'cp ../xml_path/target_file/{0} {1}/{2}'.format(xml_name, gym_xml_path, old_xml_name))

    time.sleep(0.2)

#TODO: density
def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('../xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('../xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ../xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)

#TODO: friction
def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('../xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('../xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree * float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\"" + replace_num + "\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ../xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.2)


def generate_log(extra=None):
    print(extra)
    record_data('../documents/log_{}.txt'.format(args.log_index), "{}".format(extra))


# def get_gaes(rewards, v_preds, v_preds_next):
#     deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
#     # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
#     gaes = copy.deepcopy(deltas)
#     for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
#         gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

#     return gaes


# def get_return(rewards):
#     dis_rewards = np.zeros_like(rewards).astype(np.float32)
#     running_add = 0
#     for t in reversed(range(len(rewards))):
#         running_add = running_add * hype_parameters["gamma"] + rewards[t]
#         dis_rewards[t] = running_add

#     return dis_rewards

# # FIXME: tensorflow
# def set_global_seeds(i):
#     myseed = i  # + 1000 * rank if i is not None else None
#     try:
#         import tensorflow as tf
#         tf.set_random_seed(myseed)
#     except Exception as e:
#         print("Check your tensorflow version")
#         raise e
#     np.random.seed(myseed)
#     random.seed(myseed)


# set_global_seeds(args.random_seed)


# def check_file_path():
#     check_path("./documents")
#     check_path("./result")
#     check_path("./result/summary")
#     check_path("./documents/%s" % args.log_index)