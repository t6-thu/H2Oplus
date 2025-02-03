# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file, get_server_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims.geometry_prim import GeometryPrimView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.world import World

import omni.kit

from gym import spaces
import numpy as np
import torch
import math

from scipy.spatial.transform import Rotation


class WheellegTask(BaseTask):
    def __init__(self, name, offset=None) -> None:

        # task-specific parameters
        self._cartpole_position = [0.0, 0.0, 0.3]
        self._cartpole_rotation = [1.0, 0.0, 0.0, 0.0]
        self._cube_position = [1.8, -0.75, 0.075]
        self._cube_rotation = [0.70798, 0.0, 0.0, 0.70623]
        self._cube_scale = [0.015, 0.015, 0.0015]
        self._triangle_position = [0.8, -0.75, 0.21905]
        self._triangle_rotation = [0.70711, 0.70711, 0.0, 0.0]
        self._reset_dist = 3.0
        self._max_push_effort = 3
        self._max_joint_speed = math.pi

        # values used for defining RL buffers
        self._num_observations = 12 #[alpha_l, alpha_r, beta_l, beta_r, x_l, x_r, v_l, v_r, pitch, roll, yaw, gyro_y]
        self._num_actions = 6 #[joint_effort_l&r,alpha_speed_l&r,beta_speed_l&r]
        self._device = "cpu"
        self.num_envs = 1

        # a few class buffers to store RL-related states
        self.wheel_x_l = 0
        self.wheel_x_r = 0
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        )

        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Cartpole USD file
        usd_path = "/home/lbq/rl/isaac_sim_arx6_balance_ppo/ARX-6.usd"
        usd_path_cube = "/home/lbq/rl/isaac_sim_arx6_balance_ppo/cube.usd"
        usd_path_triangle = "/home/lbq/rl/isaac_sim_arx6_balance_ppo/triangle.usd"
        # add the Cartpole USD to our stage
        create_prim(prim_path="/World/Cartpole", prim_type="Xform", position=self._cartpole_position)
        add_reference_to_stage(usd_path, "/World/Cartpole")
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._cartpoles = ArticulationView(prim_paths_expr="/World/Cartpole*", name="cartpole_view")
        # add Cartpole ArticulationView and ground plane to the Scene
        scene.add(self._cartpoles)

        create_prim(prim_path="/World/cube",prim_type="Xform", position=self._cube_position, orientation=self._cube_rotation, scale=self._cube_scale)        
        #add_reference_to_stage(usd_path_cube, "/World/cube")
        create_prim(prim_path="/World/triangle",prim_type="Xform", position=self._triangle_position, orientation=self._triangle_rotation)        
        add_reference_to_stage(usd_path_triangle, "/World/triangle")
        # self._cube = GeometryPrimView(prim_paths_expr="/World/cube*", name="geometry_prim")
        # scene.add(self._cube)
        
        scene.add_default_ground_plane()

        # set default camera viewport position and target
        self.set_initial_camera_params()

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        viewport = omni.kit.viewport_legacy.get_default_viewport_window()
        viewport.set_camera_position(
            "/OmniverseKit_Persp", camera_position[0], camera_position[1], camera_position[2], True
        )
        viewport.set_camera_target("/OmniverseKit_Persp", camera_target[0], camera_target[1], camera_target[2], True)

    def post_reset(self):
        self._joint_left_wheel_idx = self._cartpoles.get_dof_index("joint_left_wheel")
        self._joint_right_wheel_idx = self._cartpoles.get_dof_index("joint_right_wheel")
        self._joint_left_thigh_idx = self._cartpoles.get_dof_index("joint_left_thigh")
        self._joint_right_thigh_idx = self._cartpoles.get_dof_index("joint_right_thigh")
        self._joint_left_shin_idx = self._cartpoles.get_dof_index("joint_left_shin")
        self._joint_right_shin_idx = self._cartpoles.get_dof_index("joint_right_shin")

        # print(self._joint_left_wheel_idx, self._joint_right_wheel_idx, self._joint_left_thigh_idx, self._joint_right_thigh_idx, self._joint_left_shin_idx, self._joint_right_shin_idx)
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)

        # initialize dynamic control
        joint_kps, joint_kds = self._cartpoles.get_gains(indices=indices)
        joint_max_efforts = self._cartpoles.get_max_efforts(indices=indices)

        joint_kps[:, self._joint_left_shin_idx] = 0.0
        joint_kds[:, self._joint_left_shin_idx] = 10.0
        joint_max_efforts[:, self._joint_left_shin_idx] = 15.0

        joint_kps[:, self._joint_right_shin_idx] = 0.0
        joint_kds[:, self._joint_right_shin_idx] = 10.0
        joint_max_efforts[:, self._joint_right_shin_idx] = 15.0

        joint_kps[:, self._joint_left_thigh_idx] = 0.0
        joint_kds[:, self._joint_left_thigh_idx] = 10.0
        joint_max_efforts[:, self._joint_left_thigh_idx] = 15.0

        joint_kps[:, self._joint_right_thigh_idx] = 0.0
        joint_kds[:, self._joint_right_thigh_idx] = 10.0
        joint_max_efforts[:, self._joint_right_thigh_idx] = 15.0

        # joint_kps[:, self._joint_left_wheel_idx] = 0.0
        # joint_kds[:, self._joint_left_wheel_idx] = 0.0
        # joint_max_efforts[:, self._joint_left_wheel_idx] = 15.0

        # joint_kps[:, self._joint_right_wheel_idx] = 0.0
        # joint_kds[:, self._joint_right_wheel_idx] = 0.0
        # joint_max_efforts[:, self._joint_right_wheel_idx] = 15.0

        # print(joint_kps)

        self._cartpoles.set_gains(kps=joint_kps, kds=joint_kds, indices=indices)
        self._cartpoles.set_max_efforts(joint_max_efforts, indices=indices)

        self.reset(indices)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._joint_right_thigh_idx] = 1.062
        dof_pos[:, self._joint_left_thigh_idx] = 1.062
        dof_pos[:, self._joint_right_shin_idx] = -1.527
        dof_pos[:, self._joint_left_shin_idx] = -1.527

        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        
        # # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._joint_left_wheel_idx] = 0.0
        dof_vel[:, self._joint_right_wheel_idx] = 0.0

        # apply resets
        
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)
        cart_pose_world_tmp = self._cartpoles.get_world_poses()
        # print(cart_pose_world_tmp)
        cart_pose_world_tmp = (torch.zeros(num_resets, 3), torch.zeros(num_resets, 4))
        cart_pose_world_tmp[0][:,2] = 0.3
        cart_pose_world_tmp[1][:,0] = 1.0
        # print(cart_pose_world_tmp)
        #cart_pose_world_tmp = [torch.tensor(self._cartpole_position), torch.tensor(self._cartpole_rotation)]
        self._cartpoles.set_world_poses(cart_pose_world_tmp[0], cart_pose_world_tmp[1], indices=indices)

        cart_vel_tmp = (torch.zeros(num_resets, 6))
        self._cartpoles.set_velocities(cart_vel_tmp, indices=indices)

        # print('reset')
        # print(cart_pose_world_tmp)

        # bookkeeping
        self.wheel_x_l = 0
        self.wheel_x_r = 0
        self.resets[env_ids] = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        self.actions = torch.tensor(actions)

        joint_efforts = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        joint_efforts[:, self._joint_left_wheel_idx] = self._max_push_effort * self.actions[4]
        joint_efforts[:, self._joint_right_wheel_idx] = self._max_push_effort * self.actions[5]

        joint_vels = self._cartpoles.get_joint_velocities()
        joint_vels[:, self._joint_left_thigh_idx] = self._max_joint_speed * self.actions[0]
        joint_vels[:, self._joint_right_thigh_idx] = self._max_joint_speed * self.actions[1]
        joint_vels[:, self._joint_left_shin_idx] = self._max_joint_speed * self.actions[2]
        joint_vels[:, self._joint_right_shin_idx] = self._max_joint_speed * self.actions[3]


        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        joint_indices = torch.tensor([self._joint_left_thigh_idx, self._joint_right_thigh_idx, self._joint_left_shin_idx, self._joint_right_shin_idx])
        joint_vels_set = torch.tensor([joint_vels[:, self._joint_left_thigh_idx], joint_vels[:, self._joint_right_thigh_idx], joint_vels[:, self._joint_left_shin_idx], joint_vels[:, self._joint_right_shin_idx]])
        self._cartpoles.set_joint_velocity_targets(joint_vels_set, joint_indices=joint_indices)
        self._cartpoles.set_joint_efforts(joint_efforts, indices=indices)
        # print(joint_vels_set)
        # kp, kd = self._cartpoles.get_gains()
        # print(kp, kd)
        # joint_state = self._cartpoles.get_joints_state()
        # print(joint_state.positions, joint_state.velocities, joint_state.efforts)

    def get_observations(self):
        dof_pos = self._cartpoles.get_joint_positions()
        dof_vel = self._cartpoles.get_joint_velocities()

        # collect pole and cart joint positions and velocities for observation
        # cart_pos = (dof_pos[:, self._joint_left_wheel_idx] + dof_pos[:, self._joint_right_wheel_idx]) / 2.0
        # cart_vel = (dof_vel[:, self._joint_left_wheel_idx] + dof_vel[:, self._joint_right_wheel_idx]) / 2.0

        # print("car_pos:"+f"{dof_pos[:, self._joint_left_wheel_idx]}")
        # print("car_vel:"+f"{cart_vel}")
        # pole_pos = dof_pos[:, self._pole_dof_idx]
        # pole_vel = dof_vel[:, self._pole_dof_idx]

        # cart_pose_local = self._cartpoles.get_local_poses()
        cart_pose_world = self._cartpoles.get_world_poses()
        cart_vel_linear = self._cartpoles.get_linear_velocities()
        cart_vel_angular = self._cartpoles.get_angular_velocities()

        # print(self._cart_pose_world[0])
        # print(self._cart_pose_world[1])

        cart_rotation = cart_pose_world[1].numpy()


        cart_rotation[0][0], cart_rotation[0][1] = cart_rotation[0][1], cart_rotation[0][0]
        cart_rotation[0][1], cart_rotation[0][2] = cart_rotation[0][2], cart_rotation[0][1]
        cart_rotation[0][2], cart_rotation[0][3] = cart_rotation[0][3], cart_rotation[0][2]

        # print(cart_rotation)
        matrix_r = Rotation.from_quat(cart_rotation)
        euler = matrix_r.as_euler("zyx", degrees=False)
        # print(f"{euler}")
        cart_euler_angle = torch.tensor(euler)
        self._cart_euler_angle = cart_euler_angle

        joint_pos = self._cartpoles.get_joint_positions()

        # print("local:"+f"{cart_pose_local}")
        # print("world:"+f"{cart_pose_world}")

        # self.obs[:, 0] = cart_pose_world[0][:, 0]
        # self.obs[:, 1] = cart_vel_linear[:, 0]
        # self.obs[:, 0] = cart_pos * 0.065
        # delta_x = - 0.38 * np.cos(0.790246 - cart_euler_angle[:, 1])
        wheel_speed_l = dof_vel[:, self._joint_left_wheel_idx]
        wheel_speed_r = dof_vel[:, self._joint_right_wheel_idx]
        self.wheel_x_l += wheel_speed_l * 0.065 * 0.005
        self.wheel_x_r += wheel_speed_r * 0.065 * 0.005
        
        self.obs[:, 0] = joint_pos[:, self._joint_left_thigh_idx]
        self.obs[:, 1] = joint_pos[:, self._joint_right_thigh_idx]
        self.obs[:, 2] = joint_pos[:, self._joint_left_shin_idx] 
        self.obs[:, 3] = joint_pos[:, self._joint_right_shin_idx]
        self.obs[:, 4] = wheel_speed_l
        self.obs[:, 5] = wheel_speed_r
        self.obs[:, 6] = self.wheel_x_l
        self.obs[:, 7] = self.wheel_x_r
        self.obs[:, 8] = cart_euler_angle[:, 1]
        self.obs[:, 9] = cart_euler_angle[:, 2]
        self.obs[:, 10] = cart_euler_angle[:, 0]
        self.obs[:, 11] = cart_vel_angular[:, 1]

        # print("p:"+f"{self.obs[:, 0]}")
        # print("v:"+f"{self.obs[:, 1]}")
        # print("a_p:" + f"{self.obs[:, 2]}")
        # print("a_v:"+f"{self.obs[:, 3]}")

        return self.obs

    def calculate_metrics(self) -> None:
        """ Reward Function
        Each data sample consists of:
        - State Space: [0]alpha_l, [1]alpha_r, [2]beta_l, [3]beta_r, [4]wheel_speed_l, [5]wheel_speed_r, [6]x_l, [7]x_r, [8]pitch, [9]roll, [10]yaw, [11]gyro_y
        - Action Space: [0]alpha_l_speed, [1]alpha_r_speed, [2]beta_l_speed, [3]beta_r_speed, [4]torque_l, [5]torque_r
        - Done: [0]is_done

        reward = 15 - (wheel_speed_l - 0.8) ** 2 - (wheel_speed_r - 0.8) ** 2  - alpha_l_speed ** 2 - alpha_r_speed ** 2 - beta_l_speed ** 2 - beta_r_speed ** 2
        reward += 10 * 1((x_l + x_r)/2 >=1.1)
        reward -= 1(yaw > pi / 6) 
        reward -= 1(roll > pi / 18) 
        reward -= 1(pitch > pi / 6)
        """
        wheel_speed_l = self.obs[:, 4] * 0.065
        wheel_speed_r = self.obs[:, 5] * 0.065
        alpha_l_speed = self.actions[:, 0]
        alpha_r_speed = self.actions[:, 1]
        beta_l_speed = self.actions[:, 2]
        beta_r_speed = self.actions[:, 3]
        
        x_l, x_r = self.obs[:, 6], self.obs[:, 7]
        x = (x_l + x_r) / 2
        pitch, roll, yaw = self.obs[:, 8], self.obs[:, 9], self.obs[:, 10]

        # compute reward based on angle of pole and cart velocity
        reward = 15.0 - (wheel_speed_l - 0.8) ** 2 - (wheel_speed_r - 0.8) ** 2  - alpha_l_speed ** 2 - alpha_r_speed ** 2 - beta_l_speed ** 2 - beta_r_speed ** 2
        # apply a reward if cart arrives the final line
        reward += torch.where(x > 1.1, torch.ones_like(reward) * 10.0, torch.zeros_like(reward))
        # apply a penalty if pole is too far from the orginal pose
        reward = torch.where(torch.abs(yaw) > math.pi / 6, torch.ones_like(reward) * -1.0, reward)
        reward = torch.where(torch.abs(pitch) > math.pi / 6, torch.ones_like(reward) * -1.0, reward)
        reward = torch.where(torch.abs(roll) > math.pi / 18, torch.ones_like(reward) * -1.0, reward)


        return reward.item()

    def is_done(self) -> None:
        cart_pos_x = self.obs[:, 6]
        cart_pose_world = self._cartpoles.get_world_poses()
        cart_pos_y = cart_pose_world[0][:, 2]
        cart_pos_z = self.obs[:, 2]
        pole_pos = self.obs[:, 8]
        cart_roll = self._cart_euler_angle[:, 2]
        cart_yaw = self._cart_euler_angle[:,0]


        # reset the robot if cart has reached reset_dist or pole is too far from upright
        # resets = torch.where(torch.abs(cart_pos_x) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(cart_pos_y) > self._reset_dist, 1, resets)
        # resets = torch.where(torch.abs(cart_pos_z) > self._reset_dist, 1, resets)
        #if resets == 1:
            # print("a")
        resets = torch.where(torch.abs(pole_pos) > math.pi / 6, 1, 0)
        #if resets == 1:
            # print("b")
        # resets = torch.where(torch.abs(cart_pos_y) < 0.2, 1, resets)
        #if resets == 1:
            # print("c")
        resets = torch.where(torch.abs(cart_roll) > 10.0 / 180.0 * math.pi, 1, resets)
        #if resets == 1:
            # print("d")
        resets = torch.where(torch.abs(cart_yaw) > 10.0 / 60.0 * math.pi, 1, resets)
        #if resets == 1:
            # print("e")
        resets = torch.where(torch.abs(self.obs[:, 0]) > (math.pi / 6 + 1.062), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 1]) > (math.pi / 6 + 1.062), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 0]) < (-math.pi / 6 + 1.062), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 1]) < (-math.pi / 6 + 1.062), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 2]) > (math.pi / 6 + 1.527), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 3]) > (math.pi / 6 + 1.527), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 2]) < (-math.pi / 6 + 1.527), 1, resets)
        resets = torch.where(torch.abs(self.obs[:, 3]) < (-math.pi / 6 + 1.527), 1, resets)
        self.resets = resets



        return resets.item()