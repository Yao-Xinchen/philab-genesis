import torch
import math
import genesis as gs
from genesis.engine.entities import RigidEntity
from philab_genesis.utils.math import torch_rand_float
from genesis.utils.geom import xyz_to_quat, quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


class PfEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot: RigidEntity = self.scene.add_entity(
            gs.morphs.URDF(
                file="resources/robots/PF_TRON1A/urdf/robot.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.foot_links = [self.robot.get_link(name).idx_local for name in self.env_cfg["foot_names"]]
        self.foot_num = len(self.foot_links)
        self.penalize_links = [self.robot.get_link(name).idx_local for name in self.env_cfg["penalize_contact_links"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        self.dof_pos_lower_limits, self.dof_pos_upper_limits = self.robot.get_dofs_limit(self.motor_dofs)

        self._prepare_rewards()

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_acc = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_inv_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.contact_forces = self.robot.get_links_net_contact_force()  # (n_envs, n_links, 3)

        self.foot_pos = torch.zeros((self.num_envs, self.foot_num, 3), device=self.device, dtype=gs.tc_float)
        self.foot_quat = torch.zeros((self.num_envs, self.foot_num, 4), device=self.device, dtype=gs.tc_float)
        self.foot_vel = torch.zeros((self.num_envs, self.foot_num, 3), device=self.device, dtype=gs.tc_float)
        self.foot_ang_vel = torch.zeros((self.num_envs, self.foot_num, 3), device=self.device, dtype=gs.tc_float)
        self.foot_inv_quat = torch.zeros((self.num_envs, self.foot_num, 4), device=self.device, dtype=gs.tc_float)
        self.foot_rel_vel = torch.zeros((self.num_envs, self.foot_num, 3), device=self.device, dtype=gs.tc_float)

        self.gaits = torch.zeros(
            (self.num_envs, self.env_cfg["num_gait_params"]),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)

        self.extras = dict()  # extra information for logging

    def _prepare_rewards(self):
        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = torch_rand_float(*self.command_cfg["lin_vel_x_range"],
                                                      (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = torch_rand_float(*self.command_cfg["lin_vel_y_range"],
                                                      (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = torch_rand_float(*self.command_cfg["ang_vel_range"],
                                                      (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_inv_quat[:] = inv_base_quat
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_acc[:] = (self.last_dof_vel - self.dof_vel) / self.dt
        self.torques[:] = self.robot.get_dofs_control_force(self.motor_dofs)
        self.contact_forces[:] = self.robot.get_links_net_contact_force()

        self._compute_foot_state()

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)
        self._resample_gaits(envs_idx)

        self._step_contact_targets()

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                    torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def _compute_foot_state(self):
        self.foot_pos = self.robot.get_links_pos(self.foot_links)  # (n_envs, n_foot_links, 3)
        self.foot_quat = self.robot.get_links_quat(self.foot_links)  # (n_envs, n_foot_links, 4)
        self.foot_vel = self.robot.get_links_vel(self.foot_links)  # (n_envs, n_foot_links, 3)
        self.foot_ang_vel = self.robot.get_links_ang(self.foot_links)  # (n_envs, n_foot_links, 3)

        self.foot_inv_quat = inv_quat(self.foot_quat)  # (n_envs, n_foot_links, 4)

        for i in range(self.foot_num):
            self.foot_ang_vel[:, i] = transform_by_quat(
                self.foot_ang_vel[:, i], self.foot_inv_quat[:, i]
            )

        foot_rel_vel = self.foot_vel - self.base_lin_vel.unsqueeze(1).repeat(1, self.foot_num, 1)

        for i in range(self.foot_num):
            self.foot_rel_vel[:, i, :] = transform_by_quat(
                foot_rel_vel[:, i, :], self.base_inv_quat
            )
        self.foot_heights = torch.clip(
            (self.foot_pos[:, :, 2] - self.env_cfg["foot_radius"] - self._get_foot_heights()),
            0, 1
        )

    def _get_foot_heights(self):
        return torch.zeros(
            self.num_envs,
            self.foot_num,
            device=self.device,
            requires_grad=False,
        )

    def _resample_gaits(self, env_ids):
        if len(env_ids) == 0:
            return

        self.gaits[env_ids, 0] = torch_rand_float(
            self.env_cfg["gait_ranges"]["frequencies"][0],
            self.env_cfg["gait_ranges"]["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.env_cfg["gait_ranges"]["offsets"][0],
            self.env_cfg["gait_ranges"]["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.env_cfg["gait_ranges"]["durations"][0],
            self.env_cfg["gait_ranges"]["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 3] = torch_rand_float(
            self.env_cfg["gait_ranges"]["swing_height"][0],
            self.env_cfg["gait_ranges"]["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _step_contact_targets(self):
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        self.clock_inputs_sin = torch.sin(2 * torch.pi * self.gait_indices)
        self.clock_inputs_cos = torch.cos(2 * torch.pi * self.gait_indices)

        # von mises distribution
        kappa = self.reward_cfg["kappa_gait_probs"]
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_indices.view(self.num_envs, 1),
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )
        stance_idx = foot_indices < durations
        swing_idx = foot_indices > durations

        foot_indices[stance_idx] = (torch.remainder(foot_indices[stance_idx], 1)
                                    * (0.5 / durations[stance_idx]))
        foot_indices[swing_idx] = 0.5 + (
                torch.remainder(foot_indices[swing_idx], 1) - durations[swing_idx]
        ) * (0.5 / (1 - durations[swing_idx]))

        self.desired_contact_states = (
                smoothing_cdf_start(foot_indices) * (1 - smoothing_cdf_start(foot_indices - 0.5))
                + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))
        )

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non-flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_base_height(self):
        base_height = self.base_pos[:, 2]
        return torch.square(base_height - self.reward_cfg["base_height_target"])

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        # out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        out_of_limits = -(self.dof_pos - self.dof_pos_lower_limits).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_upper_limits).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["ang_tracking_sigma"])

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.foot_links, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(self.foot_num):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.reward_cfg["gait_force_sigma"])
        else:
            for i in range(self.foot_num):
                reward += (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-foot_forces[:, i] ** 2 / self.reward_cfg["gait_force_sigma"]))

        return reward / self.foot_num

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_vel, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(self.foot_num):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.reward_cfg["gait_vel_sigma"]
                )
        else:
            for i in range(self.foot_num):
                reward += desired_contact[:, i] * (
                        1 - torch.exp(-foot_velocities[:, i] ** 2 / self.reward_cfg["gait_vel_sigma"]))
        return reward / self.foot_num

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_pos[:, 0, :2] - self.foot_pos[:, 1, :2], dim=-1)
        reward = torch.clip(self.reward_cfg["min_feet_distance"] - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.reward_cfg["base_height_target"] * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_vel[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalize_links, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_vel[:, :, 2]
        contacts = self.contact_forces[:, self.foot_links, 2] > 0.1
        about_to_land = (self.foot_heights < self.reward_cfg["about_landing_threshold"]) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
