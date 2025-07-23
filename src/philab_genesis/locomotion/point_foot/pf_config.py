def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,

        "links_to_keep": [
            "foot_R_Link",
            "foot_L_Link",
        ],

        "default_joint_angles": {  # [rad]
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        },

        "dof_names": [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            # "foot_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            # "foot_R_Joint",
        ],

        "penalize_contact_links": [
            "knee_R_Link",
            "knee_L_Link",
            "hip_R_Link",
            "hip_L_Link",
            "base_Link",
        ],

        "foot_names": [
            "foot_R_Link",
            "foot_L_Link",
        ],

        "foot_radius": 0.03,  # [m]

        "num_gait_params": 4,
        "gait_ranges": {
            "frequencies": [1.5, 2.5],
            "offsets": [0.0, 1.0],
            "durations": [0.5, 0.5],
            "swing_height": [0.0, 0.1],
        },

        # PD
        "kp": 42.0,
        "kd": 2.5,

        # termination
        "termination_if_roll_greater_than": 90,  # degree
        "termination_if_pitch_greater_than": 90,

        # base pose
        "base_init_pos": [0.0, 0.0, 0.8],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }

    obs_cfg = {
        "num_obs": 30,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "dof_acc": 0.0025,
            "height_measurements": 5.0,
            "contact_forces": 0.01,
            "torque": 0.05,
        },
    }

    reward_cfg = {
        "tracking_sigma": 0.2,
        "ang_tracking_sigma": 0.25,
        "base_height_target": 0.68,
        "feet_height_target": 0.10,
        "kappa_gait_probs": 0.05,
        "gait_force_sigma": 25.0,
        "gait_vel_sigma": 0.25,
        "gait_height_sigma": 0.005,
        "min_feet_distance": 0.115,
        "about_landing_threshold": 0.08,

        "reward_scales": {
            "keep_balance": 1.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "base_height": -2.0,
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "torques": -0.00008,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "dof_pos_limits": -2.0,
            "collision": -1.0,
            "action_smooth": -0.01,
            "orientation": -10.0,
            "feet_distance": -100.0,
            "feet_regulation": -0.05,
            "foot_landing_vel": -0.15,
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-2.0, 2.0],
        "lin_vel_y_range": [1.0, 1.0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg
