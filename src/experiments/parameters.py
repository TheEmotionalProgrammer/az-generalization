
base_parameters = {
    "model_type": "seperated",
    "observation_embedding": "default",
    "activation_fn": "relu",
    "norm_layer": "batch_norm",
    "dir_epsilon": 0.4,
    "dir_alpha": 2.5,
    "selection_policy": "PUCT",
    "puct_c": 1.0,
    "use_visit_count": False,
    "regularization_weight": 1e-6,
    "tree_evaluation_policy": "visit",
    "tree_temperature": None,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "sample_batch_ratio": 4,
    "n_steps_learning": 3,
    "training_epochs": 4,
    "planning_budget": 32,
    "layers": 2,
    "replay_buffer_multiplier": 15,
    "discount_factor": 0.99,
    "lr_gamma": 1.0,
    "iterations": 40,
    "policy_loss_weight": 0.3,
    "value_loss_weight": 0.7,
    "max_episode_length": 1000,
    "episodes_per_iteration": 6,
    "eval_temp": 0,
}

env_challenges = {

    "GridWorldNoObst8x8-v1": {
        "env_description": "GridWorldNoObst8x8-v1",
        "max_episode_length": 100,
        "env_params": dict(id="GridWorldNoObst8x8-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 14,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "ncols": 8,
        "learning_rate": 1e-3,
    },

    "GridWorldNoObst16x16-v1": {
        "env_description": "GridWorldNoObst16x16-v1",
        "max_episode_length": 100,
        "env_params": dict(id="GridWorldNoObst16x16-v1", max_episode_steps=1000000000),
        "optimal_value": 1.0 * 0.95 ** 30,
        "worst_value": 0.0,
        "discount_factor": 0.95,
        "ncols": 16,
        "learning_rate": 3e-3,
    },

}

grid_env_descriptions = {
    
    "8x8_NO_OBSTS": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_SLALOM": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHHHF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_SLALOM": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHHHHHFF",
        "HHHHHHHHHHHHHHFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "8x8_DEFAULT": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],

    "16x16_DEFAULT": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFFFFFHHFFFFFF",
        "FFFFFFFFHHFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFHHFFFFFFFFFF",
        "FFFFFFFFFFFFHHFF",
        "FFHHFFFFFFFFHHFF",
        "FFHFFFFFHHFFHHFF",
        "FFHFFFFFHHFFHHFF",
        "FFHFFFFFFFFFFFFF",
        "FFFFFHHFFFFFFFFF",
        "FFFFFHHFFFFFFFFG"
    ],

    "8x8_NARROW": [
        "SFFFFFHH",
        "FFFFFFHH",
        "HHFHHHHH",
        "HHFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_NARROW": [
        "SFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "FFFFFFFFFFFFHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "HHHHFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],
    
    "8x8_MAZE_RL": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LR": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LL": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_RR": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_RC": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HHHHHFFH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHFFHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "8x8_MAZE_LC": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFHHHHH",
        "FFFFFFFF",
        "FFFFFFFF",
        "HHHFFHHH",
        "FFFFFFFF",
        "FFFFFFFG"
    ],

    "16x16_MAZE_RL": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LR": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LR_OBS": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFHFHFHFFFFF",
        "FFFFFFHFFFHFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFHFHFHFFFFF",
        "HHHHHHHHHHHFFFHH",
        "HHHHHHHHHHHFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_LL": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHFFFFHHHHHHHHHH",
        "HHFFFFHHHHHHHHHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

    "16x16_MAZE_RR": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "HHHHHHHHHHFFFFHH",
        "HHHHHHHHHHFFFFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],


    "16x16_NO_OBSTS": [
        "SFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFG"
    ],

}
