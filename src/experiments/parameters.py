
base_parameters = {
    "model_type": "separated",
    "observation_embedding": "default",
    "activation_fn": "relu",
    "norm_layer": "batch_norm",
    "dir_epsilon": 0.4,
    "dir_alpha": 2.5,
    "selection_policy": "PUCT",
    "puct_c": 1.0,
    "use_visit_count": True,
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

}
