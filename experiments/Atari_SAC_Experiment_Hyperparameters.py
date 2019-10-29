# Hyperparameters used in SAC experiments

hyperparameters = {
    "batch_size": 64,
    "buffer_size": 1000000,
    "discount_rate": 0.99,
    "update_every_n_steps": 4,
    "learning_iterations": 1,
    "learning_rate":  3*(10.0**-4),
    "optimizer" :"Adam",
    "initialiser":"he",
    "exploration_decay_num_steps":100000,
    "update_fixed_network_freq": 8000,
    "initial_random_steps": 20000,
    "steps":100000,
    "batch_norm": False,
    "tau": "N/A",
    "epsilon_decay_rate_denominator": "N/A",
    "clip_rewards": True,
    "feature_extraction": True,
}





