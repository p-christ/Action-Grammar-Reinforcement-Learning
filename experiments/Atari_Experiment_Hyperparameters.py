# Hyperparameters used in DDQN experiments

hyperparameters = {
    "batch_size": 32,
    "buffer_size": 1000000,
    "discount_rate": 0.99,
    "update_every_n_steps": 4,
    "learning_iterations": 1,
    "learning_rate": 0.0005,
    "optimizer" :"Adam",
    "momentum":0.95,
    "squared_momentum": 0.95,
    "min_squared_gradient": 0.01,
    "initialiser":"he",
    "min_epsilon": 0.1,
    "exploration_decay_num_steps":100000,
    "update_fixed_network_freq": 10000,
    "initial_random_steps": 25000,
    "steps":350000,
    "final_layer_activation": "None",
    "loss": "HUBER",
    "batch_norm": False,
    "tau": "N/A",
    "epsilon_decay_rate_denominator": "N/A",
    "clip_rewards": True,
    "feature_extraction": True,
}





