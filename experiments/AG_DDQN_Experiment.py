from agents.DDQN import DDQN
from agents.ag_rl_agents.AG_RL import AG_RL
from experiments.Experiment import Experiment
from utilities.data_structures.Config import Config
from experiments.Atari_Experiment_Hyperparameters import hyperparameters
from datetime import date
import torch


ddqn_seeds = {
"Qbert": [2222036254, 1385813590, 3229622889,2342341215, 24214],
"Seaquest": [906611417, 1617394797, 183971277,3057113908, 1194828811],
"MsPacman": [3085735466, 784810369, 3106942522, 958393300, 4092608692],
"Pong": [900374875, 425242984, 1268248291, 3059342911, 1375404729],
"SpaceInvaders": [515971329, 4115555067, 1649529769, 4246981793, 1630387507],
"Breakout": [977247108, 4256556115, 1003198519, 1099800525, 4030529493],
"BeamRider": [1602519929, 889469344, 3902704614, 2224264456, 2234846850],
"Enduro": [3279685920, 2975274488, 3745987151, 3764520445, 2207431782]
}


ag_ddqn_seeds = {
"Qbert": [3371768399, 1136464578, 3654173039, 2573980213, 853286407],
"Seaquest": [2270976657, 3708167041, 4013626520, 922183997, 3678959131],
"MsPacman": [4224147910, 53946457, 2159614645, 2923355409, 2475970369],
"Pong": [1360621188, 2051325511, 3545109725, 3312544181, 3181332361],
"SpaceInvaders": [3819817708, 3030444420, 4013470085, 1075168044, 3664632971],
"Breakout": [3029453301, 3902429328, 3053560648, 1237464986, 3105767641],
"BeamRider": [611422491, 2922706235, 2919579084, 3080826863, 3752400488],
"Enduro": [3898213037, 2909344284, 1138100642, 4183709724, 1038883768]
}


config = Config()
config.env_parameters = {}
config.num_steps_to_run = 350000
config.runs_per_agent = 1
config.use_GPU =  torch.cuda.is_available()
config.include_step_in_state = False
config.atari = True
config.hyperparameters = {
    "optimizer": hyperparameters["optimizer"],
    "initial_random_steps": hyperparameters["initial_random_steps"],
    "momentum": hyperparameters["momentum"],
    "squared_momentum": hyperparameters["squared_momentum"],
    "min_squared_gradient": hyperparameters["min_squared_gradient"],
    "min_epsilon": hyperparameters["min_epsilon"],
    "exploration_decay_num_steps": hyperparameters["exploration_decay_num_steps"],
    "update_fixed_network_freq": hyperparameters["update_fixed_network_freq"],
    "loss": hyperparameters["loss"],
    "linear_hidden_units": [1],
    "learning_rate": hyperparameters["learning_rate"],
    "buffer_size": hyperparameters["buffer_size"],
    "batch_size": hyperparameters["batch_size"],
    "final_layer_activation": "None",
    "batch_norm": False,
    "gradient_clipping_norm": None,
    "update_every_n_steps": hyperparameters["update_every_n_steps"],
    "discount_rate": hyperparameters["discount_rate"],
    "learning_iterations": 1,
    "clip_rewards": hyperparameters["clip_rewards"] ,

    #Habit-RL hyperparameters
    "evaluation_eps": 5,
    "sequitur_k": "N/A",
    "action_balanced_replay_buffer": True,
    "action_frequency_required_in_top_results": "N/A",
    "steps_per_round": 75001,
    "abandon_ship": 1.0,
    "max_macro_length": 2,
    "grammar_algorithm": "IGGI",
    "Base_Agent": "DDQN",

}


# Then to replicate the DDQN results
if __name__ == "__main__":
    agent = DDQN
    for game in ddqn_seeds.keys():
        config.environment_id = game
        for seed in ddqn_seeds[game]:
            config.seed = seed
            experiment_name = "{}_{}_{}_seed_{}".format(game, agent.__name__, date.today(), config.seed)
            experiment = Experiment(config, agent, spreadsheet_id=None)
            experiment.run_and_save_every_n_steps_to_csv(25000, experiment_name)

# And to replicate the AG-DDQN results
if __name__ == "__main__":
    agent = AG_RL
    for game in ag_ddqn_seeds.keys():
        config.environment_id = game
        for seed in ddqn_seeds[game]:
            config.seed = seed
            experiment_name = "{}_{}_{}_seed_{}".format(game, agent.__name__, date.today(), config.seed)
            experiment = Experiment(config, agent, spreadsheet_id=None)
            experiment.run_and_save_every_n_steps_to_csv(25000, experiment_name)






