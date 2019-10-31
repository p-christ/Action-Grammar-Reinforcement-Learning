from agents.DDQN import DDQN
from agents.ag_rl_agents.AG_RL import AG_RL
from experiments.Experiment import Experiment
from utilities.data_structures.Config import Config
from experiments.Atari_Experiment_Hyperparameters import hyperparameters
from datetime import date
import torch
import random
import argparse




# virtualenv venv_action_grammar
# source venv_action_grammar/bin/activate
# pip3 install -r requirements.txt


# export PYTHONPATH="$PWD/Action-Grammar-Reinforcement-Learning/*"

# python3 AG_DDQN_Custom_Experiment.py --env_id=Qbert --agent=DDQN --use_gpu=False --steps=1000000000000
#

# Add command line choice of DDQN vs. AG-DDQN
# Add command line choice of number of steps
# Add command line choice of how often to interpret an action grammar...


today = date.today()
parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str)
parser.add_argument('--agent', type=str)
parser.add_argument('--seed', type=int, default=random.randint(0, 2**32 - 2))
parser.add_argument('--use_gpu', type=str)
parser.add_argument('--steps', type=int, default=0)
args = parser.parse_args()

if args.agent == "DDQN":
        agent = DDQN
elif args.agent == "AG-DDQN":
        agent = AG_RL

config = Config()
config.environment_id = args.env_id
config.seed = args.seed
config.env_parameters = {}
config.num_steps_to_run = args.steps
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

    #Action grammar hyperparameters
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


if __name__ == "__main__":
    experiment_name = "{}_{}_{}_seed_{}".format(args.env_id, agent.__name__, date.today(), config.seed)
    experiment = Experiment(config, agent, spreadsheet_id=None)
    experiment.run_and_save_every_n_steps_to_csv(25000, experiment_name)





