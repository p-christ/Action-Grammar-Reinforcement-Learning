from agents.SAC_Discrete import SAC_Discrete
from agents.ag_rl_agents.AG_RL import AG_RL
from experiments.Experiment import Experiment
from utilities.data_structures.Config import Config
from experiments.Atari_SAC_Experiment_Hyperparameters import hyperparameters

from datetime import date
import torch

sac_seeds = {
"Qbert": [1584312663, 84723650, 2347278325, 2138920865, 2893437300],
"Seaquest": [2808335366, 2300739511, 2668553285, 2421338340, 1466510285],
"MsPacman": [2959866888, 2767846435, 47921108, 3482473054, 1569488317],
"Pong": [457828226, 1959089012, 4247553704, 1399394364, 3873026035],
"SpaceInvaders": [4228323168, 2899636816, 546127527, 2379399512, 1196351391],
"Breakout": [1394691682, 3920415066, 3889139280, 3925398669, 4113033498],
"BeamRider": [1124046921, 1980672812, 3331926649, 1095965363, 17396416],
"Enduro": [337758359, 1078309263, 3570536014, 1122342151, 572052972],
"Alien": [1468148249, 2461909633, 317766427, 2341408502, 1477604533],
"Amidar": [2222781163, 2933113218, 2669614889, 1509253504, 2003317686],
"RoadRunner": [3137399863, 21145325, 3017585184, 2603009822, 3942919701],
"Frostbite": [35913873, 182557270, 2077080966, 3074607679, 1415371122],
"CrazyClimber": [882157713, 1323304887, 3606900001, 4265900904, 2956006336],
"Asterix": [642215779, 2016905384, 1107410280, 4154876351, 804195827],
"Freeway": [2490836899, 1331200974, 4047451402, 913007748, 1845931894],
"Assault": [151686474, 1097755119, 3041653445, 2151908231, 2474777170],
"Jamesbond": [717097157, 3695732881, 970035076, 2759903519, 777312285],
"BattleZone": [659635825, 123724138, 593614205, 4120255265, 1076623584],
"Kangaroo": [2007320754, 126219658, 1250482964, 3479260728, 1357951852],
"UpNDown": [1520883355, 1974606985, 68395944, 351455183, 542833801],
}


ag_sac_seeds = {
"Qbert": [2929279936, 3200405851, 1099108964, 2305163547, 3218148386],
"Seaquest": [1508953277, 2347706604, 1495884559, 2681341367, 1843488580],
"MsPacman": [606203110, 2684993546, 857541054, 3365762077, 1521558022],
"Pong": [3608627212, 3596362292, 2581646226, 1770367043, 3053047275],
"SpaceInvaders": [637826365, 999157816, 3640825450, 49898690, 3588002768],
"Breakout": [1998842350, 585488214, 674481577, 2510500393, 2874455840],
"BeamRider": [3394300173, 3201182033, 1599318294, 1986163617, 450554464],
"Enduro": [3436136573, 664745474, 367040228, 3101616986, 3960945079],
"Alien": [206077666, 908493736, 2764173590, 3705488943, 1988926810],
"Amidar": [1187057682, 2221608290, 573221644, 2954336083, 1103999470],
"RoadRunner": [242362729, 2623042028, 272772029, 2171007484, 1793886500],
"Frostbite": [2626723042, 1340620367, 2747242697, 2136953668, 2110267591],
"CrazyClimber": [3193716910, 2905117330, 213633330, 2481707733, 3743777157],
"Asterix": [2995738504, 4184287713, 287806997, 197750713, 435054158],
"Freeway": [2601359562, 2149994563, 649032489, 2529249332, 1642061662],
"Assault": [3255659501, 856712347, 1798867307, 322477613, 3539566964],
"Jamesbond": [1717800865, 165522756, 206576282, 2483373390, 2844893760],
"BattleZone":[282190044, 1421137067, 3866926017, 4238387869, 2515030310],
"Kangaroo": [3264544660, 2597674143, 3662388535, 2788089746, 1916577905],
"UpNDown": [4008315755, 1168017983, 1165981639, 2065049538, 844395828],
}



config = Config()
config.env_parameters = {}
config.num_steps_to_run = 350000
config.runs_per_agent = 1
config.use_GPU =  torch.cuda.is_available()
config.include_step_in_state = False
config.atari = True
config.hyperparameters = {
    "Actor": {
            "learning_rate": hyperparameters["learning_rate"],
            "final_layer_activation": None,
            "gradient_clipping_norm": None,
        },

    "Critic": {
        "learning_rate":  hyperparameters["learning_rate"],
        "buffer_size": hyperparameters["buffer_size"],
        "gradient_clipping_norm":  None,
        "update_fixed_network_freq": hyperparameters["update_fixed_network_freq"],
        "tau": None
    },

    "optimizer": hyperparameters["optimizer"],
    "initial_random_steps":  hyperparameters["initial_random_steps"],
    "batch_size": hyperparameters["batch_size"],
    "update_every_n_steps": hyperparameters["update_every_n_steps"],
    "discount_rate": hyperparameters["discount_rate"],
    "learning_iterations": 1,
    "clip_rewards": hyperparameters["clip_rewards"] ,
    "automatically_tune_entropy_hyperparameter": True,
    "add_extra_noise": False,
    "do_evaluation_iterations": False,


    #Habit-RL hyperparameters
    "evaluation_eps": 5,
    "sequitur_k": "N/A",
    "action_balanced_replay_buffer": True,
    "action_frequency_required_in_top_results": "N/A",
    "steps_per_round": 30001,
    "abandon_ship": 2.0,
    "max_macro_length": 2,
    "grammar_algorithm": "IGGI",
    "Base_Agent": "SAC",

}

# Replicates the SAC results
if __name__ == "__main__":
    agent = SAC_Discrete
    for game in sac_seeds.keys():
        config.environment_id = game
        for seed in sac_seeds[game]:
            if game in ["Assault", "BankHeist", "Jamesbond", "Kangaroo", "UpNDown", "BattleZone"]:
                eval_every_n_steps = 100000
            else:
                eval_every_n_steps = 10000

            config.seed = seed
            experiment_name = "AS_{}_{}_{}_{}_seed_{}".format(config.hyperparameters["abandon_ship"], game,
                                                              agent.__name__, date.today(), config.seed)
            experiment = Experiment(config, agent, spreadsheet_id=None)
            experiment.run_and_save_every_n_steps_to_csv(eval_every_n_steps, experiment_name, eval_epsilon=0.01,
                                                         num_deterministic_eps=0)


# Replicates the Habit-SAC results
if __name__ == "__main__":
    agent = AG_RL
    for game in ag_sac_seeds.keys():
        config.environment_id = game
        for seed in ag_sac_seeds[game]:
            if game in ["Assault", "BankHeist", "Jamesbond", "Kangaroo", "UpNDown", "BattleZone"]:
                eval_every_n_steps = 100000
            else:
                eval_every_n_steps = 10000

            config.seed = seed
            experiment_name = "AS_{}_{}_{}_{}_seed_{}".format(config.hyperparameters["abandon_ship"], game,
                                                              agent.__name__, date.today(), config.seed)
            experiment = Experiment(config, agent, spreadsheet_id=None)
            experiment.run_and_save_every_n_steps_to_csv(eval_every_n_steps, experiment_name, eval_epsilon=0.01,
                                                         num_deterministic_eps=0)






































