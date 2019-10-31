import logging
import os
import sys
import gym
import random
import numpy as np
import torch
import time
import tensorflow as tf
from nn_builder.pytorch.NN import NN
from torch import optim
from environments.Atari_Environments import create_atari_environments

class Base_Agent(object):
    """Base agent that all agents inheirt from"""
    def __init__(self, config):
        if config.atari:
            self.env_id = config.environment_id
            self.environment_title = "Atari" + self.env_id
            self.training_env, self.eval_env = create_atari_environments(self.env_id + "Deterministic-v4", config.seed)
            self.environment = self.training_env
        else:
            self.environment = gym.make(config.environment_id)
            self.environment_title = str(self.environment.unwrapped)
        self.config = config
        self.set_random_seeds(config.seed)
        self.action_types = "DISCRETE" if self.environment.action_space.dtype == int else "CONTINUOUS"
        self.action_size = int(self.get_action_size())
        self.config.action_size = self.action_size
        self.lowest_possible_episode_score = self.get_lowest_possible_episode_score()
        self.state_size =  int(self.get_state_size())
        self.hyperparameters = config.hyperparameters
        self.rolling_score_window = self.get_trials()
        self.total_episode_score_so_far = 0
        self.game_full_episode_scores = []
        self.rolling_results = []
        self.max_rolling_score_seen = float("-inf")
        self.max_episode_score_seen = float("-inf")
        self.episode_number = 0
        self.device = "cuda:0" if config.use_GPU else "cpu"
        if config.use_GPU: assert torch.cuda.is_available()
        self.visualise_results_boolean = config.visualise_individual_results
        self.global_step_number = 0
        self.turn_off_exploration = False
        gym.logger.set_level(40)  # stops it from printing an unnecessary warning

    def step(self):
        """Takes a step in the game. This method must be overriden by any agent"""
        raise ValueError("Step needs to be implemented by the agent")

    def get_environment_title(self):
        """Extracts name of environment from it
        returns:
            name: str name of environment"""
        try: name = self.environment.unwrapped.id
        except AttributeError:
            try:
                name = self.environment.spec.id.split("-")[0]
            except AttributeError:
                name = str(self.environment.env)
                if name[0:10] == "TimeLimit<": name = name[10:]
                name = name.split(" ")[0]
                if name[0] == "<": name = name[1:]
                if name[-3:] == "Env": name = name[:-3]
        if "Frostbite" in name or "MsPacman" in name or "MontezumaRevenge" in name or "Breakout" in name or "Amidar" in name:
            name = name + "Atari"
        return name

    def get_lowest_possible_episode_score(self):
        """Returns the lowest possible episode score you can get in an environment"""
        if self.environment_title == "Taxi": return -800
        return None

    def get_action_size(self):
        """Gets the action_size for the gym env into the correct shape for a neural network"""
        if "overwrite_action_size" in self.config.__dict__: return self.config.overwrite_action_size
        if "action_size" in self.environment.__dict__: return self.environment.unwrapped.action_size
        if self.action_types == "DISCRETE": return self.environment.unwrapped.action_space.n
        else: return self.environment.unwrapped.action_space.shape[0]

    def get_state_size(self):
        """Gets the state_size for the gym env into the correct shape for a neural network"""
        random_state = self.environment.reset()
        print(self.environment_title)
        if "Atari" in self.environment_title: return 1024
            # return random_state.shape[1]
        if isinstance(random_state, dict):
            state_size = random_state["observation"].shape[0] + random_state["desired_goal"].shape[0]
            return state_size
        else:
            return random_state.size

    def get_trials(self):
        """Gets the number of trials to average a score over"""
        if self.environment_title in ["AntMaze", "FetchReach", "Hopper", "Walker2d"]: return 100
        if self.environment_title == "CartPole": return 200
        if "Atari" in self.environment_title: return 100
        try: return self.environment.unwrapped.trials
        except AttributeError: return self.environment.spec.trials

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        tf.set_random_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
        if hasattr(gym.spaces, 'prng'):
            gym.spaces.prng.seed(random_seed)

    def reset_atari_env(self):
        """Resets the atari environment including doing no-op starts
        returns:
            obs: state of the reset environment"""
        self.environment.seed(self.config.seed)
        obs = self.environment.reset()
        self.no_ops = 0
        if self.do_no_ops:
            self.no_ops = random.randint(1, 30)
            noop_action = 0
            assert self.environment.unwrapped.get_action_meanings()[0] == 'NOOP'
            if not "SpaceInvaders" in self.environment_title:
                assert self.environment.unwrapped.frameskip == 4
            base_frameskip = self.environment.unwrapped.frameskip
            self.environment.unwrapped.frameskip = 1
            for _ in range(self.no_ops):
                obs, _, done, _ = self.environment.step(noop_action)
                if done:
                    self.environment.seed(self.config.seed)
                    obs = self.environment.reset()
            self.environment.unwrapped.frameskip = base_frameskip
        return obs

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        if "Atari" in self.environment_title: self.state = self.reset_atari_env()
        else: self.state = self.environment.reset()
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.total_episode_score_so_far = 0
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []
        self.episode_desired_goals = []
        self.episode_achieved_goals = []
        self.episode_observations = []
        if "exploration_strategy" in self.__dict__.keys(): self.exploration_strategy.reset()

    def track_episodes_data(self, state, action, reward, next_state, done):
        """Tracks this episode's experiences"""
        self.episode_states.append(state)
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        self.episode_next_states.append(next_state)
        self.episode_dones.append(done)

    def save_episode_actions_with_score(self):
        """Keeps track of the actions taken in those episodes that got over a certain score. Also keeps track of the first
        episode regardless of its score"""
        self.episode_actions_scores_and_exploration_status.append([self.total_episode_score_so_far,
                                                                   self.episode_actions + [
                                                                       self.end_of_episode_symbol],
                                                                   self.turn_off_exploration])

    def run_n_steps(self, num_steps=None,  save_and_print_results=True):
        """Runs game to completion n times and then summarises results
        args:
            num_steps: int number of steps to run
            save_and_print_results: saves and print results
        returns:
            game_full_episode_scores: list of episode scores
            rolling_results: rolling mean of episode results
            time_taken: second it took to run n steps"""
        start = time.time()
        if num_steps is None: num_steps = self.config.num_steps_to_run
        while self.global_step_number < num_steps:
            self.reset_game()
            self.play_episode()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def run_n_episodes(self, num_episodes=None, save_and_print_results=True):
        """Runs game to completion n times and then summarises results and saves model (if asked to)
        args:
            num_steps: int number of steps to run
            save_and_print_results: saves and print results
        returns:
            game_full_episode_scores: list of episode scores
            rolling_results: rolling mean of episode results
            time_taken: second it took to run n steps"""
        if num_episodes is None: num_episodes = self.config.num_episodes_to_run
        start = time.time()
        while self.episode_number < num_episodes:
            self.reset_game()
            self.play_episode()
            if save_and_print_results: self.save_and_print_result()
        time_taken = time.time() - start
        if self.config.save_model: self.locally_save_policy()
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def conduct_action(self, action):
        """Conducts an action in the environment"""
        self.next_state, self.reward, self.done, _ = self.environment.step(action)
        self.total_episode_score_so_far += self.reward
        if self.hyperparameters["clip_rewards"]: self.reward =  max(min(self.reward, 1.0), -1.0)

    def save_and_print_result(self):
        """Saves and prints results of the game"""
        self.save_result()
        self.print_rolling_result()

    def save_result(self):
        """Saves the result of an episode of the game"""
        self.game_full_episode_scores.append(self.total_episode_score_so_far)
        self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.save_max_result_seen()

    def save_max_result_seen(self):
        """Updates the best episode result seen so far"""
        if self.game_full_episode_scores[-1] > self.max_episode_score_seen:
            self.max_episode_score_seen = self.game_full_episode_scores[-1]

        if self.rolling_results[-1] > self.max_rolling_score_seen:
            if len(self.rolling_results) > self.rolling_score_window:
                self.max_rolling_score_seen = self.rolling_results[-1]

    def print_rolling_result(self):
        """Prints out the latest episode results"""
        text = """"\r Episode {0}, Step {5}, Score: {3: .2f}, Max score seen: {4: .2f}, Rolling score: {1: .2f}, Max rolling score seen: {2: .2f}"""
        sys.stdout.write(text.format(len(self.game_full_episode_scores), self.rolling_results[-1], self.max_rolling_score_seen,
                                     self.game_full_episode_scores[-1], self.max_episode_score_seen, self.global_step_number))
        sys.stdout.flush()

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer
        args:
            memory: a replay buffer to save the experience to
            experience: an experience to save in the replay buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters
        args:
            optimizer: the optimizer to use
            network: the neural network we are updating
            loss: the loss we are using to calculate gradients
            clipping_norm: the norm to clip gradients to
            retain_graph: whether to retain graph after updating parameters"""
        if not isinstance(network, list): network = [network]
        optimizer.zero_grad() #reset gradients to 0
        loss.backward(retain_graph=retain_graph) #this calculates the gradients
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm) #clip gradients to help stabilise training
        optimizer.step() #this applies the gradients

    def create_NN(self, input_dim, output_dim, key_to_use=None, override_seed=None, hyperparameters=None):
        """Creates a neural network for the agents to use
        args:
            input_dim: input dimension for the neural network
            output_dim: output dimension of the neural network"""
        if hyperparameters is None: hyperparameters = self.hyperparameters
        if key_to_use: hyperparameters = hyperparameters[key_to_use]
        if override_seed: seed = override_seed
        else: seed = self.config.seed
        default_hyperparameter_choices = {"output_activation": None, "hidden_activations": "relu", "dropout": 0.0,
                                          "initialiser": "default", "batch_norm": False,
                                          "columns_of_data_to_be_embedded": [],
                                          "embedding_dimensions": [], "y_range": ()}
        for key in default_hyperparameter_choices:
            if key not in hyperparameters.keys():
                hyperparameters[key] = default_hyperparameter_choices[key]
        return NN(input_dim=input_dim, layers_info=hyperparameters["linear_hidden_units"] + [output_dim],
                  output_activation=hyperparameters["final_layer_activation"],
                  batch_norm=hyperparameters["batch_norm"], dropout=hyperparameters["dropout"],
                  hidden_activations=hyperparameters["hidden_activations"], initialiser=hyperparameters["initialiser"],
                  columns_of_data_to_be_embedded=hyperparameters["columns_of_data_to_be_embedded"],
                  embedding_dimensions=hyperparameters["embedding_dimensions"], y_range=hyperparameters["y_range"],
                  random_seed=seed).to(self.device)

    def turn_on_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """Turns off all exploration with respect to the epsilon greedy exploration strategy"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    @staticmethod
    def freeze_all_but_output_layers(network):
        """Freezes all layers except the output layer of a network"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """Unfreezes all layers of a network"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def create_optimizer(self):
        """Creates the optimizer for the q_network"""
        optimizer = optim.Adam(self.q_network_local.parameters(),
                                              lr=self.hyperparameters["learning_rate"])
        return optimizer

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)