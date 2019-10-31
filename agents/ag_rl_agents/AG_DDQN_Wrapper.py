import random
import torch
import time
from torch import nn
import numpy as np
from collections import deque
from agents.DDQN import DDQN
from numpy.random import choice
from utilities.Networks import load_atari_cnn_pretrained
from agents.Base_Agent import Base_Agent
from utilities.Memory_Shaper import Memory_Shaper
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.ag_rl_agents.AG_Wrapper_Base import AG_Wrapper_Base

class AG_DDQN_Wrapper(AG_Wrapper_Base, DDQN):
    """Wraps DDQN to create a version of DDQN that can be used as the base algorithm in Habit RL"""
    def __init__(self, config, end_of_episode_symbol="/"):
        DDQN.__init__(self, config)
        AG_Wrapper_Base.__init__(self)
        assert isinstance(self.hyperparameters["macro_gambling"], float)
        assert isinstance(self.hyperparameters["abandon_ship"], float)
        if config.include_step_in_state:
            self.state_size += 1
        self.original_num_primitive_actions = self.action_size
        self.do_no_ops=False
        if config.atari:
            self.q_network_local = load_atari_cnn_pretrained(self.env_id, self.original_num_primitive_actions,
                                                             config.seed).to(self.device)
            self.q_network_optimizer = self.create_optimizer()
            self.q_network_target = load_atari_cnn_pretrained(self.env_id, self.original_num_primitive_actions,
                                                              config.seed).to(self.device)
        else:
            self.q_network_local = self.create_NN(self.state_size, self.action_size).to(self.device)
            self.q_network_optimizer = self.create_optimizer()
            self.q_network_target = self.create_NN(self.state_size, self.action_size).to(self.device)
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.using_frozen_feature_extractor = True
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = end_of_episode_symbol
        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}
        self.action_id_to_stepping_stone_action_id = {}
        self.action_id_to_primitive_stepping_stone_id = {}
        self.memory_shaper = Memory_Shaper(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                           config.seed,
                                           self.hyperparameters["action_balanced_replay_buffer"], atari="Atari" in self.environment_title, device=self.device)
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                    config.seed, atari="Atari" in self.environment_title)
        self.fixed_epsilon = False
        self.last_1000_actions = deque(maxlen=1000)
        self.macro_action_disadvantages = deque(maxlen=30000)
        self.macro_abandonments = deque(maxlen=30000)
        self.macro_action_disadvantage_mean = 0
        self.macro_action_disadvantage_std = 0
        self.macro_action_exploration_weight = 4

    def gather_no_exploration_eps(self, num_episodes):
        """Runs some episodes with no exploration and returns the actions taken
        args:
            num_episodes: number of episodes you want to run without exploration
        returns:
            episode_actions_scores_and_exploration_status: the score achieved and actions taken in each episode"""
        self.turn_off_any_epsilon_greedy_exploration()
        self.episode_actions_scores_and_exploration_status = []
        for _ in range(num_episodes):
            self.reset_game()
            self.play_episode(eval_ep=False)
            self.save_and_print_result()
        self.turn_on_any_epsilon_greedy_exploration()
        self.check_episodes_data_valid()
        return self.episode_actions_scores_and_exploration_status

    def run_n_steps(self, num_steps):
        """Runs n steps of the game
        args:
            num_steps: number of steps to run"""
        self.turn_on_any_epsilon_greedy_exploration()
        self.episode_actions_scores_and_exploration_status = []
        num_steps_to_get_to = self.global_step_number + num_steps
        if "eval_eps_every" in self.hyperparameters.keys():
            eval_eps_every = self.hyperparameters["eval_eps_every"]
        else:
            if "Pong" in self.environment_title: eval_eps_every = 20
            elif "Enduro" in self.environment_title: eval_eps_every = 10
            else: eval_eps_every = 50
        no_exploration_data = []
        while self.global_step_number < num_steps_to_get_to:
            self.reset_game()
            if self.episode_number % eval_eps_every == 0 and self.global_step_number > self.hyperparameters["initial_random_steps"]:
                print("Doing no exploration episode")
                no_exploration_results = self.gather_no_exploration_eps(1)
                no_exploration_data.extend(no_exploration_results)
            else:
                self.play_episode()
                self.save_and_print_result()
        self.check_episodes_data_valid()
        return no_exploration_data

    def abandon_ship(self, state, action):
        """Checks if action we are going to do is much worse than the best action we could do in terms of q_value. Then it
        changes the action if it is worse than some threshold
        args:
            state: state of the game
            action: proposed action
        returns:
            action: action to play
            changed_action: boolean indicating whether the action got changed"""
        with torch.no_grad():
            q_values = self.calculate_q_value(state, local=True)[:, :self.original_num_primitive_actions]
        best_action_q_value, best_action = torch.max(q_values, dim=1)
        our_action_q_value = q_values[:, action]
        macro_action_disadvantage = self.how_much_worse(q_values, best_action_q_value, our_action_q_value).item()
        self.macro_action_disadvantages.append(macro_action_disadvantage)
        if macro_action_disadvantage >= self.macro_action_disadvantage_mean +  self.hyperparameters["abandon_ship"] * self.macro_action_disadvantage_std:
            action = best_action.item()  # do the best primitive action instead
            self.macro_abandonments.append(1.0)
            changed_action = True
        else:
            self.macro_abandonments.append(0.0)
            changed_action = False
        return action, changed_action

    def play_episode(self, eval_ep=False, save_experience_override=False):
        """Runs an episode of the game including learning steps if required
        args:
            eval_ep: boolean indicating whether it is an evaluation episode
            save_experience_override: boolean indicating whether we want to save the experiences in the replay buffer
        returns:
            total_episode_score_so_far: the score we got in this episode"""
        self.total_episode_score_so_far = 0
        state = self.state
        done = self.done
        episode_macro_actions = []
        while not done:
            changed_action = False
            macro_action = self.pick_action(state=state, evaluate=eval_ep)
            episode_macro_actions.append(macro_action)
            primitive_actions = self.global_action_id_to_primitive_action[macro_action]
            if eval_ep: self.attempted_move_lengths.append(len(primitive_actions))
            macro_reward = 0
            primitive_actions_conducted = 0
            for action in primitive_actions:
                if self.hyperparameters["abandon_ship"] > 0.0 and primitive_actions_conducted >= 1 and not self.random_action_move:
                    action, changed_action = self.abandon_ship(state, action)
                next_state, reward, done, _ = self.environment.step(action)
                self.total_episode_score_so_far += reward
                if self.hyperparameters["clip_rewards"]: reward = max(min(reward, 1.0), -1.0)
                macro_reward += reward
                primitive_actions_conducted += 1
                self.track_episodes_data(state, action, reward, next_state, done)
                if not eval_ep: self.global_step_number += 1
                state = next_state
                if self.time_for_q_network_to_learn() and not eval_ep:
                    for _ in range(self.hyperparameters["learning_iterations"]):
                        self.learn()
                if self.global_step_number % self.hyperparameters["update_fixed_network_freq"] == 0:
                    self.copy_model_over(self.q_network_local, self.q_network_target)
                if done or changed_action: break
            if eval_ep: self.move_lengths.append(primitive_actions_conducted)
        self.last_1000_actions.extend(episode_macro_actions)
        if random.random() < 0.1:
            proportions = self.calc_action_usage_statistics()
            print("Ep Score {} -- Action distribution {} ".format(self.total_episode_score_so_far, proportions))
        if not eval_ep or save_experience_override:
            self.memory_shaper.add_episode_experience(self.episode_states, self.episode_next_states, self.episode_rewards,
                                                      self.episode_actions, self.episode_dones)
            experiences = (self.episode_states, self.episode_next_states, self.episode_rewards, self.episode_actions, self.episode_dones)
            actions_to_action_id = {v: k for k, v in self.global_action_id_to_primitive_action.items()}
            self.memory_shaper.hindsight_action_replay(None, actions_to_action_id, self.memory, self.original_num_primitive_actions, experiences=experiences)
        if not eval_ep:
            self.save_episode_actions_with_score()
            self.episode_number += 1
        if len(self.macro_action_disadvantages) > 0 and self.episode_number % 3 == 0 and not eval_ep:
            self.macro_action_disadvantage_mean = np.mean(self.macro_action_disadvantages)
            self.macro_action_disadvantage_std = np.std(self.macro_action_disadvantages)
            self.macro_abandonments_avg = np.mean(self.macro_abandonments)
            print("Macro action disadvantage mean {} -- std {} -- abandonments {}".format(self.macro_action_disadvantage_mean, self.macro_action_disadvantage_std, self.macro_abandonments_avg))
        return self.total_episode_score_so_far

    def update_final_layers(self, num_actions_before, num_new_actions):
        """Adds neurons to the final layer of a network to allow it to choose from the new actions. It does not change the weights
        for the other actions.
        args:
            num_actions_before: the number of actions the agent had before
            num_new_actions: the number of new actions that just got added to the action set"""
        assert num_new_actions > 0, num_new_actions
        new_layer = self.transfer_learning_new_weights(self.q_network_local, self.q_network_target, num_actions_before, num_new_actions)
        self.q_network_local.output_layers.append(new_layer)
        self.q_network_target.output_layers.append(
            nn.Linear(in_features=self.q_network_local.output_layers[0].in_features,
                      out_features=num_new_actions))
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        self.q_network_local = self.q_network_local.to(self.device)
        self.q_network_target = self.q_network_target.to(self.device)
        self.q_network_optimizer = self.create_optimizer()

    def check_episodes_data_valid(self):
        """Checks the episodes data we are outputting back to the meta-agent is valid"""
        if len(self.episode_actions_scores_and_exploration_status) > 0:
            assert len(self.episode_actions_scores_and_exploration_status[0]) == 3
            assert self.episode_actions_scores_and_exploration_status[0][2] in [True, False]
            assert isinstance(self.episode_actions_scores_and_exploration_status[0][1], list)
            assert isinstance(self.episode_actions_scores_and_exploration_status[0][1][0], int)
            assert isinstance(self.episode_actions_scores_and_exploration_status[0][0], int) or isinstance(
                self.episode_actions_scores_and_exploration_status[0][0], float)

    def pick_action(self, state=None, evaluate=False):
        """Uses the local Q network and an epsilon greedy policy to pick an action
        args:
            state: the state of the game
            evaluate: boolean for whether it is an evaluation episode
        returns:
            action: the action we want to play"""
        if state is None: state = self.state
        epsilon = self.get_updated_epsilon_exploration()
        if random.random() < epsilon:
            self.random_action_move = True
            return self.randomly_pick_action(evaluate=evaluate)
        self.random_action_move = False
        self.q_network_local.eval()  # puts network in evaluation mode
        with torch.no_grad():
            action_values = self.calculate_q_value(state, local=True)
        self.q_network_local.train()  # puts network back in training mode
        action = torch.argmax(action_values, dim=1).item()
        if random.random() < 0.001:
            print("Actions {} -- Action values {} -- Picking Action {} ".format(self.global_action_id_to_primitive_action, action_values, action))
        return action

    def randomly_pick_action(self, evaluate):
        """Randomly picks an action from action set
        args:
            evaluate: boolean indicating whether it is an evaluation episode"""
        if evaluate:
            # if evaluating we only pick random action from primitive actions
            action = self.environment.action_space.sample()
        else:
            if self.original_num_primitive_actions == self.action_size:
                action = random.randint(0, self.action_size - 1)
            else:
                macro_action_weighting = [self.macro_action_exploration_weight for
                                          macro_action in self.global_action_id_to_primitive_action.keys() if
                                          macro_action >= self.original_num_primitive_actions]
                weighting = [1 for _ in range(self.original_num_primitive_actions)]
                weighting.extend(macro_action_weighting)
                probabilities = weighting / np.sum(weighting)
                action = choice(range(self.action_size), 1, p=probabilities)[0]
        return action





