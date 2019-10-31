import torch
import random
import torch.nn.functional as F
import numpy as np

from environments.Atari_Environments import create_atari_environments
from utilities.Utility_Functions import prepare_lazy_frames_for_network
from utilities.Networks import load_atari_cnn_pretrained, get_atari_CNN_model
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer

class DDQN(Base_Agent):
    """A deep double Q learning agent"""
    agent_name = "DDQN"
    def __init__(self, config):
        super().__init__(config)
        if config.atari:
            self.q_network_local = load_atari_cnn_pretrained(self.env_id, self.action_size, config.seed).to(self.device)
            self.q_network_target = load_atari_cnn_pretrained(self.env_id, self.action_size, config.seed).to(self.device)
        else:
            self.q_network_local = self.create_NN(self.state_size, self.action_size)
            self.q_network_target = self.create_NN(self.state_size, self.action_size)
        self.q_network_optimizer = self.create_optimizer()
        self.memory = Replay_Buffer(self.hyperparameters["buffer_size"], self.hyperparameters["batch_size"],
                                    config.seed, atari="Atari" in self.environment_title)
        self.fixed_epsilon = False
        self.do_no_ops = False

    def play_episode(self, eval_ep=False, save_experience_override=False):
        """Runs a step within a game including a learning step if required"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_q_network_to_learn() and not eval_ep:
                for _ in range(self.hyperparameters["learning_iterations"]):
                    self.learn()
            if not eval_ep or save_experience_override: self.save_experience()
            self.state = self.next_state #this is to set the state for the next iteration
            if not eval_ep: self.global_step_number += 1
            if self.global_step_number % self.hyperparameters["update_fixed_network_freq"] == 0:
                self.copy_model_over(self.q_network_local, self.q_network_target)
        if not eval_ep: self.episode_number += 1
        return self.total_episode_score_so_far

    def calculate_q_value(self, states, local):
        """Takes states as input and calculates q_values. Local is a binary variable indicating whether to use the
        local network or the target network"""
        if self.config.atari:
            states = prepare_lazy_frames_for_network(states).to(self.device)
        elif isinstance(states, list) or isinstance(states, np.ndarray):
            states = torch.FloatTensor([states]).to(self.device)
            if len(states.shape) == 1: states = states.unsqueeze(0)
        if local:
            network_action_values = self.q_network_local(states)
        else:
            network_action_values = self.q_network_target(states)
        return network_action_values

    def pick_action(self, state=None):
        """Uses the local Q network and an epsilon greedy policy to pick an action"""
        # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
        # a "fake" dimension to make it a mini-batch rather than a single observation
        if state is None: state = self.state
        epsilon = self.get_updated_epsilon_exploration()
        if random.random() < epsilon: return self.environment.action_space.sample()
        self.q_network_local.eval() #puts network in evaluation mode
        with torch.no_grad():
            action_values = self.calculate_q_value(state, local=True)
        self.q_network_local.train() #puts network back in training mode
        action = torch.argmax(action_values, dim=1).item()
        return action

    def get_updated_epsilon_exploration(self):
        """Gets the probability that we just pick a random action. This probability decays the more episodes we have seen"""
        if self.fixed_epsilon: return self.fixed_epsilon
        if self.turn_off_exploration: return 0.0
        if self.global_step_number < self.hyperparameters["initial_random_steps"]: return 1.0
        steps_since_burn_in = self.global_step_number - self.hyperparameters["initial_random_steps"]
        reduction = steps_since_burn_in * (1.0 - self.hyperparameters["min_epsilon"]) / float(self.hyperparameters["exploration_decay_num_steps"])
        epsilon = max(self.hyperparameters["min_epsilon"], 1.0 - reduction)
        if random.random() < 0.01: print("Epsilon ", epsilon)
        return epsilon

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        if experiences is None: states, actions, rewards, next_states, dones = self.sample_experiences() #Sample experiences
        else: states, actions, rewards, next_states, dones = experiences
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyperparameters["gradient_clipping_norm"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """Computes the loss required to train the Q network"""
        with torch.no_grad():
            max_action_indexes = self.calculate_q_value(next_states, local=True).detach().argmax(1)
            Q_targets_next = self.calculate_q_value(next_states, local=False).gather(1, max_action_indexes.unsqueeze(1))
            Q_targets = rewards + (self.hyperparameters["discount_rate"] * Q_targets_next * (1 - dones))
        Q_expected = self.calculate_q_value(states, local=True).gather(1, actions.long())  # must convert actions to long so can be used as index
        if self.hyperparameters["loss"] == "MSE":
            loss = F.mse_loss(Q_expected, Q_targets)
        elif self.hyperparameters["loss"] == "HUBER":
            loss = torch.nn.SmoothL1Loss()(Q_expected, Q_targets)
        return loss

    def time_for_q_network_to_learn(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin and there are
        enough experiences in the replay buffer to learn from"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from()

    def right_amount_of_steps_taken(self):
        """Returns boolean indicating whether enough steps have been taken for learning to begin"""
        return self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0 and self.global_step_number > self.hyperparameters["initial_random_steps"]

    def enough_experiences_to_learn_from(self):
        """Boolean indicated whether there are enough experiences in the memory buffer to learn from"""
        return len(self.memory) > self.hyperparameters["batch_size"]

    def sample_experiences(self):
        """Draws a random sample of experience from the memory buffer"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones

    def save_experience(self, memory=None, experience=None):
        """Saves the recent experience to the memory buffer"""
        if memory is None: memory = self.memory
        if experience is None: experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def atari_evaluation(self, epsilon, episodes, do_no_ops=False):
        """Evaluates the agent using the methodology from the Deepmind 2015 atari paper
        https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf"""
        self.fixed_epsilon = epsilon
        print("Not keeping track of best episodes")
        self.environment = self.eval_env
        if do_no_ops:
            self.do_no_ops = True
        for _ in range(episodes):
            self.reset_game()
            self.play_episode(eval_ep=True)
            self.game_full_episode_scores.append(self.total_episode_score_so_far)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
        self.fixed_epsilon = False
        print("Keeping track of best episodes")
        self.do_no_ops = False
        self.environment = self.training_env
        return self.game_full_episode_scores, 1, 1
