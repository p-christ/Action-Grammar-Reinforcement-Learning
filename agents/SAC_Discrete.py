import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import random
import torch.nn as nn
from utilities.Networks import create_atari_network
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from .SAC import SAC
from utilities.Utility_Functions import create_actor_distribution
torch.distributions.Distribution.set_default_validate_args(True)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == None, "Final actor layer must have no activation because we apply softmax later"
        self.hyperparameters = config.hyperparameters
        if config.atari:
            self.actor_local = create_atari_network(self.action_size, config.seed, softmax_final_layer=False).to(self.device)
            self.critic_local = create_atari_network(self.action_size, config.seed, softmax_final_layer=False).to(self.device)
            self.critic_local_2 = create_atari_network(self.action_size, config.seed+1, softmax_final_layer=False).to(self.device)
            self.critic_target = create_atari_network(self.action_size, config.seed, softmax_final_layer=False).to(self.device)
            self.critic_target_2 = create_atari_network(self.action_size, config.seed+1, softmax_final_layer=False).to(self.device)
            self.do_no_ops = False
        else:
            self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                              key_to_use="Actor")
            self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                               key_to_use="Critic", override_seed=self.config.seed + 1)
            self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                  key_to_use="Critic")
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"])
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"])
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"])
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, atari=config.atari)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.max_target_entropy_multiplier = 0.98
            self.target_entropy = self.calc_target_entropy()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"])
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters["add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]
        self.fixed_epsilon = False
        self.softmax = nn.Softmax()
        if "normalise_learning" in self.hyperparameters.keys():
            self.normalise_learning = self.hyperparameters["normalise_learning"]
        else: self.normalise_learning = False
        if "reward_scale" in self.hyperparameters.keys(): self.reward_scale = self.hyperparameters["reward_scale"]
        else: self.reward_scale = 1.0
        self.first_step_of_ep = False

    def atari_evaluation(self, epsilon, episodes, do_no_ops=False):
        """Evaluates the agent using the methodology from the Deepmind atari paper """
        assert epsilon == 0.01, "Needs to be 0.01 for atari eval with SAC"
        self.fixed_epsilon = epsilon
        print("Not keeping track of best episodes")
        self.environment = self.eval_env
        self.move_lengths = []
        self.attempted_move_lengths = []
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

    def actor_pick_action(self, state, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if not eval: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.item()
        return action

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        action_probabilities =  self.softmax(self.actor_local(state))
        if random.random() < 0.001: print("Action probabilities ", action_probabilities)
        if (self.fixed_epsilon is not False or isinstance(self.fixed_epsilon, float)) and random.random() < self.fixed_epsilon:
            max_probability_action = torch.randint(0, self.action_size, (1,))
            print("Rand action ", max_probability_action)
        else:
            max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        if self.normalise_learning:
            log_action_probabilities = torch.clamp(log_action_probabilities, LOG_STD_MIN, LOG_STD_MAX)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (_, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            next_state_log_pi = log_action_probabilities.gather(1, next_state_action.unsqueeze(-1).long())
            qf1_next_target = self.critic_target(next_state_batch).gather(1, next_state_action.unsqueeze(-1).long())
            qf2_next_target = self.critic_target_2(next_state_batch).gather(1, next_state_action.unsqueeze(-1).long())
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = action_probabilities * inside_term
        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def calc_target_entropy(self):
        """Calculates the target entropy"""
        return -np.log((1.0 / self.action_size)) * self.max_target_entropy_multiplier