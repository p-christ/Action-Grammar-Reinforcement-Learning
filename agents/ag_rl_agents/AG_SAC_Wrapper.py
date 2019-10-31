import random
from collections import deque
import numpy as np
import torch
from agents.SAC_Discrete import SAC_Discrete
from utilities.Utility_Functions import prepare_lazy_frames_for_network, create_actor_distribution
from agents.Base_Agent import Base_Agent
from utilities.Memory_Shaper import Memory_Shaper
from agents.ag_rl_agents.AG_Wrapper_Base import AG_Wrapper_Base
from torch import nn
from numpy.random import choice

class AG_SAC_Wrapper(AG_Wrapper_Base, SAC_Discrete):
    """Wraps SAC-Discrete to create a version of SAC that can be used as the base algorithm in Habit RL"""
    def __init__(self, config, end_of_episode_symbol="/"):
        SAC_Discrete.__init__(self, config)
        AG_Wrapper_Base.__init__(self)
        self.original_num_primitive_actions = self.action_size
        self.do_no_ops = False
        self.min_episode_score_seen = float("inf")
        self.end_of_episode_symbol = end_of_episode_symbol
        self.global_action_id_to_primitive_action = {k: tuple([k]) for k in range(self.action_size)}
        self.action_id_to_stepping_stone_action_id = {}
        self.action_id_to_primitive_stepping_stone_id = {}
        self.memory_shaper = Memory_Shaper(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                           config.seed,
                                           self.hyperparameters["action_balanced_replay_buffer"], atari=True,
                                           device=self.device)
        self.last_1000_actions = deque(maxlen=1000)
        self.macro_action_disadvantages = deque(maxlen=30000)
        self.macro_abandonments = deque(maxlen=30000)
        self.macro_action_disadvantage_mean = 0
        self.macro_action_disadvantage_std = 0
        self.macro_action_exploration_weight = 4.0
        self.post_habit_inference_random_steps = 5000

    def produce_action_and_action_info(self, state):
        """Produces actions using the policy and the state
        args:
            state: state of the environment
        returns:
            action: action for the agent to play
            action_probabilities: what probability the policy had of outputting action
            log_action_probabilities: log(action_probabilities)
            max_probability_action: the action that had the highest probability of being chosen"""
        actor_output = self.actor_local(state)
        action_probabilities =  self.softmax(actor_output)
        if random.random() < 0.001: print("Action probabilities ", action_probabilities)
        if (self.fixed_epsilon is not False or isinstance(self.fixed_epsilon, float)) and random.random() < self.fixed_epsilon:
            max_probability_action = torch.randint(0, self.original_num_primitive_actions, (1,))
        else:
            max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        if random.random() < 0.5: self.exploratory_move = True
        else: self.exploratory_move = False
        if self.normalise_learning:
            log_action_probabilities = log_action_probabilities / state.shape[1]
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration
         args:
            eval_ep: boolean indicating whether it is an evaluation episode
        returns:
            action: action for the agent to play"""
        if state is None: state = self.state
        state = self.prepare_state(state)
        if eval_ep: action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["initial_random_steps"]:

            if self.original_num_primitive_actions == self.action_size:
                action = self.environment.action_space.sample()
                print("Picking random action ", action)

            else:
                macro_action_weighting = [self.macro_action_exploration_weight for
                                          macro_action in self.global_action_id_to_primitive_action.keys() if
                                          macro_action >= self.original_num_primitive_actions]
                weighting = [1 for _ in range(self.original_num_primitive_actions)]
                weighting.extend(macro_action_weighting)
                probabilities = weighting / np.sum(weighting)
                action = choice(range(self.action_size), 1, p=probabilities)[0]
                print("Picking action randomly with bonus for macro action ", action)
        else: action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            self.action += self.noise.sample()
        return action

    def run_n_steps(self, num_steps):
        """Runs n steps in the environment
        args:
            num_steps: number of steps for the agent to run
        returns:
            episode_actions_scores_and_exploration_status: scores achieved in episodes played along with actions played
                                                           in those episodes"""
        self.episode_actions_scores_and_exploration_status = []
        num_steps_to_get_to = self.global_step_number + num_steps
        if any(substring in self.environment_title for substring in ["Pong", "CrazyClimber", "BattleZone", "Jamesbond"]):
            eval_eps_every = 4
        elif any(substring in self.environment_title for substring in ["Enduro", "Freeway"]): eval_eps_every = 2
        else: eval_eps_every = 10
        while self.global_step_number < num_steps_to_get_to:
            self.reset_game()
            if self.episode_number % eval_eps_every == 0 and self.global_step_number > self.hyperparameters["initial_random_steps"] \
                    and self.global_step_number < self.hyperparameters["steps_per_round"]:
                no_exploration_ep = True
            else: no_exploration_ep = False
            self.play_episode(no_exploration=no_exploration_ep)
            self.save_and_print_result()
        return self.episode_actions_scores_and_exploration_status

    def play_episode(self, eval_ep=False, no_exploration=False):
        """Runs an episode on the game, saving the experience and running a learning step if appropriate
        args:
            eval_ep: boolean to indicate whether it is an evaluation episode
            no_exploration: boolean to indicate whether to turn off exploration"""
        self.total_episode_score_so_far = 0
        episode_macro_actions = []
        while not self.done:
            changed_action = False
            macro_action = self.pick_action(eval_ep=eval_ep or no_exploration)
            episode_macro_actions.append(macro_action)
            primitive_actions = self.global_action_id_to_primitive_action[macro_action]
            if eval_ep: self.attempted_move_lengths.append(len(primitive_actions))
            macro_reward = 0
            primitive_actions_conducted = 0
            for action in primitive_actions:
                if self.hyperparameters["abandon_ship"] > 0.0 and primitive_actions_conducted >= 1 and (eval_ep or not self.exploratory_move) \
                        and self.global_step_number > self.hyperparameters["initial_random_steps"]: # (evaluate or not self.random_action_move):
                    action, changed_action = self.abandon_ship(self.state, action)
                    if changed_action: break
                self.conduct_action(action)
                macro_reward += self.reward
                primitive_actions_conducted += 1
                self.track_episodes_data(self.state, action, self.reward, self.next_state, self.done)
                if not eval_ep: self.global_step_number += 1
                self.state = self.next_state
                if self.time_for_critic_and_actor_to_learn() and not eval_ep:
                    for _ in range(self.hyperparameters["learning_iterations"]):
                        self.learn()
                if not eval_ep and self.global_step_number % self.hyperparameters["Critic"][
                    "update_fixed_network_freq"] == 0:
                    self.copy_model_over(self.critic_local, self.critic_target)
                    self.copy_model_over(self.critic_local_2, self.critic_target_2)
                if self.done or changed_action: break
            if eval_ep: self.move_lengths.append(primitive_actions_conducted)
        self.last_1000_actions.extend(episode_macro_actions)
        if random.random() < 0.1:
            proportions = self.calc_action_usage_statistics()
            print("Ep Score {} -- Action distribution {} -- Actions {}".format(self.total_episode_score_so_far, proportions, self.global_action_id_to_primitive_action))
        self.store_information(eval_ep=eval_ep, no_exploration=no_exploration)
        if eval_ep: self.print_summary_of_latest_evaluation_episode()

    def store_information(self, eval_ep, no_exploration):
        """Stores all required information after running an episode of the game
        args:
            eval_ep: boolean indicating whether it is an evaluation episode
            no_exploration: boolean to indicate whether to turn off exploration"""
        if not eval_ep:
            self.memory_shaper.add_episode_experience(self.episode_states, self.episode_next_states, self.episode_rewards,
                                                      self.episode_actions, self.episode_dones)
            experiences = (self.episode_states, self.episode_next_states, self.episode_rewards, self.episode_actions, self.episode_dones)
            actions_to_action_id = {v: k for k, v in self.global_action_id_to_primitive_action.items()}
            self.memory_shaper.hindsight_action_replay(None, actions_to_action_id, self.memory, self.original_num_primitive_actions, experiences=experiences)
            self.episode_number += 1
        if len(self.macro_action_disadvantages) > 0 and self.episode_number % 10 == 0 and not eval_ep:
            self.macro_action_disadvantage_mean = np.mean(self.macro_action_disadvantages)
            self.macro_action_disadvantage_std = np.std(self.macro_action_disadvantages)
            self.macro_abandonments_avg = np.mean(self.macro_abandonments)
            print("Macro action disadvantage mean {} -- std {} -- abandonments {}".format(self.macro_action_disadvantage_mean, self.macro_action_disadvantage_std, self.macro_abandonments_avg))
            print("Alpha ", self.alpha)
        if no_exploration:
            self.save_episode_actions_with_score()


    def abandon_ship(self, state, action):
        """Checks if action we are going to do is much worse than the best action we could do in terms of q_value. Then it
        changes the action if it is worse enough
        args:
            state: state of the environment
            action: proposed action
        returns:
            action: action to execute
            changed_action: boolean indicating whether action got changed"""
        with torch.no_grad():
            q_values = self.critic_local(self.prepare_state(state))[:, :self.original_num_primitive_actions]
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


    def update_final_layers(self, num_actions_before, num_new_actions):
        """Appends to the end of a network to allow it to choose from the new actions. It does not change the weights
        for the other actions
        args:
            num_actions_before: int number of actions we had before the most recent change to action set
            num_new_actions: int number of new actions just added"""
        print("Appending options to final layer")
        assert num_new_actions > 0

        new_actor_layer = self.transfer_learning_new_weights(self.actor_local, self.actor_local, num_actions_before, num_new_actions)
        new_critic_1_layer = self.transfer_learning_new_weights(self.critic_local, self.critic_target, num_actions_before, num_new_actions)
        new_critic_2_layer = self.transfer_learning_new_weights(self.critic_local_2, self.critic_target_2, num_actions_before, num_new_actions)

        self.actor_local.output_layers.append(new_actor_layer)
        self.critic_local.output_layers.append(new_critic_1_layer)
        self.critic_local_2.output_layers.append(new_critic_2_layer)

        self.critic_target.output_layers.append(
            nn.Linear(in_features=self.critic_target.output_layers[0].in_features,
                      out_features=num_new_actions))
        self.critic_target_2.output_layers.append(
            nn.Linear(in_features=self.critic_target_2.output_layers[0].in_features,
                      out_features=num_new_actions))

        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"])
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"])
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"])

        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.actor_local = self.actor_local.to(self.device)
        self.critic_local = self.critic_local.to(self.device)
        self.critic_local_2 = self.critic_local_2.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
        self.critic_target_2 = self.critic_target_2.to(self.device)

        # So that it runs some random steps to get some experiences of the macro actions
        self.hyperparameters["initial_random_steps"] = self.global_step_number + self.post_habit_inference_random_steps


