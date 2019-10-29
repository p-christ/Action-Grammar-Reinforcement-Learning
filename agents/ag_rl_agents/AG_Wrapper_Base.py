from collections import Counter
import torch
import numpy as np
from torch import nn

class AG_Wrapper_Base(object):
    """Base class that all Habit RL wrappers inherit from"""
    def __init__(self):
        self.atari_eval_ep_info = []

    def update_final_layers(self):
        raise NotImplementedError

    def atari_evaluation(self, epsilon, episodes, do_no_ops):
        """Evaluates the agent using the methodology from the Deepmind 2015 atari paper
        https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
        args:
            - epsilon: float to indicate the epsilon we want to enforce in an epsilon greedy policy
            - episodes: the number of evaluation episodes we want to run
            - do_no_ops: boolean to indicate whether we want to use the no ops start condition described in Deepmind 2015 paper
        returns:
            - game_full_episode_scores: list of total scores achieved in the evaluation episodes
            - mean attempted move lenght: mean length of moves attempted during evaluation
                                         (will equal 1 if there are no macro actions with more than 1 action)
            - mean move lengths: mean length of moves actually conducted during evaluation (will equal 1 if there are no
                                 macro actions with more than 1 action)"""
        self.fixed_epsilon = epsilon
        self.environment = self.eval_env
        self.move_lengths = []
        self.attempted_move_lengths = []
        self.episode_actions_scores_and_exploration_status = []
        if do_no_ops:
            self.do_no_ops = True
        for _ in range(episodes):
            self.reset_game()
            self.play_episode(eval_ep=True)
            self.game_full_episode_scores.append(self.total_episode_score_so_far)
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_episode_actions_with_score()
        self.fixed_epsilon = False
        self.do_no_ops = False
        self.environment = self.training_env
        self.atari_eval_ep_info.extend(self.episode_actions_scores_and_exploration_status)
        return self.game_full_episode_scores, np.mean(self.attempted_move_lengths), np.mean(self.move_lengths)

    def transfer_learning_new_weights(self, network, target_network, num_actions_before, num_new_actions):
        """Makes the weights in the new layers just added equal to the current weights for their base primitive action.
        args:
            network: the new network we want to copy the weights from
            target_network:  the target network we want to copy the weights from
            num_actions_before: the number of actions in the action set before new actions were added
            num_new_actions: the number of new actions added
        returns:
            layer: a final layer of the correct size with the transfer learning applied to it"""
        new_layer = nn.Linear(in_features=network.output_layers[0].in_features,
                              out_features=num_new_actions)
        weights_done = False
        bias_done = False
        primitive_actions_layer = target_network.output_layers[0]
        weights = None
        bias = None
        # First we store the current weights in the final layer
        for m in primitive_actions_layer.named_parameters():
            if m[0] == "weight":
                assert weights is None
                weights = m[1]
                print("Weights to copy ", weights)
            elif m[0] == "bias":
                assert bias is None
                bias = m[1]
                print("Bias to copy ", bias)
            else:
                raise ValueError
        # Then we copy them over to a new layer
        for m in new_layer.named_parameters():
            print("Before ", m[1])
            if m[0] == "weight":
                assert weights_done is False
                for new_act in range(num_new_actions):
                    stepping_stone = self.action_id_to_primitive_stepping_stone_id[num_actions_before + new_act]
                    print("Copying action id {} weights to new action id {}".format(stepping_stone, new_act + num_actions_before))
                    m[1][new_act].data.copy_(weights[stepping_stone].data.clone())
            elif m[0] == "bias":
                assert bias_done is False
                for new_act in range(num_new_actions):
                    stepping_stone = self.action_id_to_primitive_stepping_stone_id[num_actions_before + new_act]
                    print("Copying action id {} bias to new action id {}".format(stepping_stone,
                                                                                    new_act + num_actions_before))
                    m[1][new_act].data.copy_(bias[stepping_stone].data.clone())
            else:
                raise ValueError
            print("After ", m[1])
        return new_layer

    def calc_action_usage_statistics(self):
        """Calculates how often each action was played in the last 1000 actions"""
        counter = Counter(self.last_1000_actions)
        proportions = ["{}: {}%".format(action, 100.0*count / 1000.0) for action, count in counter.items()]
        return proportions


    def update_replay_buffer(self):
        """Updates the replay buffer by using the Hindsight Action Replay technique"""
        replay_buffer = self.memory_shaper.put_adapted_experiences_in_a_replay_buffer(self.global_action_id_to_primitive_action)
        assert replay_buffer is not None
        self.memory = replay_buffer

    def update_agent(self, global_action_id_to_primitive_action):
        """Updates the agent according to new action set by changing its action set and updating the replay buffer"""
        self.update_agent_for_new_actions(global_action_id_to_primitive_action)
        self.update_replay_buffer()

    def update_agent_for_new_actions(self, global_action_id_to_primitive_action):
        """Adds nodes to the final layers of an agent's networks if new actions have been added to the action set
         args:
            global_action_id_to_primitive_action: the action set"""
        num_actions_before = self.action_size
        self.global_action_id_to_primitive_action = global_action_id_to_primitive_action
        self.action_size = len(global_action_id_to_primitive_action)
        num_new_actions = self.action_size - num_actions_before
        if num_new_actions > 0:
            for new_action_id in range(num_actions_before, num_actions_before + num_new_actions):
                self.update_action_id_to_stepping_stone_action_id(new_action_id)
            self.update_final_layers(num_actions_before, num_new_actions)

    def update_action_id_to_stepping_stone_action_id(self, new_action_id):
        """Update action_id_to_stepping_stone_action_id with the new actions created. action_id_to_stepping_stone_action_id
        is a dictionary that stores the action id of the first primitive action of any macro-action.
        args:
            new_action_id: the action_id of the new action"""
        new_action = self.global_action_id_to_primitive_action[new_action_id]
        length_macro_action = len(new_action)

        self.action_id_to_primitive_stepping_stone_id[new_action_id] = new_action[0]

        print(" update_action_id_to_stepping_stone_action_id ")
        for sub_action_length in reversed(range(1, length_macro_action)):
            sub_action = new_action[:sub_action_length]
            if sub_action in self.global_action_id_to_primitive_action.values():
                sub_action_id = list(self.global_action_id_to_primitive_action.keys())[
                    list(self.global_action_id_to_primitive_action.values()).index(sub_action)]

                self.action_id_to_stepping_stone_action_id[new_action_id] = sub_action_id
                print("Action {} has largest sub action {}".format(new_action_id, sub_action_id))
                break

    def macro_actions_starting_with_a(self, action):
        """Returns the action_id of macro actions that start with the given action
        params:
            action: integer to represent an action we want the macro actions to begin with
        returns:
            results: list of macro action ids where the macro action begins with the given action"""
        results = []
        for action_id, primitive_actions in self.global_action_id_to_primitive_action.items():
            if len(primitive_actions) > 1:
                if primitive_actions[0] == action:
                    results.append(action_id)
        return results

    def how_much_worse(self, action_values, q_value_A, q_value_B):
        """Calculates the proportional difference between 2 q_values, telling you how much worse/better is q_value_A than
        q_value_B
        params:
            action_values: torch tensor of all q_values for given state
            q_value_A: the q_value for a given action
            q_value_B: the q_value for another action
        returns:
            difference: the proportional difference in the q_values after the mean q_value has been substracted"""
        primitive_action_values = action_values[:, :self.original_num_primitive_actions]
        avg_value = torch.mean(primitive_action_values)
        q_value_A -= avg_value
        q_value_B -= avg_value
        difference = 1.0 -  (torch.exp(q_value_B) / torch.exp(q_value_A))
        assert difference <= 1.0 and difference >= 0.0, difference
        return difference