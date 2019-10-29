from operator import itemgetter
import copy
import numpy as np
from utilities.grammar_algorithms.IGGI import IGGI
from utilities.Utility_Functions import flatten_action_id_to_actions
from utilities.grammar_algorithms.k_Sequitur import k_Sequitur

class Grammar_Generator(object):
    """Takes as input the actions in the best performing episodes and prdocues an updated action list"""
    def __init__(self, num_top_results_to_use, action_size,
                 logger, sequitur_k,  action_frequency_required_in_top_results,
                 max_macro_length, use_raw_rule_counts, grammar_algorithm="Sequitur", end_of_episode_symbol = "/"):
        self.max_macro_length = max_macro_length
        self.global_list_of_best_results = []
        self.num_top_results_to_use = num_top_results_to_use
        self.action_size = action_size
        self.end_of_episode_symbol = end_of_episode_symbol
        self.sequitur_k = sequitur_k
        self.action_frequency_required_in_top_results = action_frequency_required_in_top_results
        self.use_raw_rule_counts = use_raw_rule_counts
        self.grammar_algorithm = grammar_algorithm
        self.iggi = IGGI()

    def generate_new_grammar(self, episode_actions_scores_and_exploration_status, global_action_id_to_primitive_action):
        """Infers a new action grammar and updates the global action set"""
        print("Episodes of data provided  {}".format(len(episode_actions_scores_and_exploration_status)))

        self.action_id_to_action = copy.deepcopy(global_action_id_to_primitive_action)

        if len(episode_actions_scores_and_exploration_status) == 0 and len(self.global_list_of_best_results) == 0:
            return self.action_id_to_action, []

        print("ACTIONS BEFORE ", self.action_id_to_action)
        good_actions = self.pick_actions_to_infer_grammar_from(episode_actions_scores_and_exploration_status)
        num_actions_before = len(global_action_id_to_primitive_action)

        print("Actions eps to use {} -- action set to use: ".format(self.global_list_of_best_results, good_actions))


        self.update_action_choices(good_actions)
        self.check_new_global_actions_valid()
        new_actions_just_added = list(range(num_actions_before, num_actions_before + len(self.action_id_to_action) - num_actions_before))
        print("New Actions {} -- ACTIONS AFTER {}".format(new_actions_just_added, self.action_id_to_action))
        return self.action_id_to_action, new_actions_just_added

    def pick_actions_to_infer_grammar_from(self, episode_actions_scores_and_exploration_status, greater_than_condition=True):
        """Takes in data summarising the results of the latest games the agent played and then picks the actions from which
        we want to base the subsequent action grammar on"""
        if len(episode_actions_scores_and_exploration_status) != 0:
            episode_scores = [data[0] for data in episode_actions_scores_and_exploration_status]
            episode_actions = [data[1] for data in episode_actions_scores_and_exploration_status]
            reverse_ordering = np.argsort(episode_scores)
            top_result_indexes = list(reverse_ordering[-self.num_top_results_to_use:])

            if len(top_result_indexes) == 1:
                best_episode_actions = [episode_actions[top_result_indexes[0]]]
                best_episode_rewards = [episode_scores[top_result_indexes[0]]]
            else:
                best_episode_actions = list(itemgetter(*top_result_indexes)(episode_actions))
                best_episode_rewards = list(itemgetter(*top_result_indexes)(episode_scores))

            best_result_this_round = max(best_episode_rewards)

            if len(self.global_list_of_best_results) == 0:
                worst_best_result_ever = float("-inf")
            else:
                worst_best_result_ever = min([data[0] for data in self.global_list_of_best_results])

            if greater_than_condition: condition = best_result_this_round > worst_best_result_ever
            else: condition = best_result_this_round >= worst_best_result_ever

            if condition:
                combined_best_results = [(result, actions) for result, actions in zip( best_episode_rewards, best_episode_actions)]
                self.global_list_of_best_results, new_old_best_results = self.keep_track_of_best_results_seen_so_far(
                    self.global_list_of_best_results, combined_best_results)
        best_episode_actions = [data[1] for data in self.global_list_of_best_results]
        print("New best results ", [data[0] for data in self.global_list_of_best_results])
        print("Num eps in best list ", len(self.global_list_of_best_results))
        best_episode_actions = [item for sublist in best_episode_actions for item in sublist]
        return best_episode_actions

    def keep_track_of_best_results_seen_so_far(self, global_results, combined_result):
        """Keeps a track of the top episode results so far & the actions played in those episodes"""
        global_results += combined_result
        global_results.sort(key=lambda x: x[0], reverse=True)
        global_results, old_best_results = global_results[:self.num_top_results_to_use], global_results[self.num_top_results_to_use:]
        assert isinstance(global_results, list)
        if len(global_results) > 0:
            assert isinstance(global_results[0], tuple)
            assert len(global_results[0]) == 2
        return global_results, old_best_results

    def check_new_global_actions_valid(self):
        """Checks that global_action_id_to_primitive_action still only has valid entries"""
        assert len(set(self.action_id_to_action.values())) == len(
            self.action_id_to_action.values()), \
            "Not all actions are unique anymore: {}".format(self.action_id_to_action)
        for key, value in self.action_id_to_action.items():
            assert max(value) < self.action_size, "Actions should be in terms of primitive actions"

    def update_action_choices(self, good_actions):
        """Creates a grammar out of the latest list of macro actions conducted by the agent"""
        if self.grammar_algorithm == "IGGI":
            new_grammar_rules = self.iggi.generate_action_grammar(good_actions, len(self.action_id_to_action), max_rule_length=self.max_macro_length)
            new_actions = flatten_action_id_to_actions(new_grammar_rules, self.action_id_to_action, self.action_size)
        else:
            good_episode_rule_appearance = self.get_rule_appearance_count(good_actions)
            new_actions = self.pick_new_macro_actions(good_episode_rule_appearance)
        self.update_global_action_id_to_primitive_action(new_actions)

    def get_rule_appearance_count(self, actions):
        """Takes as input a list of actions and infers rules using Sequitur and then returns a dictionary indicating how
        many times each rule was used"""
        if actions == []: return {}
        grammar_calculator = k_Sequitur(k=self.sequitur_k,
                                        end_of_episode_symbol=self.end_of_episode_symbol)
        print("latest_macro_actions_seen ", actions)
        _, _, raw_rules_appearance_count, rules_episode_appearance_count = grammar_calculator.generate_action_grammar(actions)
        print("NEW rules_episode_appearance_count ", rules_episode_appearance_count)

        if self.use_raw_rule_counts: return raw_rules_appearance_count
        else: return rules_episode_appearance_count

    def update_global_action_id_to_primitive_action(self, new_actions):
        """Updates global_action_id_to_primitive_action by adding any new actions in that aren't already represented"""
        print("update_global_action_id_to_primitive_action ", new_actions)
        unique_new_actions = {k: v for k, v in new_actions.items() if v not in self.action_id_to_action.values()}
        print("Unique new actions ", unique_new_actions)
        next_action_name = max(self.action_id_to_action.keys()) + 1
        for _, value in unique_new_actions.items():
            print("adding {} -- {}".format(next_action_name, value))
            self.action_id_to_action[next_action_name] = value
            next_action_name += 1

    def calculate_cutoff_for_macro_actions(self):
        """Calculates how many times a macro action needs to appear before we include it in the action set"""
        cutoff = self.num_top_results_to_use * self.action_frequency_required_in_top_results
        return cutoff

    def restrict_to_short_enough_macro_actions(self, proposed_actions):
        """Takes a dictionary of form {action_id: (primitive actions)} and returns an equivalent dictionary with only
        those macro actions below a certain length"""
        short_enough_macros = {}
        for action_id, primitive_actions in proposed_actions.items():
            if len(primitive_actions) <= self.max_macro_length:
                short_enough_macros[action_id] = primitive_actions
        return short_enough_macros

    def pick_new_macro_actions(self, rules_episode_appearance_count):
        """Picks the new macro actions to be made available to the agent. Returns them in the form {action_id: (action_1, action_2, ...)}.
        NOTE there are many ways to do this... i should do experiments testing different ways and report the results
        """
        new_unflattened_actions = {}
        cutoff = self.calculate_cutoff_for_macro_actions()
        action_id = len(self.action_id_to_action.keys())
        counts = {}
        for rule in rules_episode_appearance_count.keys():
            count = rules_episode_appearance_count[rule]
            print("Rule {} -- Count {}".format(rule, count))
            if count >= cutoff:
                new_unflattened_actions[action_id] = rule
                counts[action_id] = count
                action_id += 1
        new_actions = flatten_action_id_to_actions(new_unflattened_actions, self.action_id_to_action,
                                                   self.action_size)
        new_actions = self.restrict_to_short_enough_macro_actions(new_actions)
        return new_actions

    def reset(self):
        """Resets the data so that it doesn't keep track of results from past episode rounds"""
        self.global_list_of_best_results = []
