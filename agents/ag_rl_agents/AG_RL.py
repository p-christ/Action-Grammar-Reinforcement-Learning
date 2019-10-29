import time
from agents.habit_rl_agents.Habit_DDQN_Wrapper import Habit_DDQN_Wrapper
from utilities.grammar_algorithms.Grammar_Generator import Grammar_Generator
from agents.habit_rl_agents.Habit_SAC_Wrapper import Habit_SAC_Wrapper

class AG_RL(object):
    agent_name = "HRL"
    def __init__(self, config):
        assert config.hyperparameters["Base_Agent"] in ["DDQN", "SAC"], "Base agent must be DDQN or SAC"
        if config.hyperparameters["Base_Agent"] == "DDQN": self.agent = Habit_DDQN_Wrapper(config)
        else: self.agent = Habit_SAC_Wrapper(config)
        self.config = config
        self.action_id_to_action = {k: tuple([k]) for k in range(self.agent.action_size)}
        self.hyperparameters = config.hyperparameters
        self.grammar_generator = Grammar_Generator(self.hyperparameters["evaluation_eps"], self.agent.action_size, None,
                                                   self.hyperparameters["sequitur_k"],
                                                   self.hyperparameters["action_frequency_required_in_top_results"],
                                                   self.hyperparameters["max_macro_length"], self.hyperparameters["use_raw_rule_counts"],
                                                   grammar_algorithm=self.hyperparameters["grammar_algorithm"])
        self.global_step_number = 0
        self.grammar_iteration = 0
        self.max_number_of_grammar_iterations = 1

    def run_n_steps(self, num_steps=None):
        """Runs n steps of the game for the Habit RL agent including grammar iterations
        args:
            num_steps: the number of steps we want to run. If left blank then we use config.num_steps_to_run
        returns:
            game_full_episode_scores: list of scores for each episode played
            rolling_results: list of rolling mean of scores for episodes played
            time_taken: number of seconds the method took
        """
        self.steps_conducted = 0
        start = time.time()
        if num_steps is None: num_steps = self.config.num_steps_to_run
        self.episodes_conducted = 0
        min_score_to_make_it_into_top_list = float("-inf")
        while self.global_step_number < num_steps:
            steps_until_next_grammar_iteration =  (self.grammar_iteration + 1) * self.hyperparameters["steps_per_round"] - self.global_step_number
            if steps_until_next_grammar_iteration <=  num_steps - self.global_step_number and self.grammar_iteration < self.max_number_of_grammar_iterations:
                steps_this_round = steps_until_next_grammar_iteration
                grammar_iteration_round = True
            else:
                steps_this_round = num_steps - self.global_step_number
                grammar_iteration_round = False
            live_playing_data = self.agent.run_n_steps(num_steps=steps_this_round, min_score=min_score_to_make_it_into_top_list)
            self.global_step_number += steps_this_round
            if grammar_iteration_round and not self.grammar_iteration >= self.max_number_of_grammar_iterations:
                self.action_id_to_action, new_actions_just_added =  self.grammar_generator.generate_new_grammar(live_playing_data,
                                                                                                               self.action_id_to_action)
                self.agent.update_agent(self.action_id_to_action, new_actions_just_added)
                self.grammar_iteration += 1
            else:
                if len(live_playing_data) > 0:
                    self.grammar_generator.pick_actions_to_infer_grammar_from(live_playing_data)
        time_taken = time.time() - start
        return self.agent.game_full_episode_scores, self.agent.rolling_results, time_taken

    def atari_evaluation(self, epsilon, episodes, do_no_ops=False):
        """Runs the evaluation procedure described in 2015 Deepmind Atari paper
        args:
            epsilon: the proportion of moves to choose randomly when running evaluation episodes
            episodes: the number of evaluation episodes to run
            do_no_ops: whether to use the no_ops start condition from the 2015 Deepmind Atari paper or not
        returns:
            game_full_episode_scores: list of scores for each episode played
            attempted_move_lengths: average attempted macro-action move lengths in the evaluation episodes
            move_lengths: average executed macro-action move lengths in the evaluation episodes
        """
        print("------------------------------------------------------")
        print("Evaluation")
        game_episode_scores, attempted_move_lengths, move_lengths = self.agent.atari_evaluation(epsilon, episodes, do_no_ops)
        print(" ")
        print("Evaluation Over")
        print("------------------------------------------------------")
        return game_episode_scores, attempted_move_lengths, move_lengths