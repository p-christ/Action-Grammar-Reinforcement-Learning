import copy

import numpy as np
import gspread
import os
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


class Experiment(object):
    """Runs an experiment and saves the results in a google sheet"""
    def __init__(self, base_config, agent_type, spreadsheet_id):
        self.agent_type = agent_type
        self.base_config = base_config
        self.runs_per_agent = base_config.runs_per_agent
        self.hyperparameters = base_config.hyperparameters
        self.hyperparameter_list = ["Score", "Name"]
        self.hyperparameter_list.extend(sorted(list(self.hyperparameters.keys())))
        self.spreadsheet_id = spreadsheet_id

    def run_and_save_every_n_steps_to_csv(self, steps, experiment_name, do_evaluation=True, eval_epsilon=0.05, num_eval_eps=30, num_deterministic_eps=3):
        """Runs an experiment and saves the results periodically to a csv file"""
        results = {"max_result_seen": [], "steps": [], "run": [], "random_seed": [], "deterministic_policy_result": [],
                   "no_ops_policy_result": [], "no_ops_policy_ALL_results": [], "attempted_move_lengths": [], "move_lengths": []}
        config = self.base_config
        assert not os.path.isfile(experiment_name + ".csv")
        for run in range(config.runs_per_agent):
            max_result_seen = float("-inf")
            agent = self.agent_type(config)
            num_rounds = int(config.num_steps_to_run / steps)
            print("NUM ROUNDS ", num_rounds)
            for ix in range(num_rounds):
                steps_so_far = agent.global_step_number
                ep_scores, _, _ = agent.run_n_steps(num_steps=steps_so_far + steps)

                result = max(max_result_seen, max(ep_scores))
                if result > max_result_seen:
                    max_result_seen = result

                if do_evaluation:

                    print(" ")
                    print("------------------------------------------------------------------------------")
                    print("EVALUATION TIME")

                    if num_deterministic_eps > 0:

                        ep_scores, _, _ = agent.atari_evaluation(epsilon=eval_epsilon, episodes=num_deterministic_eps)
                        deterministic_policy_result = np.mean(ep_scores[-num_deterministic_eps:])
                        print("Deterministic scores: ", ep_scores)
                    # assert ep_scores[-2] == ep_scores[-1]

                    ep_scores, attempted_move_lengths, move_lengths = agent.atari_evaluation(epsilon=eval_epsilon, episodes=num_eval_eps, do_no_ops=True)
                    no_op_results = ep_scores[-num_eval_eps:]
                    assert len(no_op_results) == num_eval_eps
                    no_op_policy_results = np.mean(no_op_results)
                    print("No op results ", no_op_results)
                    print("Best no op result ", no_op_policy_results)

                    results["attempted_move_lengths"].append(attempted_move_lengths)
                    results["move_lengths"].append(move_lengths)
                    results["max_result_seen"].append(max_result_seen)

                    if num_deterministic_eps == 0: deterministic_policy_result = "N/A"
                    results["deterministic_policy_result"].append(deterministic_policy_result)

                    results["no_ops_policy_result"].append(no_op_policy_results)
                    results["no_ops_policy_ALL_results"].append(no_op_results)
                    results["steps"].append(steps * (ix+1))
                    results["random_seed"].append(config.seed)
                    results["run"].append(run)
                    df = pd.DataFrame.from_dict(results)
                    print("SAVING TO CSV")

                    df.to_csv(experiment_name + ".csv")
                    print("SAVED TO CSV")

                    print("------------------------------------------------------------------------------")
        return results