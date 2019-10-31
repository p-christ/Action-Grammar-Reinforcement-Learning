import copy
import numpy as np
import torch
from torch.distributions import Categorical, normal, MultivariateNormal

def save_score_results(file_path, results):
    """Saves results as a numpy file at given path"""
    np.save(file_path, results)

def normalise_rewards(rewards):
    """Normalises rewards to mean 0 and standard deviation 1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8) #1e-8 added for stability

def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution

def flatten_action_id_to_actions(action_id_to_actions, global_action_id_to_primitive_action, num_primitive_actions):
    """Converts the values in an action_id_to_actions dictionary back to the primitive actions they represent"""
    global_action_id_to_primitive_action = copy.deepcopy(global_action_id_to_primitive_action)
    flattened_action_id_to_actions_dict = {}
    for key in sorted(action_id_to_actions.keys()):
        actions = action_id_to_actions[key]
        raw_actions = backtrack_action_to_primitive_actions(actions, global_action_id_to_primitive_action, num_primitive_actions)
        flattened_action_id_to_actions_dict[key] = raw_actions
        global_action_id_to_primitive_action.update(flattened_action_id_to_actions_dict)
    return flattened_action_id_to_actions_dict

def backtrack_action_to_primitive_actions(action_tuple, global_action_id_to_primitive_action, num_primitive_actions):
    """Converts an action tuple back to the primitive actions it represents in a recursive way."""
    print("Recursing to backtrack on ", action_tuple)
    primitive_actions = range(num_primitive_actions)
    if all(action in primitive_actions for action in action_tuple): return action_tuple #base case
    new_action_tuple = []
    for action in action_tuple:
        if action in primitive_actions: new_action_tuple.append(action)
        else:
            converted_action = global_action_id_to_primitive_action[action]
            print(new_action_tuple)
            new_action_tuple.extend(converted_action)
            print("Should have changed: ", new_action_tuple)
    new_action_tuple = tuple(new_action_tuple)
    return backtrack_action_to_primitive_actions(new_action_tuple, global_action_id_to_primitive_action, num_primitive_actions)

def prepare_lazy_frames_for_network(state):
    """Prepares the lazy frames states provided by an Atari environment to be input into a neural network"""
    state = np.array(state)
    if len(state.shape) == 3:
        state = np.expand_dims(state, axis=0)
    state = np.transpose(state, (0, 3, 1, 2))
    state = state / 255.0
    state = torch.tensor(state).float()
    return state