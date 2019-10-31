from collections import namedtuple, deque
from multiprocessing.pool import ThreadPool
import random
import torch
import numpy as np


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""

    def __init__(self, buffer_size, batch_size, seed, data_saved_as_tensors=False, atari=False):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.data_saved_as_tensors = data_saved_as_tensors
        self.atari = atari
        # self.state_thread_pool = ThreadPool(self.batch_size)
        # self.next_state_thread_pool = ThreadPool(self.batch_size)

    def add_named_experience_tuples(self, experience_tuples):
        assert isinstance(experience_tuples, list), experience_tuples
        assert experience_tuples[0]._fields == ('state', 'action', 'reward', 'next_state', 'done')
        self.memory.extend(experience_tuples)

    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                           for state, action, reward, next_state, done in
                           zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)

    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, actions, rewards, next_states, dones
        else:
            return experiences

    # def atari_thread_state_sampling(self, experience):
    #     state = experience.state
    #     value = np.array(state)
    #     value = torch.Tensor(value)
    #     return value
    #
    # def atari_thread_next_state_sampling(self, experience):
    #     next_state = experience.next_state
    #     value = np.array(next_state)
    #     value = torch.Tensor(value)
    #     return value
    #
    # def prepare(self, state):
    #     value = np.array(state)
    #     value = torch.from_numpy(value)
    #     value = torch.unsqueeze(value, 0)
    #     return value

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        if self.atari:
            states = [e.state for e in experiences if e is not None]
            next_states = [e.next_state for e in experiences if e is not None]

            # states = self.state_thread_pool.map_async(self.prepare, states)
            # next_states = self.next_state_thread_pool.map_async(self.prepare, next_states)

        elif self.data_saved_as_tensors:
            states = torch.cat([e.state for e in experiences if e is not None], dim=0)
            next_states = torch.cat([e.next_state for e in experiences if e is not None], dim=0)
        else:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()

        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float()

        # if self.atari:
        #     states.wait()
        #     next_states.wait()

        # return torch.cat(states.get(), dim=0), actions, rewards, torch.cat(next_states.get(), dim=0), dones
        return states, actions, rewards, next_states, dones

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)