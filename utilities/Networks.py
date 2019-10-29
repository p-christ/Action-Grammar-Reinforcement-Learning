import torch
import torch.nn as nn
from nn_builder.pytorch.CNN import CNN
import os
import numpy as np
from stable_baselines import DQN

def load_pre_trained_model(env_id):
    path = os.getcwd()
    print(path)
    project_name = "Habit-Reinforcement-Learning"
    end_string = len(project_name)
    end = path[-end_string:]
    while end != project_name:
        path = path[:-1]
        end = path[-end_string:]
    path = path + "/pre_trained_dqn_agents/{}NoFrameskip-v4.pkl".format(env_id)
    baselines_cnn_model = DQN.load(path, verbose=2)
    print("DONE")
    return baselines_cnn_model


def load_atari_cnn_pretrained(env_id, output_dim, seed):
    """Loads a pre-trained Atari CNN"""
    baselines_cnn_model = load_pre_trained_model(env_id)
    cnn = CNN(input_dim=(4, 84, 84),
              layers_info=[["conv", 32, 8, 4, 0], ["conv", 64, 4, 2, 0], ["conv", 64, 3, 1, 0], ["linear", 512],
                           ["linear", output_dim]],
              converted_from_tf_model=True, initialiser="he", random_seed=seed)
    cnn = copy_baselines_model_over(baselines_cnn_model, cnn)
    print("Freezing hidden layers")
    for param in cnn.named_parameters():
        param_name = param[0]
        assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, "Name {} of network layers not understood".format(
            param_name)
        if "output" not in param_name and "hidden_layers.3" not in param_name:
            print("Freezing ", param_name)
            param[1].requires_grad = False
    return cnn

def create_atari_network(output_dim, seed, softmax_final_layer):
    """Creates and returns a CNN of same architecture as the Nature 2015 Deepmind Atari paper"""
    if softmax_final_layer:
        cnn = CNN(input_dim=(4, 84, 84),
                  layers_info=[["conv", 32, 8, 4, 0], ["conv", 64, 4, 2, 0], ["conv", 64, 3, 1, 0], ["linear", 512],
                               ["linear", output_dim]], output_activation="Softmax",
                  converted_from_tf_model=True, initialiser="he", random_seed=seed)
    else:
        cnn = CNN(input_dim=(4, 84, 84),
                  layers_info=[["conv", 32, 8, 4, 0], ["conv", 64, 4, 2, 0], ["conv", 64, 3, 1, 0], ["linear", 512],
                               ["linear", output_dim]], converted_from_tf_model=True, initialiser="he", random_seed=seed)
    return cnn


def copy_baselines_model_over(baselines_cnn_model, cnn):
    """Copies the weights of a Open AI stable-baselines CNN model over to the CNN provided"""
    model_params = baselines_cnn_model.get_parameters()
    policy_keys = [key for key in model_params.keys() if
                   ("pi" in key or "c" in key) and not "target" in key and not "state_value" in key]

    print(policy_keys)

    policy_keys = [key for key in policy_keys if not "fully_connected" in key
                   and not "fc1" in key]

    policy_params = [model_params[key] for key in policy_keys]
    assert len(policy_keys) == 6, policy_keys

    for (th_key, pytorch_param), key, policy_param in zip(cnn.named_parameters(), policy_keys, policy_params):
        print(key)
        param = policy_param.copy()
        # Copy parameters from stable baselines model to pytorch model
        print(param.shape)
        print(pytorch_param.shape)
        # Conv layer
        if len(param.shape) == 4:
            # https://gist.github.com/chirag1992m/4c1f2cb27d7c138a4dc76aeddfe940c2
            # Tensorflow 2D Convolutional layer: height * width * input channels * output channels
            # PyTorch 2D Convolutional layer: output channels * input channels * height * width
            param = np.transpose(param, (3, 2, 0, 1))

        # weight of fully connected layer
        if len(param.shape) == 2:
            param = param.T

        # bias
        if 'b' in key:
            param = param.squeeze()

        param = torch.from_numpy(param)
        pytorch_param.data.copy_(param.data.clone())
    return cnn


class Atari_CNN(nn.Module):
    """Initialises a CNN with same specifications as the Deepmind 2013 Atari paper"""
    def __init__(self, output_dim):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

        self.linear_layers = nn.ModuleList([self.conv1, self.conv2, self.conv3, self.fc1, self.fc2])
        for param in self.linear_layers:
            nn.init.kaiming_normal_(param.weight)

    def initialise_parameters(self, parameters_list):
        """Initialises the list of parameters given"""
        initialiser = self.str_to_initialiser_converter[self.initialiser.lower()]
        if initialiser != "use_default":
            for parameters in parameters_list:
                if type(parameters) == nn.Linear:
                    initialiser(parameters.weight)
                elif type(parameters) in [nn.LSTM, nn.RNN, nn.GRU]:
                    initialiser(parameters.weight_hh_l0)
                    initialiser(parameters.weight_ih_l0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_atari_CNN_model(output_dim, seed):
    """Initialises a CNN with same specifications as the Deepmind 2013 Atari paper"""
    model = CNN(input_dim=(4, 84, 84), layers_info=[ ["conv", 32, 8, 4, 0], ["conv", 64, 4, 2, 0], ["conv", 64, 3, 1, 0], ["linear", 512], ["linear", output_dim]],
               initialiser="he", random_seed=seed)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model


