import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from IPython import embed
import numpy as np
agent = 0
dumper = 0
now = False
route_opt = 0


def set_route_opt(route_opt_):
    global route_opt
    route_opt = route_opt_


def set_agent(agent_):
    global now
    now = False
    global agent
    agent = agent_


def set_dumper(dumper_):
    global dumper
    dumper = dumper_


def set_now(now_):
    global now
    now = now_


class Linear_QNet(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(network) - 1):
            self.layers.append(nn.Linear(network[i], network[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        x = self.layers[-1](x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model_transport'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, node_agent=False):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.node_agent = node_agent
        self.alpha = 0.5

    def set_alpha(self, alpha):
        print('the alpha is set')
        self.alpha = alpha

    def train_step(self, state, action, reward, next_state, done):
        # TODO: extremly slow UserWarning
        try:
            # embed(header='state')
            state = torch.tensor(state, dtype=torch.float)
        except TypeError:
            embed(header='Type')
        try:
            next_state = torch.tensor(next_state, dtype=torch.float)
        except TypeError:
            embed(header='heree')

        action = torch.tensor(action, dtype=torch.long)
        try:
            reward = torch.tensor(reward, dtype=torch.float)
        except ValueError:
            embed(header='ValueError_model')
        # (n, x) # n number of batche

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)

            done = (done, )

        # 1: predicted Q values with the current state
        pred = self.model(state)

        # TODO: check out this
        target = pred.clone()
        for idx in range(len(done)):
            Q_old = target[idx][torch.argmax(action[idx]).item()]
            Q_new = (1 - self.alpha) * Q_old + self.alpha * reward[idx]

            # if not done[idx]:
            #     Q_new = reward[idx] + self.gamma * \
            #         torch.max(self.model(
            #             next_state[idx]))  # https://en.wikipedia.org/wiki/Q-learning, med learning rate 1
            # # embed()
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        # # TODO: desperate, try to make predictions lower
        # if self.node_agent:
        #     target0 = torch.full((target.shape), 0, dtype=torch.float)
        #     loss0 = self.criterion(pred, target0)

        loss = self.criterion(pred, target)

        # if self.node_agent:
        #     (loss + loss0).backward()
        # else:

        loss.backward()
        self.optimizer.step()

        if route_opt.lol:
            # print(self.model(state))
            dumper = route_opt.dumper
            node = route_opt.node
            # embed(header='look at model')

        # if Q_new == -10:
        #     if torch.argmax(action[idx]) == 28:
        #         embed(header='yooo28')
        # if Q_new > 0:
        #     if torch.argmax(action[idx]) == 49:
        #         embed(header='yooo49')
