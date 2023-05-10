import time as timePackage
import sys
import signal
import matplotlib.pyplot as plt
import os
from IPython import embed
import model
from collections import deque
import numpy as np
import random
import maps
import helper

from game import TravelingGameAI
import torch

"""
Create Nodes before dumpers
    - dumpers uses nodes information


"""

# TODO: next, start up a model

# TODO: distance = 0, no edge between the nodes (can't go to itself)


# TODO: waiting time, driving time, start/end time to nearest second.

# EXPLORATION_NUM = 10


MAX_MEMORY = 1000
BATCH_SIZE = 100
LR = 0.001

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AgentNode:
    agents = []

    def __init__(self, node, nodes, nodes_loading_or_dumping, exploration_num=100):
        AgentNode.agents.append(self)
        self.n_games = 0
        self.exploration_counter = 0
        self.node = node
        self.nodes = nodes
        self.nodes_loading_or_dumping = nodes_loading_or_dumping
        self.exploration_num = exploration_num
        self.epsilon = 0    # randomness
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 5  # %

        # input, the number of
        self.input_size = len(self.nodes_loading_or_dumping)  # 7
        self.output_size = len(self.node.get_edges())  # up to 5-7
        if self.output_size == 0:
            print(
                'AGENT NODE (isolated): PROBABLY NEED TO DO SOMETHING, OR NOT INITIALIZE')
            # embed(
            #     header='AGENT NODE (isolated): PROBABLY NEED TO DO SOMETHING, OR NOT INITIALIZE')
        # network = [self.input_size, 30, 10, 30, self.output_size]
        self.initialize_network()

    def initialize_network(self):
        network = [self.input_size, 50, 20, 50, self.output_size]
        # network = [self.input_size, 200, 100, 200, 100, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(
            self.model, lr=LR, gamma=self.gamma, node_agent=True)

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt

    def get_state(self, dumper):

        # get what should be sendt into the model
        state = []

        for node in self.nodes_loading_or_dumping:
            state.append(dumper.get_on_my_way_to() == node)

        return np.array(state, dtype=int)  # maybe float

    def set_exploration_num(self, num):

        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_counter = 0
        self.exploration_num = num

    def set_random_choice_prob(self, prob):
        self.random_choice_prob = prob

    def get_random_choice_prob(self):
        return self.random_choice_prob

    def remember(self, state, action, reward, next_state, done):
        # TODO: why next_state and how to do this? Maybe wait for the next state to be executed, then put in
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # TODO fix this
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # If there is something to train
        if len(mini_sample) > 0:
            states, actions, rewards, next_states, dones = zip(*mini_sample)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            self.trainer.train_step(
                states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # TODO fix this TO go from
        # TODO: train_medium_memory load -> dump (then add all states, and actions for this route)
        # TODO: train_short_memory should be that you can't go from current node to itself,
        #      #############        or any other node that aren't connected.
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, dumper):
        # random moves: tradeoff exploration / exploitation

        self.epsilon = self.exploration_num - self.exploration_counter
        final_move = [0] * self.output_size

        # If random_move or, TODO: changed the exploration_num + 10
        random_move = random.randint(0, self.exploration_num) < self.epsilon or \
            random.randint(0, 100) < self.random_choice_prob

        if random_move:
            # TODO: going only to possible nodes,
            move = random.randint(0, self.output_size - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        try:
            final_move[move] = 1
        except IndexError:
            embed(header='indxerror')

        return final_move

    def save_model(self, path):
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)

        torch.save(self.model.state_dict(), f'{path}/{self.node}')
        # TODO: model.train() and model.eval()

    def load_model(self, map_path, foldername=None):

        if foldername is None:
            print('MODELS:')
            for file in os.listdir(map_path):
                print(''*7 + file)
            print('Write the file you filename you wanna load ')
            finished = False
            while not finished:
                folder_name_input = input('>> filename: ')
                try:
                    self.model.load_state_dict(
                        torch.load(f'{map_path}/{foldername}/{self.node}'))
                    finished = True
                except FileNotFoundError:
                    print('You need to choose a folder written above')

            self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)
        else:
            self.model.load_state_dict(
                torch.load(f'{map_path}/{foldername}/{self.node}'))

        return foldername

    def save_info(self):
        # TODO: maybe reset all informasjon before every run
        path2 = f'plans/set_{self.n_games//25}/n_games_{self.n_games}/'

        if not os.path.exists(path2):
            os.makedirs(path2)

        for dumper in self.dumpers:

            outfile = open(
                path2 + f'dumper_{dumper.get_num()}.txt', 'w')
            try:
                for route_key in dumper.completed_routes:
                    outfile.write('###' + route_key + '### ')
                    outfile.write('start: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['start_end_time'][0]) + '\n')

                    for i, key in enumerate(dumper.completed_routes[route_key]):
                        outfile.write(key + ': ')

                        for val in dumper.completed_routes[route_key][key]:
                            try:
                                outfile.write(f'{val : .5f}' + '   ')
                            except TypeError:
                                outfile.write(str(val) + '   ')

                        outfile.write('\n')

                        if (i+1) % len(dumper.completed_routes[route_key]) == 0:
                            outfile.write('\n')

            except AttributeError:
                embed(header='yeah')

            outfile.close()

        # Save a all plans shortage, for easier interpretation.
        outfile_all_plans_short = open(path2 + f'all_plans.txt', 'w')
        for dumper in self.dumpers:
            outfile_all_plans_short.write(
                f'##############{dumper}##############\n')
            for i, route_key in enumerate(dumper.completed_routes):
                if i != 0 and i != (len(dumper.completed_routes) - 1):
                    start_node = dumper.completed_routes[route_key]['actual_route'][0]
                    end_node = dumper.completed_routes[route_key]['actual_route'][-1]
                    mass_start = dumper.completed_routes[route_key]['mass'][0]
                    mass_after = dumper.completed_routes[route_key]['mass'][-1]
                    reward = dumper.completed_routes[route_key]['actual_reward'][0]

                    outfile_all_plans_short.write('start: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['start_end_time'][0]) + '\n')
                    outfile_all_plans_short.write('end: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['start_end_time'][1]) + '\n')
                    outfile_all_plans_short.write(
                        f'PATH: {start_node}, {end_node} - {reward > 0}\n')
                    outfile_all_plans_short.write('waiting: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['waiting_time'][0]) + '\n')
                    outfile_all_plans_short.write('driving: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['driving_time'][0]) + '\n')
                    outfile_all_plans_short.write('time_since_last_used_node: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['time_since_last_used_node'][0]) + '\n')
                    outfile_all_plans_short.write(
                        f'mass: [{mass_start:.0f}, {mass_after:.0f}] \n')
                    outfile_all_plans_short.write(
                        "reward: " + f"{reward : .3f}" + "\n\n")

            outfile_all_plans_short.write(
                '----------------------------------------------------------------\n')

        outfile_all_plans_short.close()

    def restart_round_memory(self):
        self.round_memory = deque(maxlen=MAX_MEMORY)

    def improve_memory(self):
        # TODO: improve

        unique = {}
        for idx, m in enumerate(self.memory):
            in_name = "".join(np.char.mod('%d', m[0]))

            if (not in_name in unique) or (unique[in_name][1] < m[2]):
                unique[in_name] = [idx, m[2]]

        indeces = [idx for idx, max in unique.values()]
        new_memory = [self.memory[idx] for idx in indeces]
        self.memory = new_memory

    def add_route_to_round_memory(self, sample):

        self.train_short_memory(*sample)
        self.round_memory.append(sample)

    def train_current_game(self):
        # TODO: maybe train a sample, more samples

        if len(self.round_memory) > 0:
            states, actions, rewards, next_states, dones = zip(
                *self.round_memory)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            self.train_short_memory(
                states, actions, rewards, next_states, dones)

            # TODO: if this is necessary

            self.memory += self.round_memory

            # self.improve_memory()
