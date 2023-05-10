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
BATCH_SIZE = 50
LR = 0.001

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AgentPlan:
    agents = []

    def __init__(self, nodes, edges, nodes_loading_or_dumping, dumping_nodes, loading_nodes, dumpers, exploration_num=100):
        AgentPlan.agents.append(self)
        self.n_games = 0
        self.nodes = nodes
        self.dumpers = dumpers
        self.dumping_nodes = dumping_nodes
        self.loading_nodes = loading_nodes

        self.nodes_loading_or_dumping = nodes_loading_or_dumping
        self.exploration_num = exploration_num
        self.epsilon = 0    # randomness
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.extended_memory = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 0  # %

        # input: one dumper, and all nodes
        # TODO: coordinates
        # + len(self.nodes)
        self.input_size = len(self.nodes_loading_or_dumping)*2
        self.output_size = len(self.loading_nodes) * \
            len(self.dumping_nodes)
        network = [self.input_size, 200, 100, 200, 100, self.output_size]
        # network = [self.input_size, 70, 40, 20, 40, 70, self.output_size]
        # network = [self.input_size, 30, 10, 30, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, dumper, curr_time):

        # get what should be sendt into the model
        state = []
        curr_x = int(dumper.get_current_node().get_coordinates()['x'])
        curr_y = int(dumper.get_current_node().get_coordinates()['y'])

        # for node in self.nodes:
        #     state.append(dumper.get_current_node() == node)
        list_distances = []
        for node in self.nodes_loading_or_dumping:
            next_x = int(node.get_coordinates()['x'])
            next_y = int(node.get_coordinates()['y'])

            distance = np.sqrt((curr_x - next_x)**2 + (curr_y - next_y)**2)
            list_distances.append(distance)
            # state.append(distance)
            # TODO: maybe substract half if node is used
            val = len(node.get_dumpers_incoming())
            if val == 0:
                val = -1
            else:
                if node.is_used():
                    start_time = node.get_time_start_using()
                    finish_time = node.get_time_finished_use()
                    frac_finished = (curr_time - start_time) / \
                        (finish_time - start_time)
                    val -= frac_finished

            state.append(val)

        # for node in self.nodes:
        #     state.append(node == dumper.get_current_node())

        # distances_argsort = list(np.argsort(list_distances))
        for idx, node in enumerate(self.nodes_loading_or_dumping):
            if list_distances[idx] == 0:
                list_distances[idx] = 1

            state.append(list_distances[idx] / 1000)

            # state.append(distances_argsort.index(idx) + 1)

        #     # TODO: need info of if job is actual possible.

        return np.array(state, dtype=float)  # maybe float?

    def set_exploration_num(self, num):
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

        if len(mini_sample) > 0:
            states, actions, rewards, next_states, dones = zip(*mini_sample)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            self.trainer.train_step(
                states, actions, rewards, next_states, dones)
            # for state, action, reward, nexrt_state, done in mini_sample:
            #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, dumper):
        # random moves: tradeoff exploration / exploitation

        self.epsilon = self.exploration_num - self.n_games
        final_move = [0] * self.output_size

        # If random_move or
        random_move = random.randint(0, self.exploration_num + 10) < self.epsilon or \
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
            embed(header='indx in agentplan')

        return final_move

    def save_model(self, foldername):
        path = f'models_new/{foldername}'
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)

        torch.save(self.model.state_dict(),
                   'models_new/' + foldername + '/plan')
        # TODO: model.train() and model.eval()

    def load_model(self, foldername=None):

        if foldername is None:
            print('MODELS:')
            for file in os.listdir('models_new'):
                print(''*7 + file)

            print('Write the file you filename you wanna load ')
            finished = False
            while not finished:
                foldername = input('>> filename: ')
                try:
                    self.model.load_state_dict(
                        torch.load('models_new/' + foldername + '/plan'))
                    finished = True
                except FileNotFoundError:
                    print('You need to choose a file written above')
        else:
            self.model.load_state_dict(torch.load(
                'models_new/' + foldername + '/plan'))

        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

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
                        if key in ['full_route', 'state_info', 'full_reward']:
                            continue

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

        outfile_all_plans_nodes = open(path2 + f'all_plans_nodes.txt', 'w')
        for node in self.nodes_loading_or_dumping:
            time_scheduling = node.get_time_scheduling()
            outfile_all_plans_nodes.write(f'{node} \n')
            for key in time_scheduling:
                outfile_all_plans_nodes.write(
                    f'{key} - {time_scheduling[key]} \n')

            outfile_all_plans_nodes.write(
                '################################################ \n')

        outfile_all_plans_nodes.close()

        # Save a all plans shortage, for easier interpretation.
        outfile_all_plans_short = open(path2 + f'all_plans.txt', 'w')
        for dumper in self.dumpers:
            outfile_all_plans_short.write(
                f'##############{dumper}##############\n')
            for i, route_key in enumerate(dumper.completed_routes):
                if len(dumper.completed_routes[route_key]) >= 13:
                    ac_route = dumper.completed_routes[route_key]['actual_route']
                    start_node = ac_route[0]
                    end_node = ac_route[-1]
                    route_is_back_to_back = len(set(ac_route)) == len(ac_route)

                    mass_start = dumper.completed_routes[route_key]['mass'][0]
                    mass_after = dumper.completed_routes[route_key]['mass'][-1]
                    state_plan = dumper.completed_routes[route_key]['state_plan'][0].get_state(
                    )
                    state_plan = [f'{i:.1f}' for i in list(state_plan)]
                    state_plan = '[' + ", ".join(state_plan) + ']'

                    plan_reward = 0
                    if len(dumper.completed_routes[route_key]) == 14:
                        plan_reward = dumper.completed_routes[route_key]['plan_reward'][0]

                    outfile_all_plans_short.write('start: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['start_end_time'][0]) + '\n')
                    outfile_all_plans_short.write('end: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['start_end_time'][1]) + '\n')
                    outfile_all_plans_short.write(
                        f'PATH: {start_node}, {end_node} - {route_is_back_to_back}\n')
                    outfile_all_plans_short.write('waiting: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['waiting_time'][0]) + '\n')
                    outfile_all_plans_short.write('driving: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['driving_time'][0]) + '\n')
                    outfile_all_plans_short.write('time_since_last_used_node: ' + helper.get_time_formatted(
                        dumper.completed_routes[route_key]['time_since_last_used_node'][0]) + '\n')
                    outfile_all_plans_short.write(
                        f'state_plan: ' + state_plan + '\n')
                    outfile_all_plans_short.write(
                        f'mass: [{mass_start:.0f}, {mass_after:.0f}] \n')
                    outfile_all_plans_short.write(
                        "plan_reward: " + f"{plan_reward : .3f}" + "\n\n")

            outfile_all_plans_short.write(
                '----------------------------------------------------------------\n')

        outfile_all_plans_short.close()

    def restart_round_memory(self):
        self.round_memory = deque(maxlen=MAX_MEMORY)

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
