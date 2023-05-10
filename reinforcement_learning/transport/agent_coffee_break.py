import os
from IPython import embed
import model
from collections import deque
import numpy as np
import random
import helper
import torch
import sys

"""
Create Nodes before dumpers
    - dumpers uses nodes information


"""

# TODO: next, start up a model

# TODO: distance = 0, no edge between the nodes (can't go to itself)


# TODO: waiting time, driving time, start/end time to nearest second.

# EXPLORATION_NUM = 10


MAX_MEMORY = 5000
BATCH_SIZE = 500
# MAX_MEMORY = 500
# BATCH_SIZE = 100
LR = 0.001


def set_seed(SEED=10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


set_seed()


class AgentCoffeeBreak:

    def __init__(self, nodes, edges, dumping_nodes, loading_nodes, parking_nodes, dumpers, exploration_num=100):
        self.n_games = 0
        self.exploration_counter = 0
        self.nodes = nodes
        self.dumpers = dumpers
        self.loading_nodes = loading_nodes
        self.parking_nodes = parking_nodes

        self.dumping_nodes = dumping_nodes
        self.exploration_num = exploration_num
        self.epsilon = 0    # randomness
        self.gamma = 0    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.best_memory = deque(maxlen=MAX_MEMORY)
        self.extended_memory = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 0  # %
        self.times = 1
        self.exploration_break = 5

        # Find the average and worst speed of dumpers
        self.dumper_avg_speed = 0
        self.dumper_worst_speed = sys.maxsize
        self.dumper_best_speed = 0
        for dumper in self.dumpers:
            self.dumper_avg_speed += dumper.get_speed()
            self.dumper_worst_speed = min(
                self.dumper_worst_speed, dumper.get_speed())
            self.dumper_best_speed = max(
                self.dumper_best_speed, dumper.get_speed())
        self.dumper_avg_speed = self.dumper_avg_speed/len(self.dumpers)

        self.initialize_network()

    def initialize_network(self):
        self.input_size = len(self.dumping_nodes) + \
            2  # + len(self.loading_nodes)
        self.input_size = 4  # mine, and shortest dumping and loading, reward loading
        self.output_size = 2

        # network = [self.input_size, 100, self.output_size]
        network = [self.input_size, 100, 50, 100, self.output_size]
        # network = [self.input_size, 70, 40, 20, 40, 70, self.output_size]
        # network = [self.input_size, 30, 10, 30, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt

    def create_time_converter_constant(self):
        self.longest_distance = 0
        self.shortest_distance = sys.maxsize
        for dumping_node in self.dumping_nodes + self.parking_nodes:
            for loading_node in self.loading_nodes:
                _dist = dumping_node.get_shortest_observed_distance_to_node(
                    loading_node)
                self.longest_distance = max(self.longest_distance, _dist)
                self.shortest_distance = min(self.shortest_distance, _dist)

        self.shortest_time = (self.shortest_distance /
                              self.dumper_best_speed) / 60
        self.longest_time = (self.longest_distance /
                             self.dumper_worst_speed) / 60  # minutes
        self.time_converter_constant = self.func_find_constant(
            self.longest_time)

    def func_find_constant(self, worst_time, value=9.9):
        return -(1/worst_time) * np.log(1 - value/10)

    # def time_converter_state(self, time):
    #     return 10 * (1 - np.exp(-self.time_converter_constant * time))

    def time_converter_state(self, time):
        return 10 * time/self.longest_time

    def get_state(self, dumper, curr_time, loading_node_plan, pred_reward=None):
        '''
        pred_reward, time_from_here,
        dumping_node -> shortest way * all
        onehot vector for loading node to
        best time from loading node -> dumping node -> correct l node
            -> this one should may be more accurate (find shortest path to corret l node, via dump)
        '''

        # get what should be sendt into the model
        # state = [pred_reward]
        state = []

        # for node in self.nodes:
        #     state.append(dumper.get_current_node() == node)
        curr_node = dumper.get_current_node()
        dumper_time_usage_to_end = (curr_node.get_shortest_observed_distance_to_node(
            loading_node_plan) / dumper.get_speed()) / 60  # min
        state.append(self.time_converter_state(dumper_time_usage_to_end))

        # embed(header='in_state_of_coffee_man')
        # TODOOOOOOO: USE LOADING_NODE FOR SOM
        # TODO: add my distance to it???

        # time for the shortest to dumping_node -> to go back to loading_node
        distance_from_dumping_nodes = []
        for dumping_node in self.dumping_nodes:

            distance_to_load = dumping_node.get_shortest_observed_distance_to_node(
                loading_node_plan)

            distance_from_dumping_nodes.append(distance_to_load)

            frac_done = 0

            if len(dumping_node.queue) == 0:
                state.append(10)  # add highest value
                continue

            nearest_dumper = dumping_node.queue[0]
            time_to_load_node = distance_to_load / nearest_dumper.get_speed()
            time_arrival = dumping_node.get_dumpers_incoming()[
                nearest_dumper][0]

            dumping_time = nearest_dumper.predict_dumping_time()
            driving_time = time_to_load_node
            if dumping_node.is_used():
                # Is used
                start_time = dumping_node.get_time_start_using()
                finish_time = dumping_node.get_time_finished_use()
                frac_done = (curr_time - start_time) / \
                    (finish_time - start_time)

                dumping_time = dumping_time * (1-frac_done)
            else:
                # /nearest_dumper.get_speed()
                driving_time += (time_arrival - curr_time)

            fastest_time = (driving_time + dumping_time) / 60  # min

            # Add predictive waiting time
            state.append(self.time_converter_state(fastest_time))

        # If just using the minimum time -> from the dumping nodes (less info, but all the necessary??)
        state = [state[0], min(state[1:])]

        minimum_time = sys.maxsize
        via_dumping_node = self.dumping_nodes[np.argmin(
            distance_from_dumping_nodes)]
        for loading_node in self.loading_nodes:
            # state.append(loading_node == loading_node_plan)

            # Some predefined queue
            if len(loading_node.queue) > 0:
                nearest_dumper = loading_node.queue[0]

            elif len(loading_node.get_dumpers_incoming()) > 0:
                # continue
                nearest_dumper = None
                dumper_time = sys.maxsize
                incoming_dict = loading_node.get_dumpers_incoming()
                for _dumper in incoming_dict:
                    if incoming_dict[_dumper][0] < dumper_time:
                        dumper_time = incoming_dict[_dumper][0]
                        nearest_dumper = _dumper

            else:
                # TODO: continue if there are no nearest dumper
                continue

            time_arrival = loading_node.get_dumpers_incoming()[
                nearest_dumper][0]

            driving_time = loading_node.get_shortest_observed_distance_to_node(
                via_dumping_node) / nearest_dumper.get_speed()

            driving_time += via_dumping_node.get_shortest_observed_distance_to_node(
                loading_node_plan) / nearest_dumper.get_speed()

            dumping_time = nearest_dumper.predict_dumping_time()
            loading_time = nearest_dumper.predict_loading_time(
                loading_node)
            if loading_node.is_used():
                # Is used
                start_time = loading_node.get_time_start_using()
                finish_time = loading_node.get_time_finished_use()
                frac_done = (curr_time - start_time) / \
                    (finish_time - start_time)

                loading_time = loading_time * (1-frac_done)
            else:
                # this should just be added, not divided by speed
                # / nearest_dumper.get_speed()
                driving_time += max((time_arrival - curr_time), 0)

            curr_fastest_time = driving_time + dumping_time + loading_time
            if curr_fastest_time < minimum_time:
                if curr_fastest_time < 0:
                    embed(header='why is this negative??')

                minimum_time = curr_fastest_time

        if minimum_time != sys.maxsize:
            minimum_time = minimum_time / 60  # min
            state.append(self.time_converter_state(minimum_time))
        else:
            state.append(10)

        state.append(pred_reward)

        # embed(header='check out loading_dudes')
        # for node in self.nodes:
        #     state.append(node == dumper.get_current_node())

        # distances_argsort = list(np.argsort(list_distances))

        # state.append(list_distances[idx] / 1000)    # in km

        # state.append(distances_argsort.index(idx) + 1)

        return np.array(state, dtype=float)  # maybe float?

    def set_exploration_num(self, num, times=1, exploration_break=5):
        self.memory = deque(maxlen=MAX_MEMORY)
        self.exploration_counter = 0
        self.exploration_num = num
        self.times = times
        self.exploration_break = exploration_break

    def increase_game_number(self):
        self.n_games += 1
        self.exploration_counter += 1

        if self.times > 1:
            if self.exploration_counter == self.exploration_num + self.exploration_break:
                self.times -= 1
                self.exploration_counter = 0

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
            mini_sample = list(self.memory)

        if len(self.best_memory) > BATCH_SIZE:
            best_mini_sample = random.sample(
                self.best_memory, BATCH_SIZE)  # list of tuples
        else:
            best_mini_sample = list(self.best_memory)

        # sample = mini_sample + best_mini_sample
        sample = mini_sample

        if len(sample) > 0:
            states, actions, rewards, next_states, dones = zip(*sample)

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

        self.epsilon = self.exploration_num - self.exploration_counter
        try:
            final_move = [0] * self.output_size
        except AttributeError:
            embed(header='attr')

        # If random_move or
        random_move = random.randint(0, self.exploration_num) < self.epsilon or \
            random.randint(0, 100) < self.random_choice_prob

        if random_move:
            move = random.randint(0, self.output_size - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

            move = torch.argmax(prediction).item()

        try:
            final_move[move] = 1
        except IndexError:
            embed(header='indx in agent_coffee')

        if random_move:
            return final_move, None
        else:
            return final_move, prediction

    def save_model(self, path):
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)

        torch.save(self.model.state_dict(),
                   f'{path}/coffee_break')

        # torch.save(self.model.state_dict(),
        #            f'{path}/plan_{end_notation}')
        # TODO: model.train() and model.eval()

    def save_coffee_break_info(self):

        path2 = f'plans/set_{self.n_games//25}/n_games_{self.n_games}/'
        outfile = open(path2 + f'coffee_info.txt', 'w')
        text = ['COFFEE BREAK', 'FOLLOW PLAN']
        for dumper in self.dumpers:
            outfile.write(f'##############{dumper}#############\n')
            idx = 0
            for key in dumper.coffee_dict:
                for state, action in dumper.coffee_dict[key]:
                    time = helper.get_time_formatted(
                        dumper.coffee_all_times[idx])

                    outfile.write(f'{time} - {text[action.index(1)]}\n')
                    try:
                        reward = dumper.coffee_rewards[idx]
                        outfile.write(f'        Reward: {reward: 8.3f}\n')
                    except IndexError:
                        outfile.write(f'        Reward: none \n')

                    outfile.write(
                        f'        State: {list(np.round(state, 1))}\n')
                    idx += 1

            outfile.write(
                '\n----------------------------------------------------------------\n')

        outfile.close()

    def load_model(self, map_path, foldername=None):

        if foldername is None:
            print('MODELS:')
            for file in os.listdir(map_path):
                item_path = os.path.join(map_path, file)
                if os.path.isdir(item_path):
                    print("### " + file)

            print('Write the file you filename you wanna load ')
            finished = False

            while not finished:
                foldername = input('>> filename: ')
                if foldername == 'NO':
                    return foldername

                try:
                    self.model.load_state_dict(
                        torch.load(f'{map_path}/{foldername}/coffee_break'))
                    finished = True
                except FileNotFoundError:
                    print('You need to choose a file written above')
        else:
            self.model.load_state_dict(torch.load(
                f'{map_path}/{foldername}/coffee_break'))

        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

        return foldername

    def restart_round_memory(self):
        self.round_memory = deque(maxlen=MAX_MEMORY)

    def add_route_to_round_memory(self, sample):

        if self.random_choice_prob != 100:
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
