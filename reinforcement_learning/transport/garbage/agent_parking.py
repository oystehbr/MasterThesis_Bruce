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


MAX_MEMORY = 5000
BATCH_SIZE = 500
# MAX_MEMORY = 500
# BATCH_SIZE = 100
LR = 0.001

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class AgentParking:
    def __init__(
        self,
        nodes,
        edges,
        dumping_nodes,
        loading_nodes,
        dumpers,
        parking_nodes,
        exploration_num=100,
    ):
        self.n_games = 0
        self.exploration_counter = 0
        self.nodes = nodes
        self.dumpers = dumpers
        self.loading_nodes = loading_nodes
        self.loading_node_count = len(loading_nodes)
        self.parking_nodes = parking_nodes

        self.dumping_nodes = dumping_nodes
        self.exploration_num = exploration_num
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.best_memory = deque(maxlen=MAX_MEMORY)
        self.extended_memory = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 0  # %
        self.prob_parking = 20
        self.times = 1
        self.exploration_break = 5

        # Find the average and worst speed of dumpers

        self.initialize_network()

    def initialize_network(self):
        # min(waiting), min_distance parking.(additional sum of all)
        # self.input_size = 1
        self.input_size = 1
        self.output_size = 2

        network = [self.input_size, 10, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt

    def func_find_constant(self, worst_time, value=9.9):
        return -(1 / worst_time) * np.log(1 - value / 10)

    # def time_converter_state(self, time):
    #     return 10 * (1 - np.exp(-self.time_converter_constant * time))

    def time_converter_state(self, time):
        return 10 * time / self.longest_time

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
            if (
                self.exploration_counter
                == self.exploration_num + self.exploration_break
            ):
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
        # return
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = list(self.memory)

        if len(self.best_memory) > BATCH_SIZE:
            best_mini_sample = random.sample(
                self.best_memory, BATCH_SIZE
            )  # list of tuples
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

            self.trainer.train_step(states, actions, rewards, next_states, dones)
            # for state, action, reward, nexrt_state, done in mini_sample:
            #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_state(
        self,
        dumper,
        load_action,
        loading_state,
        amount_of_dumper_in_simulation,
        still_active_load_nodes,
    ):
        """
        pred_reward, time_from_here,
        dumping_node -> shortest way * all
        onehot vector for loading node to
        best time from loading node -> dumping node -> correct l node
            -> this one should may be more accurate (find shortest path to corret l node, via dump)
        """

        # get what should be sendt into the model
        state = [amount_of_dumper_in_simulation]
        # state.append(np.sum(still_active_load_nodes))  # active loading nodes
        # waiting_times = loading_state[:self.loading_node_count]
        # state.append(min(waiting_times))

        # idx_distance = np.argmin(waiting_times) + self.loading_node_count
        # distance_to_min_waiting_time = loading_state[idx_distance]
        # state.append(distance_to_min_waiting_time)

        # state.append(sum(waiting_times))

        # Adding the distance to the parking node
        curr_node = dumper.get_current_node()
        minimum_distance = sys.maxsize
        the_parking_node = None
        for parking_node in self.parking_nodes:
            distance_to_parking = curr_node.get_shortest_observed_distance_to_node(
                parking_node
            )

            if distance_to_parking < minimum_distance:
                minimum_distance = distance_to_parking
                the_parking_node = parking_node

        # TODO: maybe unnecessary to provide distance to parking node..
        # state.append(minimum_distance/1000)  # in km
        # state += list(still_active_load_nodes)

        return np.array(state, dtype=float), the_parking_node

    def get_action(self, state, dumper):
        # random moves: tradeoff exploration / exploitation

        self.epsilon = self.exploration_num - self.exploration_counter
        try:
            final_move = [0] * self.output_size
        except AttributeError:
            embed(header="attr")

        # If random_move or
        random_move = (
            random.randint(0, self.exploration_num) < self.epsilon
            or random.randint(0, 100) < self.random_choice_prob
        )

        if random_move:
            if random.randint(0, 100) < self.prob_parking:
                move = 0
                # TODO: check at this er riktig
            else:
                move = 1

            # move = random.randint(0, self.output_size - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)

            move = torch.argmax(prediction).item()

        try:
            final_move[move] = 1
        except IndexError:
            embed(header="indx in agent_parking")

        if random_move:
            return final_move, None
        else:
            return final_move, prediction

    def save_model(self, path):
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)

        torch.save(self.model.state_dict(), f"{path}/parking")

        # torch.save(self.model.state_dict(),
        #            f'{path}/plan_{end_notation}')
        # TODO: model.train() and model.eval()

    def load_model(self, map_path, foldername=None):
        if foldername is None:
            print("MODELS:")
            for file in os.listdir(map_path):
                item_path = os.path.join(map_path, file)
                if os.path.isdir(item_path):
                    print("### " + file)

            print("Write the file you filename you wanna load ")
            finished = False

            while not finished:
                foldername = input(">> filename: ")
                if foldername == "NO":
                    return foldername

                try:
                    self.model.load_state_dict(
                        torch.load(f"{map_path}/{foldername}/parking")
                    )
                    finished = True
                except FileNotFoundError:
                    print("You need to choose a file written above")
        else:
            self.model.load_state_dict(torch.load(f"{map_path}/{foldername}/parking"))

        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

        return foldername

    def restart_round_memory(self):
        self.round_memory = deque(maxlen=MAX_MEMORY)

    def restart_memory(self):
        self.memory = deque(maxlen=MAX_MEMORY)

    def add_route_to_round_memory(self, sample):
        if self.random_choice_prob != 100:
            self.train_short_memory(*sample)
            self.round_memory.append(sample)

    def train_current_game(self):
        # TODO: maybe train a sample, more samples
        if len(self.round_memory) > 0:
            states, actions, rewards, next_states, dones = zip(*self.round_memory)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            self.train_short_memory(states, actions, rewards, next_states, dones)

            # TODO: if this is necessary
            self.memory += self.round_memory
