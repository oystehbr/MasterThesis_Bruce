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


class AgentPlan:
    agents = []

    def __init__(self, nodes, edges, loading_or_dumping_nodes, dumping_nodes, loading_nodes, parking_nodes, dumpers, exploration_num=100):
        AgentPlan.agents.append(self)
        self.n_games = 0
        self.exploration_counter = 0
        self.nodes = nodes
        self.dumpers = dumpers
        self.dumping_nodes = dumping_nodes
        self.loading_nodes = loading_nodes
        self.parking_nodes = parking_nodes

        self.loading_or_dumping_nodes = loading_or_dumping_nodes
        self.exploration_num = exploration_num
        self.epsilon = 0    # randomness
        self.gamma = 0    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.best_memory = deque(maxlen=MAX_MEMORY)
        self.extended_memory = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 0  # %
        self.times = 1
        self.exploration_break = 5

        self.initialize_network()

    # def initialize_network(self):
    #     self.input_size = len(self.loading_or_dumping_nodes)*2
    #     self.output_size = len(self.loading_nodes) * \
    #         len(self.dumping_nodes)

    #     network = [self.input_size, 200, 100, 200, 100, self.output_size]
    #     # network = [self.input_size, 70, 40, 20, 40, 70, self.output_size]
    #     # network = [self.input_size, 30, 10, 30, self.output_size]

    #     self.model = model.Linear_QNet(network)
    #     self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt
        self.game = route_opt.game

    def get_state(self, dumper, curr_time, amount_of_dumpers_in_simulation=None):

        # get what should be sendt into the model
        state = []

        # for node in self.nodes:
        #     state.append(dumper.get_current_node() == node)
        curr_node = dumper.get_current_node()
        list_distances = []

        if self.class_id == 'agent_dumping':
            # TODO: if more mass types, add here
            running_through_nodes = self.dumping_nodes

            # adding information about how much dumping mass there are left
            rest_capacity = []
            max_capacity = []
            mass_type = dumper.get_current_node().get_mass_type()
            rest_capacity = list(
                self.game.rest_dumping_mass[mass_type].values())

            num_fully_dumping_behind = list((
                np.array(rest_capacity) - min(rest_capacity))/dumper.get_mass_capacity())

            dumper.add_knowledge_mass_differences(
                num_fully_dumping_behind)

            state += num_fully_dumping_behind

        elif self.class_id == 'agent_loading':
            the_parking_node = None
            running_through_nodes = self.loading_nodes

            # LOADING PARKING
            minimum_parking_distance = sys.maxsize
            for parking_node in self.parking_nodes:
                distance_to_parking = curr_node.get_shortest_observed_distance_to_node(
                    parking_node)

                if distance_to_parking < minimum_parking_distance:
                    minimum_parking_distance = distance_to_parking
                    the_parking_node = parking_node

            state.append((minimum_parking_distance /
                          dumper.get_speed())/60)  # min
            # state.append(amount_of_dumpers_in_simulation)
            # state.append(minimum_parking_distance / 1000) #km
        else:
            embed(header='problems in agentplan2, getstate')

        for node in running_through_nodes:
            distance = curr_node.get_shortest_observed_distance_to_node(node)
            list_distances.append(distance)

            incoming_dumpers = len(node.get_dumpers_incoming())
            # TODO: is this still in use??
            # if incoming_dumpers == 0:
            #     incoming_dumpers = -1    # TODO: check this out
            # else:
            #     if node.is_used():
            #         start_time = node.get_time_start_using()
            #         finish_time = node.get_time_finished_use()
            #         frac_finished = (curr_time - start_time) / \
            #             (finish_time - start_time)
            #         incoming_dumpers -= frac_finished

            # How long is the predicted waiting time
            curr_available_time = node.all_on_the_way_finished_used
            dumpers_incoming = node.get_dumpers_incoming()
            dumpers_incoming_on_the_way = dumpers_incoming.copy()

            for _dumper in node.queue:
                dumpers_incoming_on_the_way.pop(_dumper)

            for _dumper in dumpers_incoming_on_the_way:
                if node.class_id == 'loading':
                    predicted_use_time = _dumper.predict_loading_time(
                        node) + 1
                else:
                    predicted_use_time = _dumper.predict_dumping_time()

                if curr_available_time < dumpers_incoming[_dumper][0]:
                    curr_available_time = dumpers_incoming[_dumper][0] + \
                        predicted_use_time
                else:
                    curr_available_time += predicted_use_time

            dumper_reached_destination = curr_time + distance/dumper.speed

            if node.class_id == 'loading':
                predicted_use_time = dumper.predict_loading_time(node) + 1
            else:
                predicted_use_time = dumper.predict_dumping_time() + 1

            # Waiting time in # loads/ dumps you need to wait for
            # waiting_time = (curr_available_time -
            #                 dumper_reached_destination)/predicted_use_time  # TODO: maybe non-negative values

            # TODO: in minuites instead
            waiting_time = (curr_available_time -
                            dumper_reached_destination)/60  # TODO: maybe non-negative values

            # If loader not started, and no-one on it's way.
            if not node.active and incoming_dumpers == 0:
                state.append(0)
            else:
                # Add predictive waiting time
                state.append(waiting_time)

        # for node in self.nodes:
        #     state.append(node == dumper.get_current_node())

        # distances_argsort = list(np.argsort(list_distances))
        for idx, node in enumerate(running_through_nodes):
            # if list_distances[idx] == 0:
            #     state.append(-10)
            # else:
            state.append((list_distances[idx] / dumper.get_speed())/60)
            # state.append(list_distances[idx] / 1000)    # in km
            # state.append(distances_argsort.index(idx) + 1)

        #     # TODO: need info of if job is actual possible.

        # TODO: not in use
        # lol = [1.0, 0.0, 0.0, 0.0]
        # if self.class_id == 'agent_loading':
        #     count = 0
        #     for num1, num2 in zip(lol, state):
        #         if abs(num1 - num2) < 0.1:
        #             count += 1

        #     if count == len(lol):
        #         embed(header='what in hell')
        if self.class_id == 'agent_loading':
            return np.array(state, dtype=float), the_parking_node
        else:
            return np.array(state, dtype=float)

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

    def get_action(self, state, dumper, more_mass_vec=None):
        # random moves: tradeoff exploration / exploitation

        self.epsilon = self.exploration_num - self.exploration_counter
        try:
            final_move = [0] * self.output_size
        except AttributeError:
            embed(header='attr')

        # If random_move or
        random_move = random.randint(0, self.exploration_num) < self.epsilon or \
            random.randint(0, 100) < self.random_choice_prob

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        if random_move:
            # TODO: going only to possible nodes,
            if more_mass_vec is None:
                move = random.randint(0, self.output_size - 1)
            else:
                move = random.choice(np.where(more_mass_vec == True)[0])

        else:
            if more_mass_vec is None:
                move = torch.argmax(prediction).item()
            else:
                numpy_pred = prediction.cpu().detach().numpy()
                except_index = np.where(more_mass_vec == False)[0]
                m = np.zeros(numpy_pred.size, dtype=bool)
                m[except_index] = True
                a = np.ma.array(numpy_pred, mask=m)
                move = np.argmax(a)

        # if not more_mass_vec is None:
        #     if np.sum(more_mass_vec) != len(more_mass_vec):
        #         embed(header='check this out')

        try:
            final_move[move] = 1
        except IndexError:
            embed(header='indx in agentplan')

        return final_move, prediction

    def save_model(self, path):
        isExist = os.path.exists(path)

        if not isExist:
            os.makedirs(path)

        end_notation = self.class_id
        if self.class_id == 'agent_dumping':
            end_notation += self.mass_type

        torch.save(self.model.state_dict(),
                   f'{path}/plan_{end_notation}')
        # TODO: model.train() and model.eval()

    def load_model(self, map_path, foldername=None):

        end_notation = self.class_id
        if self.class_id == 'agent_dumping':
            end_notation += self.mass_type

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
                        torch.load(f'{map_path}/{foldername}/plan_' + end_notation))
                    finished = True
                except FileNotFoundError:
                    print('You need to choose a file written above')
        else:
            self.model.load_state_dict(torch.load(
                f'{map_path}/{foldername}/plan_' + end_notation))

        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

        return foldername

    def save_info(self):
        # TODO: maybe reset all informasjon before every run
        path2 = f'plans/set_{self.n_games//25}/n_games_{self.n_games}/'

        if not os.path.exists(path2):
            os.makedirs(path2)

        for the_file in os.listdir(path2):
            file_path = os.path.join(path2, the_file)

            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)

            except Exception as e:
                print(f'{e}')

        for dumper in self.dumpers:

            outfile = open(
                path2 + f'dumper_{dumper.get_num()}.txt', 'w')
            try:
                dumper_completed_routes = dumper.completed_routes
                for route_key in dumper_completed_routes:
                    outfile.write('###' + route_key + '### ')
                    outfile.write('start: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['start_end_time'][0]) + '\n')

                    for i, key in enumerate(dumper_completed_routes[route_key]):
                        outfile.write(key + ': ')
                        if key in ['full_route', 'state_info', 'full_reward']:
                            continue

                        for val in dumper_completed_routes[route_key][key]:
                            try:
                                outfile.write(f'{val : .5f}' + '   ')
                            except TypeError:
                                outfile.write(str(val) + '   ')
                            except ValueError:
                                outfile.write(str(val) + '   ')

                        outfile.write('\n')

                        if (i+1) % len(dumper_completed_routes[route_key]) == 0:
                            outfile.write('\n')

            except AttributeError:
                embed(header='yeah')

            outfile.close()

        outfile_all_plans_nodes = open(path2 + f'all_plans_nodes.txt', 'w')
        for node in self.loading_or_dumping_nodes:
            time_scheduling = node.get_time_scheduling()
            outfile_all_plans_nodes.write(f'{node} \n')

            first = True
            for idx, key in enumerate(time_scheduling):
                current_time = key
                if idx % 2 == 0 and not first:
                    curr_time_sec = helper.get_second_from_time_formatted(
                        current_time)
                    last_time_sec = helper.get_second_from_time_formatted(
                        last_time)
                    if curr_time_sec - last_time_sec > 2:
                        outfile_all_plans_nodes.write('-- \n')

                first = False

                outfile_all_plans_nodes.write(
                    f'{current_time} - {time_scheduling[key]} \n')
                last_time = key

            outfile_all_plans_nodes.write(
                '################################################ \n')

        outfile_all_plans_nodes.close()

        # Save a all plans shortage, for easier interpretation.
        outfile_all_plans_short = open(path2 + f'all_plans.txt', 'w')
        for dumper in self.dumpers:
            outfile_all_plans_short.write(
                f'##############{dumper}##############\n')
            dumper_completed_routes = dumper.completed_routes
            for i, route_key in enumerate(dumper_completed_routes):
                # if the dumper went on the parking slot as the last plan
                if i == len(dumper_completed_routes) - 1:
                    if dumper.is_parked():
                        ac_route = dumper_completed_routes[route_key]['actual_route']
                        start_node = ac_route[0]
                        end_node = dumper.get_on_my_way_to()
                        state_plan = dumper_completed_routes[route_key]['state_plan'][0].get_state(
                        )
                        state_plan = [f'{i:.1f}' for i in list(state_plan)]
                        state_plan = '[' + ", ".join(state_plan) + ']'

                        outfile_all_plans_short.write('start: ' + helper.get_time_formatted(
                            dumper_completed_routes[route_key]['start_end_time'][0]) + '\n')
                        try:
                            outfile_all_plans_short.write('end: ' + helper.get_time_formatted(
                                dumper_completed_routes[route_key]['start_end_time'][1]) + '\n')
                        except IndexError:
                            # If not able to reach the parking slot before game finishes
                            outfile_all_plans_short.write('end: NaN \n')

                        outfile_all_plans_short.write(
                            f'PLAN: {start_node}, {end_node}\n')
                        node_strings = [str(node) for node in ac_route]
                        outfile_all_plans_short.write(
                            f'PATH: {", ".join(node_strings)}\n')
                        outfile_all_plans_short.write(
                            f'state_plan: ' + state_plan + '\n')
                        # LOADING PARKING
                        try:
                            plan_reward = dumper_completed_routes[route_key]['plan_reward'][0]
                            outfile_all_plans_short.write(
                                "plan_reward: " + f"{plan_reward : .3f}" + "\n\n")
                        except:
                            nothing = 'yeah'

                        outfile_all_plans_short.write(
                            f'################ PARKED ################# \n\n')
                        continue

                if len(dumper_completed_routes[route_key]) >= 13:
                    ac_route = dumper_completed_routes[route_key]['actual_route']
                    start_node = ac_route[0]
                    end_node = ac_route[-1]
                    route_is_back_to_back = len(set(ac_route)) == len(ac_route)

                    mass_start = dumper_completed_routes[route_key]['mass'][0]
                    mass_after = dumper_completed_routes[route_key]['mass'][-1]
                    mass_type = dumper_completed_routes[route_key]['mass_type']
                    state_plan = dumper_completed_routes[route_key]['state_plan'][0].get_state(
                    )
                    state_plan = [f'{i:.1f}' for i in list(state_plan)]
                    state_plan = '[' + ", ".join(state_plan) + ']'

                    plan_reward = 0
                    if len(dumper_completed_routes[route_key]) >= 14:
                        plan_reward = dumper_completed_routes[route_key]['plan_reward'][0]

                    outfile_all_plans_short.write('start: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['start_end_time'][0]) + '\n')
                    outfile_all_plans_short.write('end: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['start_end_time'][1]) + '\n')
                    outfile_all_plans_short.write(
                        f'PLAN: {start_node}, {end_node}\n')

                    node_strings = [str(node) for node in ac_route]
                    outfile_all_plans_short.write(
                        f'PATH: {", ".join(node_strings)}\n')

                    outfile_all_plans_short.write('waiting: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['waiting_time'][0]) + '\n')
                    outfile_all_plans_short.write('driving: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['driving_time'][0]) + '\n')
                    outfile_all_plans_short.write('time_since_last_used_node: ' + helper.get_time_formatted(
                        dumper_completed_routes[route_key]['time_since_last_used_node'][0]) + '\n')
                    outfile_all_plans_short.write(
                        f'state_plan: ' + state_plan + '\n')
                    outfile_all_plans_short.write(
                        f'mass: [{mass_start:.0f}, {mass_after:.0f}], type: {mass_type} \n')
                    outfile_all_plans_short.write(
                        "plan_reward: " + f"{plan_reward : .3f}" + "\n\n")

            outfile_all_plans_short.write(
                '----------------------------------------------------------------\n')

        outfile_all_plans_short.close()

        # TODO: if short_simple should be provided
        if True:
            # TODO: alternatively
            outfile_all_plans_short_simple = open(
                path2 + f'all_plans_simple.txt', 'w')
            for dumper in self.dumpers:
                outfile_all_plans_short_simple.write(
                    f'##############{dumper}##############\n')
                dumper_completed_routes = dumper.completed_routes
                for i, route_key in enumerate(dumper_completed_routes):
                    # if the dumper went on the parking slot as the last plan
                    if i == len(dumper_completed_routes) - 1:
                        if dumper.is_parked():
                            ac_route = dumper_completed_routes[route_key]['actual_route']
                            end_plan = dumper.get_on_my_way_to()
                            start_time = helper.get_time_formatted(
                                dumper_completed_routes[route_key]['start_end_time'][0])
                            try:
                                end_time = helper.get_time_formatted(
                                    dumper_completed_routes[route_key]['start_end_time'][1])
                            except IndexError:
                                # If not able to reach the parking slot before game finishes
                                end_time = 'NaN'

                            str_ = f'Time: [' + start_time + ', ' + end_time + \
                                ']' + \
                                f', PLAN: {ac_route[0]}, {end_plan}\n'
                            outfile_all_plans_short_simple.write(str_)
                            outfile_all_plans_short_simple.write(
                                f'################ PARKED ################# \n\n')
                            continue

                    if len(dumper_completed_routes[route_key]) >= 13:
                        ac_route = dumper_completed_routes[route_key]['actual_route']
                        start_time = helper.get_time_formatted(
                            dumper_completed_routes[route_key]['start_end_time'][0])
                        try:
                            end_time = helper.get_time_formatted(
                                dumper_completed_routes[route_key]['start_end_time'][1])
                        except IndexError:
                            # If not able to reach the parking slot before game finishes
                            end_time = 'NaN'

                        node_strings = [str(node) for node in ac_route]
                        str_ = f'Time: [' + start_time + ', ' + end_time + \
                            ']' + f', PLAN: {ac_route[0]}, {ac_route[-1]}\n'
                        outfile_all_plans_short_simple.write(str_)

                outfile_all_plans_short_simple.write(
                    '--------------------------------------------\n')

            outfile_all_plans_short_simple.close()

    def restart_round_memory(self):
        self.round_memory = deque(maxlen=MAX_MEMORY)

    def add_route_to_round_memory(self, sample):

        if self.random_choice_prob != 100:
            self.train_short_memory(*sample)
            self.round_memory.append(sample)

    def train_current_game(self, curr_score, max_score):
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
            if curr_score > max_score:
                self.best_memory = deque(maxlen=MAX_MEMORY)
                self.best_memory += self.round_memory
            else:
                if curr_score >= max_score - 40:
                    self.best_memory += self.round_memory


class AgentPlanLoading(AgentPlan):
    class_id = 'agent_loading'

    def __init__(self, nodes, edges, loading_or_dumping_nodes, dumping_nodes, loading_nodes, dumpers, exploration_num=100):
        super().__init__(nodes, edges, loading_or_dumping_nodes,
                         dumping_nodes, loading_nodes, dumpers, exploration_num)

    def initialize_network(self):
        # LOADING PARKING
        self.input_size = len(self.loading_nodes)*2 + 1
        self.output_size = len(self.loading_nodes) + 1

        network = [self.input_size, 100, 50, 100, self.output_size]  # small
        # network = [self.input_size, 70, 40, 20, 40, 70, self.output_size]
        # network = [self.input_size, 100, self.output_size]
        # network = [self.input_size, 100, self.output_size]
        # network = [self.input_size, 200, 100, 200, 100, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, dumper, curr_time, amount_of_dumpers_in_simulation=None):

        # get what should be sendt into the model
        state = []

        # for node in self.nodes:
        #     state.append(dumper.get_current_node() == node)
        curr_node = dumper.get_current_node()
        list_distances = []

        the_parking_node = None
        running_through_nodes = self.loading_nodes

        # LOADING PARKING
        minimum_parking_distance = sys.maxsize
        for parking_node in self.parking_nodes:
            distance_to_parking = curr_node.get_shortest_observed_distance_to_node(
                parking_node)

            if distance_to_parking < minimum_parking_distance:
                minimum_parking_distance = distance_to_parking
                the_parking_node = parking_node

        # TODO: should it be -10
        state.append((minimum_parking_distance /
                      dumper.get_speed())/60)  # min
        # state.append(amount_of_dumpers_in_simulation)
        # state.append(minimum_parking_distance / 1000) #km

        for node in running_through_nodes:
            distance = curr_node.get_shortest_observed_distance_to_node(node)
            list_distances.append(distance)

            incoming_dumpers = len(node.get_dumpers_incoming())
            # TODO: is this still in use??
            # if incoming_dumpers == 0:
            #     incoming_dumpers = -1    # TODO: check this out
            # else:
            #     if node.is_used():
            #         start_time = node.get_time_start_using()
            #         finish_time = node.get_time_finished_use()
            #         frac_finished = (curr_time - start_time) / \
            #             (finish_time - start_time)
            #         incoming_dumpers -= frac_finished

            # How long is the predicted waiting time
            curr_available_time = node.all_on_the_way_finished_used
            dumpers_incoming = node.get_dumpers_incoming()
            dumpers_incoming_on_the_way = dumpers_incoming.copy()

            for _dumper in node.queue:
                dumpers_incoming_on_the_way.pop(_dumper)

            for _dumper in dumpers_incoming_on_the_way:
                if node.class_id == 'loading':
                    predicted_use_time = _dumper.predict_loading_time(
                        node) + 1
                else:
                    predicted_use_time = _dumper.predict_dumping_time()

                if curr_available_time < dumpers_incoming[_dumper][0]:
                    curr_available_time = dumpers_incoming[_dumper][0] + \
                        predicted_use_time
                else:
                    curr_available_time += predicted_use_time

            dumper_reached_destination = curr_time + distance/dumper.speed

            if node.class_id == 'loading':
                predicted_use_time = dumper.predict_loading_time(node) + 1
            else:
                predicted_use_time = dumper.predict_dumping_time() + 1

            # Waiting time in # loads/ dumps you need to wait for
            # waiting_time = (curr_available_time -
            #                 dumper_reached_destination)/predicted_use_time  # TODO: maybe non-negative values

            # TODO: in minuites instead
            waiting_time = (curr_available_time -
                            dumper_reached_destination)/60  # TODO: maybe non-negative values

            # If loader not started, and no-one on it's way.
            if not node.active and incoming_dumpers == 0:
                state.append(0)
            else:
                # Add predictive waiting time
                state.append(waiting_time)

        # for node in self.nodes:
        #     state.append(node == dumper.get_current_node())

        # distances_argsort = list(np.argsort(list_distances))
        for idx, node in enumerate(running_through_nodes):
            # if list_distances[idx] == 0:
            #     state.append(-10)
            # else:
            state.append((list_distances[idx] / dumper.get_speed())/60)
            # state.append(list_distances[idx] / 1000)    # in km
            # state.append(distances_argsort.index(idx) + 1)

        return np.array(state, dtype=float), the_parking_node


class AgentPlanDumping(AgentPlan):
    # TODO: maybe: change all dumping/loading_nodes -> possible nodes to go through -> more general.
    class_id = 'agent_dumping'

    def __init__(self, nodes, edges, loading_or_dumping_nodes, dumping_nodes, loading_nodes, dumpers, mass_type, exploration_num=100):
        self.mass_type = mass_type

        super().__init__(nodes, edges, loading_or_dumping_nodes,
                         dumping_nodes, loading_nodes, dumpers, exploration_num)

    def initialize_network(self):
        # self.input_size = len(self.dumping_nodes)*2
        self.input_size = len(self.dumping_nodes)*2
        self.output_size = len(self.dumping_nodes)

        network = [self.input_size, 100, 50, 100, self.output_size]
        # network = [self.input_size, 20, 10, 20, self.output_size]  # small
        # network = [self.input_size, 70, 40, 20, 40, 70, self.output_size]
        # network = [self.input_size, 30, 10, 30, self.output_size]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, dumper, curr_time, amount_of_dumpers_in_simulation=None):

        # get what should be sendt into the model
        state = []

        # for node in self.nodes:
        #     state.append(dumper.get_current_node() == node)
        curr_node = dumper.get_current_node()
        list_distances = []

        # TODO: if more mass types, add khere
        running_through_nodes = self.dumping_nodes

        # adding information about how much dumping mass there are left
        rest_capacity = []
        max_capacity = []
        mass_type = dumper.get_current_node().get_mass_type()
        rest_capacity = list(
            self.game.rest_dumping_mass[mass_type].values())

        # num_fully_dumping_behind = list((
        #     np.array(rest_capacity) - min(rest_capacity))/dumper.get_mass_capacity())

        for dumping_node in self.dumping_nodes:
            max_capacity.append(dumping_node.get_max_capacity())

        percent_completed = (np.array(max_capacity) -
                             np.array(rest_capacity))/(np.array(max_capacity))
        # completed_of_10 = percent_completed * 10

        # more mass runs
        base_ = max_capacity[list(percent_completed).index(
            min(percent_completed))]
        more_mass_runs = (percent_completed - min(percent_completed)) * \
            base_ / dumper.get_mass_capacity()
        dumper.add_knowledge_mass_differences(more_mass_runs)

        # np.clip(more_mass_run, 0, TOPCAP)

        state += list(more_mass_runs)

        for node in running_through_nodes:
            distance = curr_node.get_shortest_observed_distance_to_node(node)
            list_distances.append(distance)

            incoming_dumpers = len(node.get_dumpers_incoming())
            # TODO: is this still in use??
            # if incoming_dumpers == 0:
            #     incoming_dumpers = -1    # TODO: check this out
            # else:
            #     if node.is_used():
            #         start_time = node.get_time_start_using()
            #         finish_time = node.get_time_finished_use()
            #         frac_finished = (curr_time - start_time) / \
            #             (finish_time - start_time)
            #         incoming_dumpers -= frac_finished

            ### How long is the predicted waiting time, maybe not necessary ###
            # curr_available_time = node.all_on_the_way_finished_used
            # dumpers_incoming = node.get_dumpers_incoming()
            # dumpers_incoming_on_the_way = dumpers_incoming.copy()

            # for _dumper in node.queue:
            #     dumpers_incoming_on_the_way.pop(_dumper)

            # for _dumper in dumpers_incoming_on_the_way:
            #     if node.class_id == 'loading':
            #         predicted_use_time = _dumper.predict_loading_time(
            #             node) + 1
            #     else:
            #         predicted_use_time = _dumper.predict_dumping_time()

            #     if curr_available_time < dumpers_incoming[_dumper][0]:
            #         curr_available_time = dumpers_incoming[_dumper][0] + \
            #             predicted_use_time
            #     else:
            #         curr_available_time += predicted_use_time

            # dumper_reached_destination = curr_time + distance/dumper.speed

            # if node.class_id == 'loading':
            #     predicted_use_time = dumper.predict_loading_time(node) + 1
            # else:
            #     predicted_use_time = dumper.predict_dumping_time() + 1

            # Waiting time in # loads/ dumps you need to wait for
            # waiting_time = (curr_available_time -
            #                 dumper_reached_destination)/predicted_use_time  # TODO: maybe non-negative values

            # TODO: in minuites instead
            # waiting_time = (curr_available_time -
            #                 dumper_reached_destination)/60  # TODO: maybe non-negative values

            # # If loader not started, and no-one on it's way.
            # if not node.active and incoming_dumpers == 0:
            #     state.append(0)
            # else:
            #     # Add predictive waiting time
            #     state.append(waiting_time)

        # for node in self.nodes:
        #     state.append(node == dumper.get_current_node())

        # distances_argsort = list(np.argsort(list_distances))
        for idx, node in enumerate(running_through_nodes):
            # if list_distances[idx] == 0:
            #     state.append(-10)
            # else:
            state.append((list_distances[idx] / dumper.get_speed())/60)
            # state.append(list_distances[idx] / 1000)    # in km
            # state.append(distances_argsort.index(idx) + 1)

        return np.array(state, dtype=float)
