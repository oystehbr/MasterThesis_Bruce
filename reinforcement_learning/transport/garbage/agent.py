import time as timePackage
import sys
import signal
import matplotlib.pyplot as plt
import os
from IPython import embed
from dumpers import Dumper
from nodes_edges import Dumping, Loading, Node, Edge
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


MAX_MEMORY = 10000
BATCH_SIZE = 500
LR = 0.001

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class Agent:
    agents = []

    def __init__(self, nodes, edges, dumpers, exploration_num=0):
        Agent.agents.append(self)
        self.n_games = 0
        self.nodes = nodes
        self.dumpers = dumpers
        self.exploration_num = 0
        self.epsilon = 0    # randomness
        self.gamma = 0.9    # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.memory_positive = deque(maxlen=MAX_MEMORY)
        self.random_choice_prob = 5  # %

        # input: one dumper, and all nodes
        self.input_size = len(self.nodes)*2 + 2*len(edges) + 2
        # self.input_size = len(self.nodes)*3 + 1*len(edges) + 2
        # network = [self.input_size, 1000, len(self.nodes)]
        network = [self.input_size, 150, 100,  150, len(self.nodes)]
        # network = [self.input_size, 120, 110, 120, len(self.nodes)]

        self.model = model.Linear_QNet(network)
        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game, dumper):

        # get what should be sendt into the model
        state = [
            # dumper.get_mass_capacity(),
            dumper.get_amount_of_mass_on() > 0,  # has mas on? -> is that enough information
            dumper.get_speed(),
        ]

        # for dumper_runner in self.dumpers:
        #     if dumper != dumper_runner:
        #         # TODO: maybe need to encode as 0 0 0 1 0 0 0 etc - but will include many input
        #         state.append(dumper_runner.get_current_node().get_num())
        #         state.append(dumper_runner.get_next_node().get_num())

        # TODO: maybe do only add all nodes, not including txhe one you are already on

        for node in self.nodes:
            # node connected by an edge
            # state.append(node.get_rest_capacity() > 0)
            state.append(node.get_rest_capacity() > 0)
            state.append(dumper.get_distance_to_node(node) != 0)
            # TODO: unncessecary??
            # state.append(dumper.is_node_used_this_route(node))
            # TODO: probably not necessary??? - included in edge below?
            # state.append(node.get_num_queue())    # Included in edge below
            # TODO: could add a number telling if it is a Dumping or Loading place -> but maybe it can learn that from itself

        # len(edges)*2*2 # 2 features per edge, and '2 edges per edge' (forward, backward)
        for node in self.nodes:
            for edge in node.get_edges():
                state.append(edge.get_num_visit())
                state.append(edge.get_num_visit_with_mass())
        # TODO: could add: max dumper per same edge. Then connected edges should have some connections

        # TODO: add information about the edges??

        try:
            np.array(state, dtype=int)
        except ValueError:
            embed(header='hihay')

        return np.array(state, dtype=int)  # maybe float?

    # def get_state(self, game, dumper):

    #     # get what should be sendt into the model
    #     state = [
    #         # dumper.get_mass_capacity(),
    #         dumper.get_amount_of_mass_on() > 0,  # has mas on? -> is that enough information
    #         dumper.get_speed(),
    #     ]

    #     for node in self.nodes:
    #         # node connected by an edge
    #         state.append(node.get_rest_capacity())
    #         state.append(dumper.get_distance_to_node(node) != 0)
    #         state.append(dumper.is_node_used_this_route(node))

    #     # for node in self.nodes:
    #     #     if node.class_id == 'dumping' or node.class_id == 'loading':
    #     #         state.append(node.get_rest_capacity())

    #     # len(edges)*2*2 # 2 features per edge, and '2 edges per edge' (forward, backward)
    #     for node in self.nodes:
    #         for edge in node.get_edges():
    #             state.append(edge.get_num_visit())

    #     try:
    #         np.array(state, dtype=int)
    #     except ValueError:
    #         embed(header='hihay')

    #     return np.array(state, dtype=int)  # maybe float?

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
        if len(self.memory_positive) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory_positive, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory_positive

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        # TODO fix this TO go from
        # TODO: train_medium_memory load -> dump (then add all states, and actions for this route)
        # TODO: train_short_memory should be that you can't go from current node to itself,
        #      #############        or any other node that aren't connected.
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, dumper):
        # random moves: tradeoff exploration / exploitation

        self.epsilon = self.exploration_num - self.n_games
        final_move = [0] * len(self.nodes)

        # If random_move or
        random_move = random.randint(0, self.exploration_num + 10) < self.epsilon or \
            random.randint(0, 100) < self.random_choice_prob

        if random_move:
            # TODO: going only to possible nodes,
            move = random.choice(
                dumper.get_current_node().get_all_connected_node_ids())
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1

        # try:
        #     if dumper.get_current_completed_route()['actual_route'][-1] == self.nodes[43]:
        #         embed(header='lolzzz')
        # except:
        #     embed(header='find out')

        return final_move

    def save_model(self, filename):
        torch.save(self.model.state_dict(), 'models/' + filename)
        # TODO: model.train() and model.eval()

    def load_model(self):
        print('MODELS:')
        for file in os.listdir('models'):
            print(''*7 + file)

        print('Write the file you filename you wanna load ')
        finished = False
        while not finished:
            filename = input('>> filename: ')
            try:
                self.model.load_state_dict(
                    torch.load('models/' + filename))
                finished = True
            except FileNotFoundError:
                print('You need to choose a file written above')

        self.trainer = model.QTrainer(self.model, lr=LR, gamma=self.gamma)

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

    def add_route_to_round_memory(self, route):
        # TODO: is this taking much time
        rewards = route['actual_reward']

        ok_trip = (route['actual_route'][0].class_id == 'loading' and route['actual_route'][-1].class_id == 'dumping') or \
            (route['actual_route'][0].class_id ==
             'dumping' and route['actual_route'][-1].class_id == 'loading')

        for i in range(len(rewards)):

            # last one, is the last in the route
            done = i == len(rewards) - 1

            state_info = route['state_info'][i]
            if not done:
                try:
                    next_state = route['state_info'][i+1].get_state()
                except IndexError:
                    embed(header='Indexeerrr')

            else:
                # Will not be used, so just setting it equal
                next_state = state_info.get_state()

            sample = (
                state_info.get_state(),
                state_info.get_prediction(),
                rewards[i],
                next_state,
                done
            )

            self.train_short_memory(*sample)
            self.round_memory.append(sample)

            if rewards[i] > 0:
                self.memory_positive.append(sample)

    def train_current_game(self):
        # TODO: maybe train a sample, more samples

        states, actions, rewards, next_states, dones = zip(*self.round_memory)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        self.train_short_memory(states, actions, rewards, next_states, dones)

        # TODO: if this is necessary
        self.memory += self.round_memory


def train():
    MAX_TIME = 3600

    NODES, EDGES, LOADERS, DUMPERS, NODES_DICT = maps.map3(False)

    plot_scores = []
    plot_scores_mean = []
    plot_left_over_mass = []
    plot_left_over_mass_mean = []
    plot_left_over_mass_per_time = []
    plot_left_over_mass_per_time_mean = []

    record = 0  # TODO: make score the amount of mass that will be moved.
    agent = Agent(NODES, EDGES, DUMPERS)
    model.set_agent(agent)
    game = TravelingGameAI(NODES, DUMPERS)
    amount_of_nodes = len(agent.nodes)
    amount_of_dumpers = len(agent.dumpers)
    start_time = None
    print_information_next_round = False
    plot_shower = 'score'

    while True:

        if not(start_time is None):
            stop_time = timePackage.time()
            time_usage = stop_time - start_time
            print(
                f'### Time used for {num_games} games: {time_usage : .2f} seconds ### \n\n')

        print('OPTIONS:')
        print(">> stop  (STOP PROGRAM)")
        print('>> n     (RUN n GAMES)')
        print('>> embed (DEBUGGING)')
        print(
            ">> random_prob p (PROBABILITY p, [0, 100], to make a random action (correct))")
        print(">> plot_shower (score/ mass, mass_time) (PLOT SHOWING)")
        print('>> save_model filename (SAVE CURRENT MODEL)')
        print('>> load_model (LOAD ALREADY MADE MODEL)')
        print('>> exploration m (EXPLORATION m GAMES)')
        print('>> print_information/ pf (PRINTING INFO FOR NEXT GAME)')
        print("(NOT IMPLEMENTED) >>MAX_TIME t (SIMULATION LENGTH t in seconds, default = 3600 - 1 hour)")
        print('(NOT IMPLEMENTED) >> model.train() and model.eval()')

        input_ = input('Answer: ')
        input_lower = input_.lower()
        input_lower_first = input_lower.split()[0]

        if input_lower == 'stop':
            plt.close()
            exit(0)
        elif input_lower == 'embed':
            embed()
        elif input_lower_first == 'random_prob':
            try:
                agent.set_random_choice_prob(int(input_lower.split()[1]))
            except ValueError:
                print(input_lower.split()[1] + ' is not an int')
        elif input_lower_first == 'plot_shower':
            if input_lower.split()[1] == 'score':
                plot_shower = 'score'
            elif input_lower.split()[1] == 'mass':
                plot_shower = 'mass'
            elif input_lower.split()[1] == 'mass_time':
                plot_shower = 'mass_time'
        elif input_lower_first == 'save_model':
            filename = input_.split()[1]
            agent.save_model(filename)
        elif input_lower_first == 'load_model':
            agent.load_model()
        elif input_lower_first == 'exploration':
            try:
                EXPLORATION_NUM = int(input_lower.split()[1])
            except ValueError:
                print(input_lower.split()[1] + 'is not an int')
        elif input_lower_first in ['print_information', 'pf']:
            print_information_next_round = True
            input_ = '1'
        elif input_lower_first == 'max_time':
            try:
                MAX_TIME = int(input_lower.split()[1])
            except Exception as e:
                print(e)
                print('Could not set new MAX_TIME')

        # print(input_lower.split())
        try:
            num_games = int(input_)
        except:
            num_games = 0

        iter = 0
        start_time = timePackage.time()
        while iter < num_games:
            # get old state
            time = 0
            time_tracker_1000 = 0
            # increment the time, such that we always have a new prediction after the increment
            time_increment = 0
            time_to_next_move = np.zeros(amount_of_dumpers)
            state_old = [None]*amount_of_dumpers
            done_old = [None] * amount_of_dumpers
            reward_old = [None]*amount_of_dumpers
            final_move_old = [None]*amount_of_dumpers
            finished = [False]*amount_of_dumpers
            total_score = 0
            control_score = 0
            game_over = False
            agent.restart_round_memory()

            # if first node is load, start loading
            for dumper in Dumper.dumpers:
                if dumper.get_current_node().class_id == 'loading':
                    dumper.set_destination_finish(True)

            # TODO: if you are at loading in the first iteration - try load.

            # One simulation
            failed = False
            failed_counter = 0
            while ((time < MAX_TIME) and not game_over) and not failed:
                time_to_next_move -= time_increment
                # TODO: should probably clean everything from the dumper, visits etc.
                # If only taking a break or something

                # Looping over dumpers that needs a new move
                activity_idx = np.where(time_to_next_move == 0)[0]
                for idx in activity_idx:
                    dumper = DUMPERS[idx]
                    # TODO: if dumper is ready for next move, if not just remove time on current task
                    dumper.set_current_node(dumper.get_next_node())

                    # TODO: if final destination is because it could not move backwards
                    if dumper.get_destination_finish():  # TODO: reached final destination (name)
                        # ready to start dumping or loading

                        current_node = dumper.get_next_node()
                        # if dumper have no queue ticket, then set it
                        # if agent.n_games == 1 and dumper.get_num() == 16 and time == 1204.0:
                        #     print(time)
                        #     print(dumper)
                        #     embed(header='LOLOLOL uyeah_dump1')

                        # if agent.n_games == 1 and dumper.get_num() == 16 and abs(time - 1636) < 0.01:
                        #     embed(header='uyeah_dump1')

                        if dumper.get_num_in_queue() is None:
                            # If there are someone in queue, that should go next
                            if current_node.get_num_queue() > 0 and not current_node.is_used():
                                time_to_next_move[idx] = 1
                                continue

                            # TODO: add next_time_interaction 1second!!!!

                            current_node.add_queue(dumper)
                            dumper.set_num_in_queue(
                                current_node.get_num_queue())

                            # if current_node.get_id() == 45:
                            #     print('Queue: ', end='')
                            #     print(dumper)
                        else:
                            # if current_node.get_id() == 45:
                            # print('Refresh: ', end='')
                            # print(dumper, end=' ')
                            # print(f'No. {dumper.get_num_in_queue() - 1}')
                            dumper.set_num_in_queue(
                                dumper.get_num_in_queue() - 1)

                        # if agent.n_games == 1 and dumper.get_num() == 16 and abs(time - 1636) < 0.01:
                        #     embed(header='uyeah_dump2')
                        # if agent.n_games == 1 and dumper.get_num() == 16 and time == 1204.0:
                        #     print(time)
                        #     print(dumper)
                        #     embed(header='LOLOLOL uyeah_dump2')

                        if not current_node.is_used() and dumper.get_num_in_queue() == 1:
                            # if current_node.get_id() == 45:
                            #     print('Start: ', end='')
                            #     print(dumper)
                            mass_before = dumper.get_amount_of_mass_on()

                            # TODO: could be in start_ - functions
                            current_node.use()
                            if current_node.class_id == 'loading':
                                # TODO: check if it is possible to load mass here

                                dumper.start_loading(
                                    current_node, time, game)
                                mass_after = dumper.get_amount_of_mass_on()

                                mass_fraction = (
                                    mass_after - mass_before) / dumper.get_mass_capacity()

                                next_time_interaction = dumper.get_loading_time()

                                if next_time_interaction > 10000:
                                    embed(header='pokker 4')

                            else:
                                # TODO: check if it is possible to dump mass here
                                dumper.start_dumping(current_node, time)
                                mass_after = dumper.get_amount_of_mass_on()

                                mass_fraction = (
                                    mass_before - mass_after) / dumper.get_mass_capacity()

                                next_time_interaction = dumper.get_dumping_time()

                            max_reward = dumper.get_reward() * mass_fraction

                            # Finished loading/ dumping, triggers:
                            dumper.change_last_direct_reward(max_reward)
                            dumper.set_destination_finish(False)
                            dumper.set_num_in_queue(None)
                        else:

                            try:
                                # Need to wait to the node is ready to be used.
                                time_finished = current_node.get_time_finished_use()
                                next_time_interaction = time_finished - time + \
                                    1 + dumper.get_num_in_queue()  # TODO: 1 sec to switch, and first in first out
                            except AttributeError:
                                embed(header="3")

                            try:
                                dumper.add_waiting_time(
                                    next_time_interaction)
                            except TypeError:
                                embed(header='10_')

                        if next_time_interaction > 10000:
                            embed(header='pokker 3')

                        time_to_next_move[idx] = next_time_interaction
                    else:
                        # TODO: If finished dumping or loading - calculate the correct reward for path

                        if dumper.is_loading() or dumper.is_dumping() or dumper.is_all_possible_nodes_used():
                            # give some score everytime it loads mass
                            control_score += 1
                            total_score += 1*(MAX_TIME - time)
                            # embed(header='finish loading')

                            time_of_task = 0
                            if dumper.is_loading():
                                time_of_task = dumper.get_loading_time()
                                dumper.finish_loading()
                            elif dumper.is_dumping():
                                time_of_task = dumper.get_dumping_time()
                                dumper.finish_dumping()

                            # used all - and last used was not success dumping or loading
                            if dumper.is_all_possible_nodes_used() and time_of_task == 0:
                                # TODO: if negativ reward, to not discount it
                                dumper.change_last_direct_reward(-5)
                                # TODO: is the above the same?? actual_reward[-1] = -5  # TODO: does this work
                                time_of_task = 1

                            # could end the route also - but give penalty
                            if time_of_task != 0:
                                dumper.end_the_route(time, agent)
                                # TODO: remove, is at the bottom. dumper.init_new_route(time)
                                # TODO: reached load/ dump, when nothing there - can be some penalty

                                route = dumper.get_current_completed_route()

                                calculator = helper.Reward_Calculator(
                                    dumper=dumper,
                                    rewards=route['actual_reward'],
                                    route=route['actual_route'],
                                    waiting_time=dumper.get_waiting_time(),
                                    time_of_task=time_of_task)

                                calculator.calculate_rewards_according_to_time()

                                # if agent.n_games > 150:
                                #     if len(route['actual_route']) == 4:
                                #         embed(header='bad algorithm!')

                                route['actual_reward'] = calculator.get_updated_rewards(
                                )

                                # TODO: need to be previous
                                dumper.add_new_info_to_current_completed_route(
                                    'waiting_time', dumper.get_waiting_time())

                                dumper.add_new_info_to_current_completed_route(
                                    'driving_time', calculator.get_driving_time())

                                dumper.add_new_info_to_current_completed_route(
                                    'mass', dumper.get_amount_of_mass_on())

                                agent.add_route_to_round_memory(route)

                                # and train memory
                                dumper.init_new_route(time)
                                dumper.reset_waiting_time()
                            else:
                                # if for instance, going to dump place, and can't dump - or load
                                alfapha = 2

                            reward_old[idx] = 0
                            if len(route['actual_reward']) > 0:
                                reward_old[idx] = route['actual_reward'][-1]

                        # check if finished
                        if dumper.get_amount_of_mass_on() == 0:
                            if game.get_total_mass_dumping() == 0 or game.get_total_mass_loading() == 0:
                                finished[idx] = True
                                time_to_next_move[idx] = MAX_TIME - time

                        if finished[idx]:
                            continue
                        # This happens if finished with dumping or loading, or went to wrong location

                        # TODO: call the trigger in the dumper class - to select mass etc.

                        # Find a possible move, negativ reward if not
                        possible_move = False
                        state = agent.get_state(game, dumper)
                        not_possible_move_counter = 0
                        while not possible_move:    # TODO: make it bad if you go backwards
                            # get move
                            final_move = agent.get_action(state, dumper)

                            # perform move and get new state    # TODO: remember amount of mass on dumper etc -> special case
                            reward, done, score, possible_move = game.play_step(
                                final_move, dumper, time)   # TODO: make play_step

                            # train short memory (state-parameter depend on time - several dumpers)
                            # TODO: could make an active function inside dumper, so it should have maked one move first.

                            # If it tries to move to a location that isn't connected
                            if not possible_move:
                                not_possible_move_counter += 1
                                agent.train_short_memory(
                                    state, final_move, reward, state, False)

                                # TODO: maybe?
                                # agent.remember(state, final_move,
                                #                reward, state, False)

                            model.set_dumper(dumper)
                            if (not_possible_move_counter + 1) % 500 == 0:
                                agent.set_random_choice_prob(
                                    agent.get_random_choice_prob() + 1)
                                model.set_now(True)
                                embed(header='long time')

                            # TODO: maybe next time stamp will be whenever they have reached a new things
                            # 1. start loading, end loading/ drive from loading, start dumping, end dumping/ drive from dumping

                        # TODO: HMM
                        # if dumper.get_active():
                        #     agent.train_short_memory(
                        #         state_old[idx], final_move_old[idx], reward_old[idx], state, done_old[idx])
                        #     # the reward is returned when finishing with loading/ dumping
                        #     agent.remember(
                        #         state_old[idx], final_move_old[idx], reward_old[idx], state, done_old[idx])

                        state_info = helper.State(state, final_move)
                        dumper.add_state_info(state_info)
                        # if agent.n_games == 247:
                        #     if dumper.dumper_num == 1:
                        #         if dumper.counter == 10:
                        #             embed(header='10')
                        #         elif dumper.counter == 12:
                        #             embed(header='12')

                        # Store values, can't predict next step before the step shall be taken.
                        state_old[idx] = state
                        final_move_old[idx] = final_move
                        done_old[idx] = done
                        reward_old[idx] = reward
                        if dumper.get_time_to_next_node() > 10000:
                            embed(header='pokker 2')

                        time_to_next_move[idx] = dumper.get_time_to_next_node()

                        dumper.set_active()

                # If all are finished, game_over
                game_over = len(finished) == np.sum(finished)

                time_increment = np.min(time_to_next_move)
                time += time_increment

                if time // 1000 > time_tracker_1000:
                    print(f'Time: {time}, of max_time: {MAX_TIME}')
                    time_tracker_1000 += 2

                if time_increment == 0:
                    failed_counter += 1
                    if failed_counter == 400:
                        failed = True
                    if failed_counter == 400:
                        embed(header='time_increment')
                else:
                    failed_counter = 0

            # if done or time >= MAX_TIME:
            # embed(header="check routes")

            ### TODO: Make function ###

            agent.save_info()
            score_actual = 0
            for dumper in agent.dumpers:
                for i, route_key in enumerate(dumper.completed_routes):
                    try:
                        score_actual += dumper.completed_routes[route_key]['actual_reward'][-1]
                    except IndexError:
                        # TODO: is needed?
                        NotImplemented
                    except KeyError:
                        # TODO: is needed?
                        NotImplemented

            # TODO: Train long memory, should include path to success also, not only
            # the last choice
            # agent.train_long_memory()

            if score_actual > record:
                record = score_actual
                # TODO: save the model
                # agent.model.save()

            print('Game', agent.n_games, 'Control_score', control_score, 'Score',
                  f'{score_actual : .0f}', 'Record:', f'{record:.0f}')
            # TODO: find out why Sum and score are not the same,
            # TODO: ^^probably the first, or last that are counted/ or not

            agent.n_games += 1
            iter += 1
            plot_scores.append(int(score_actual))
            # mean score of the last 10
            plot_scores_mean.append(np.mean(plot_scores[-50:]))

            plot_left_over_mass.append(
                (game.max_load_capacity - game.get_total_mass_loading()) +
                (game.max_dump_capacity - game.get_total_mass_dumping())
            )
            plot_left_over_mass_mean.append(
                np.mean(plot_left_over_mass[-50:]))

            if plot_shower == 'score':
                helper.plot(plot_scores, plot_scores_mean)
            elif plot_shower == 'mass':
                helper.plot_mass(plot_left_over_mass,
                                 plot_left_over_mass_mean)
            elif plot_shower == 'mass_time':
                helper.plot_mass_per_time(plot_left_over_mass_per_time,
                                          plot_left_over_mass_per_time_mean)

            if print_information_next_round:
                print('\n\nINFO:')
                print('NODES:')
                print_information_next_round = False
                for node in NODES:
                    if node.class_id == 'dumping' or node.class_id == 'loading':
                        print(
                            node, f'MAX: {node.get_max_capacity()}, REST: {node.get_rest_capacity()}')

                print('\n\n')

            agent.train_current_game()
            agent.restart_round_memory()
            agent.train_long_memory()
            game.reset()


if __name__ == "__main__":

    from Route_opt import RouteOptimization

    # NODES, EDGES, LOADERS, DUMPERS, NODES_DICT = maps.map3(False)
    tester = RouteOptimization(maps.map3)
    # Commandline: N_GAMES MAX_TIME EXPLORATION RANDOM_PROB SAVE_MODEL_FILENAME

    command_line_args = True
    values = sys.argv[1:6]
    if len(values) == 5:
        tester.set_values_from_commandline(values)
    else:
        command_line_args = False
        print(f'GOT: {len(sys.argv) - 1} of 5 commmandline values')

    print(values)
    tester.main(command_line_args)
