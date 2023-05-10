from nodes_edges import Node
import random
from IPython import embed
import numpy as np


class Dumper:
    dumper_counter = 0
    dumpers = []
    """
    Controlling the specifications of the dumper. Where it is and where it is heading, and the time it takes to 
    fulfil the current task. 
    """

    def __init__(self, mass_capacity, speed, unloading_rate=None,
                 mass_independent_dumping_time=None, dumper_id=None):
        self.dumper_num = Dumper.dumper_counter
        Dumper.dumper_counter += 1

        self.mass_capacity = mass_capacity
        self.speed = speed
        self.unloading_rate = unloading_rate
        self.dumper_id = dumper_id
        self.mass_independent_dumping_time = mass_independent_dumping_time
        Dumper.dumpers.append(self)

    def trigger(self, dumper_position='fixed'):
        """
        Some trigger, to know which time is passed
        """

        self.counter = 0
        self.time_to_next_prediction = 0
        self.amount_of_mass_on = 0
        self.direct_reward = 0
        self.waiting_time = 0
        self.game_waiting_time = 0
        self.game_total_distance = 0
        self.time_since_last_used_node = 0
        self.starting_dumper = 0

        self.loading = False
        self.dumping = False
        self.driving = False
        self.destination_finish = False
        self.active = False
        self.reported = False
        self.coffee_break = False
        self.used_coffee_agent = False
        self.parked = False
        self.first_route_finished = False

        self.coffee_action = None
        self.coffee_state = None
        self.num_in_queue = None
        self.loading_node = None
        self.dumping_node = None
        self.previous_node = None
        self.knowledge_mass_differences = None

        self.completed_routes = {}
        self.edges_used = []    # TODO: no functionality
        self.coffee_dict = {}
        self.coffee_list = []
        self.coffee_actions = []
        self.coffee_states = []
        self.coffee_times = []
        self.coffee_all_times = []
        self.coffee_preds = []
        self.coffee_rewards = []
        self.coffee_keys_not_finished = []

        if dumper_position == 'random':
            self.set_random_start_node()
        elif dumper_position == 'fixed':
            # TODO: more stuff here, where to start
            self.set_fixed_start_node()
        elif dumper_position == 'parking':
            self.set_parking_start_node()
        else:
            embed(header='something wrong in dumpers setting intial pos')

        self.add_new_key_to_complete_routes(self.counter, 0)
        self.routes = [self.get_current_node()]
        self.nodes_used_this_route = [self.get_current_node().get_id()]

    def set_parked(self, parked):
        self.parked = parked

    def is_parked(self):
        return self.parked

    def add_knowledge_mass_differences(self, knowledge_mass_differences):
        self.knowledge_mass_differences = knowledge_mass_differences

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt
        self.loading_nodes = self.route_opt.loading_nodes
        self.parking_nodes = self.route_opt.parking_nodes

    def __str__(self):
        return f'Dumper num: {self.dumper_num}'

    def __repr__(self):
        return f'Dumper num: {self.dumper_num}'

    def get_num(self):
        return self.dumper_num

    def set_time_since_last_used_node(self, time):
        self.time_since_last_used_node = time

    def get_time_since_last_used_node(self):
        return self.time_since_last_used_node

    def add_new_key_to_complete_routes(self, key, start_time):
        self.completed_routes[f'{key}'] = {
            'full_route': [self.get_current_node()],
            'full_reward': [0],
            'state_info': [],
            'start_end_time': [start_time],
            'actual_route': [self.get_current_node()],
            'mass': [self.get_amount_of_mass_on()],
            'actual_reward': [],
        }

    def add_new_info_to_prev_completed_route(self, key, val):
        self.completed_routes[f'{self.counter - 1}'][key] = [val]

    def add_new_info_to_current_completed_route(self, key, val):
        self.completed_routes[f'{self.counter}'][key] = [val]

    def add_waiting_time(self, time):
        self.waiting_time += time

        # If you are where you should be, do not turn the dumper on before you need it.
        if len(self.get_current_completed_route()['actual_route']) > 1:
            if self.first_route_finished:
                self.game_waiting_time += time

    def add_waiting_time_coffee(self):
        if self.first_route_finished:
            self.game_waiting_time += self.get_coffee_break_time()

    def get_waiting_time(self):
        return self.waiting_time

    def reset_waiting_time(self):
        self.waiting_time = 0

    def get_game_waiting_time(self):
        return self.game_waiting_time

    def get_game_total_distance(self, parking_nodes):
        parking_node = parking_nodes[0]
        find_parking_distance = self.current_node.get_shortest_observed_distance_to_node(
            parking_node)
        total_distance = self.game_total_distance + find_parking_distance
        total_time_driving = total_distance / self.get_speed()
        return total_distance, total_time_driving

    def add_plan_to_current_completed_route(self):
        self.completed_routes[f'{self.counter}']['plan'] = self.get_plan()
        self.completed_routes[f'{self.counter}']['mass_type'] = self.get_plan()[
            0].get_mass_type()

    def set_on_my_way_to(self, node, start_time):

        node.add_dumper_incoming(self, start_time)  # TODO: maybe add time
        if not node.class_id == 'parking':
            node.reserve_mass(self)
        self.on_my_way_to = node

    def get_on_my_way_to(self):
        return self.on_my_way_to

    def set_plan(self, next_node, start_time):
        self.plan = [next_node]
        self.set_on_my_way_to(next_node, start_time)

    def get_plan(self):
        return self.plan

    def add_coffee_reward(self, coffee_reward):
        self.coffee_rewards.append(coffee_reward)

    def set_used_coffee_agent(self, used):
        self.used_coffee_agent = used
        if not used:
            self.coffee_states = []
            self.coffee_actions = []
            self.coffee_times = []
            self.coffee_pred = []
            self.coffee_keys_not_finished = []

    def get_used_coffee_agent(self):
        return self.used_coffee_agent

    def set_coffee_info(self, state, action, time, pred, loading_plan_counter):
        """
        Saving the necessary info to train the coffee break agent
        """

        self.coffee_states.append(state)
        self.coffee_actions.append(action)
        self.coffee_times.append(time)
        self.coffee_all_times.append(time)
        self.coffee_preds.append(pred)
        self.coffee_state = state
        self.coffee_action = action
        self.coffee_pred = pred
        self.coffee_keys_not_finished.append(loading_plan_counter)

        if loading_plan_counter in self.coffee_dict:
            self.coffee_dict[loading_plan_counter].append([state, action])
        else:
            self.coffee_dict[loading_plan_counter] = [[state, action]]

    def set_coffee_break(self, coffee_break, time):
        if coffee_break != self.coffee_break:
            self.coffee_list.append(time)
        else:
            embed(header='does this even happen? dumpers')

        self.coffee_break = coffee_break
        if not self.coffee_break:
            coffee_break_time = self.coffee_list[-1] - self.coffee_list[-2]
            self.completed_routes[f'{self.counter}']['coffee_break'] = [
                coffee_break_time]

    def get_coffee_break_time(self):
        """
        Can call this when finished a route

        """

        if 'coffee_break' in self.completed_routes[f'{self.counter}']:
            return self.completed_routes[f'{self.counter}']['coffee_break'][0]

        return 0

    def get_coffee_break(self):
        return self.coffee_break

    def set_random_start_node(self):
        """
        Set the current node random, and ensure that it 
        is not an isolated node

        """

        random_node = Node.nodes[
            random.randint(0, len(Node.nodes) - 1)]

        while random_node.is_isolated():
            random_node = Node.nodes[
                random.randint(0, len(Node.nodes) - 1)]

        self.current_node = random_node
        self.next_node = self.current_node

    def set_fixed_start_node(self):

        self.current_node = self.loading_nodes[self.dumper_num % len(
            self.loading_nodes)]
        self.next_node = self.current_node

    def set_parking_start_node(self):
        self.current_node = self.parking_nodes[self.dumper_num % len(
            self.parking_nodes)]
        self.next_node = self.current_node

    def start_edge(self):
        try:
            current_edge = self.current_node.get_edge_to_node(self.next_node)
            current_edge.use(self)
        except AttributeError:
            embed(header='aatt')

    def finish_edge(self):
        """
        Finishing the edge, then setting the new current node

        """
        if self.previous_node != None:
            try:
                previous_edge = self.previous_node.get_edge_to_node(
                    self.current_node)
                previous_edge.finish(self)
            except KeyError:
                embed(header='keysss')

    def get_previous_node(self):
        return self.previous_node

    def end_the_route(self, time):

        self.completed_routes[f'{self.counter}']['start_end_time'].append(time)

    def init_new_route(self, start_time):
        """
        Start new route in the dictionary, for visualization, interpretation
        and for collecting full path information to be able to train the full path
        and not just one single move. 

        Thing early in the path could be important for the results
        """

        self.completed_routes[f'{self.counter}']['mass'].append(
            self.get_amount_of_mass_on())
        self.counter += 1
        self.add_new_key_to_complete_routes(self.counter, start_time)
        self.nodes_used_this_route = [self.get_current_node().get_id()]

    def add_state_info(self, state_info):
        self.completed_routes[f'{self.counter}']['state_info'].append(
            state_info)

    def get_speed(self):
        return self.speed

    def get_routes(self):
        return self.routes

    def get_completed_routes(self):
        return self.completed_routes

    def get_prev_completed_route(self):
        return self.completed_routes[f'{self.counter - 1}']

    def get_current_completed_route(self):
        return self.completed_routes[f'{self.counter}']

    def get_distance_to_node(self, node):
        return self.current_node.distance_to_node(node)

    def set_amount_of_mass_on(self, amount_of_mass_on):
        self.amount_of_mass_on = amount_of_mass_on

    def get_amount_of_mass_on(self):
        return self.amount_of_mass_on

    def get_time_to_next_prediction(self):
        return self.time_to_next_prediction

    def get_mass_capacity(self):
        return self.mass_capacity

    def set_current_node(self, new_current_node):

        if self.current_node != new_current_node:
            edge = self.current_node.get_edge_to_node(new_current_node)
            self.game_total_distance += edge.get_distance()
            self.previous_node = self.current_node  # skipping first
        self.current_node = new_current_node

    def get_current_node(self):
        return self.current_node

    def add_node_to_completed_routes(self, node):
        self.completed_routes[f'{self.counter}']['full_route'].append(
            node)

    def add_node_to_actual_completed_routes(self, node, reward):
        self.completed_routes[f'{self.counter}']['actual_route'].append(
            node)
        self.completed_routes[f'{self.counter}']['actual_reward'].append(
            reward)

    def set_next_node(self, next_node):
        """
        Set the next node and add 
            - time taken to that node
        """

        self.next_node = next_node

        self.nodes_used_this_route.append(next_node.get_id())

        self.finish_edge()
        self.start_edge()

        self.routes.append(next_node)
        self.add_node_to_completed_routes(next_node)

        distance_to_next_node = self.current_node.distance_to_node(
            self.next_node)  # TODO: need to add the speed parameter

        self.time_to_next_node = np.ceil(
            distance_to_next_node/self.get_speed())

        self.driving = True
        self.loading = False
        self.dumping = False

    def get_next_node(self):
        return self.next_node

    def get_time_to_next_node(self):
        return self.time_to_next_node

    def is_driving(self):
        return self.driving

    def start_loading(self, node, time):
        # TODO: remove node, and use current_node?
        # Do not load more, if you can't dump it somewhere
        rest_capacity = self.mass_capacity - self.get_amount_of_mass_on()
        # max_load = mrest_capacity
        # TODO: probably not necessary?: max_load = min(rest_capacity, game.get_total_mass_dumping())

        self.amount_loaded = node.load_mass(rest_capacity)

        # TODO: maybe set mass on after finish_loading -
        # TODO: ((maybemaybe 1.problem if no more mass left then no one should drive anymore))

        self.set_loading_time(node)
        node.set_time_finished_use(time + self.loading_time)

        self.loading_node = node
        self.loading = True
        self.driving = False

        return self.amount_loaded

    def finish_loading(self, time_finished):
        """
        Mass will be "placed" after finished loading

        """

        self.set_amount_of_mass_on(
            self.get_amount_of_mass_on() + self.amount_loaded)
        self.loading_node.finished_using(self, time_finished)
        self.loading = False
        self.first_route_finished = True

    def set_loading_time(self, node):
        self.loading_time = self.amount_loaded * \
            node.get_loading_rate()  # s/ tonn
        self.loading_time += node.get_mass_independent_loading_time()

    def get_loading_time(self):
        return self.loading_time

    def is_loading(self):
        return self.loading

    def is_node_used_this_route(self, node):
        return node.get_id() in self.nodes_used_this_route

    def is_all_possible_nodes_used(self):
        nodes = self.current_node.get_all_connected_node_ids()
        for node in nodes:
            if not (node in self.nodes_used_this_route):
                return False

        return True

    def start_dumping(self, node, time):
        self.amount_not_dumped = node.dump_mass(self.get_amount_of_mass_on())
        self.amount_dumped = self.get_amount_of_mass_on() - self.amount_not_dumped

        self.set_dumping_time()
        node.set_time_finished_use(time + self.dumping_time)

        self.dumping_node = node
        self.dumping = True
        self.driving = False

        return self.amount_dumped

    def finish_dumping(self, time_finished):
        self.set_amount_of_mass_on(self.amount_not_dumped)
        self.dumping_node.finished_using(self, time_finished)
        self.dumping = False
        self.first_route_finished = True

    def predict_dumping_time(self):
        return self.mass_capacity * self.unloading_rate + self.mass_independent_dumping_time

    def predict_loading_time(self, loading_node):
        return self.mass_capacity * loading_node.get_loading_rate() + loading_node.get_mass_independent_loading_time()

    def predict_use_time(self, node):
        if node.class_id == 'loading':
            return self.mass_capacity * node.get_loading_rate() + node.get_mass_independent_loading_time()
        elif node.class_id == 'dumping':
            return self.mass_capacity * self.unloading_rate + self.mass_independent_dumping_time
        else:
            embed(header='big failer, Dumpers')

    def get_dumping_time(self):
        return self.dumping_time

    def set_dumping_time(self):
        self.dumping_time = (self.amount_of_mass_on - self.amount_not_dumped) * \
            self.unloading_rate
        self.dumping_time += self.mass_independent_dumping_time

    def is_dumping(self):
        return self.dumping

    def set_direct_reward(self, direct_reward):
        """
        Max reward for the current task

        """

        # TODO: do this
        self.direct_reward = direct_reward
        # if self.dumper_num == 0:
        #     embed()

        self.completed_routes[f'{self.counter}']['full_reward'].append(
            direct_reward)

    def change_last_direct_reward(self, new_reward):
        self.completed_routes[f'{self.counter}']['full_reward'][-1] = new_reward

        try:
            self.completed_routes[f'{self.counter}']['actual_reward'][-1] = new_reward
        except IndexError:
            # Nothing will happen. This is only if you start at loading node.
            print(end='')

        self.direct_reward = new_reward

    def get_reward(self):
        """
        Max reward for the current task

        """

        return self.direct_reward

    def set_destination_finish(self, destination_finish):
        self.destination_finish = destination_finish
        # TODO: should probably not add to a queue if not dumper or loading

    def set_reported(self, reported):
        self.reported = reported

    def get_reported(self):
        return self.reported

    def get_destination_finish(self):
        return self.destination_finish

    def set_active(self):
        self.active = True

    def get_active(self):
        return self.active

    def set_num_in_queue(self, num_in_queue):
        self.num_in_queue = num_in_queue

    def get_num_in_queue(self):
        return self.num_in_queue

# two_dumpers = [Dumper(40, 10, 0), Dumper(40, 20, 0)]
# four_dumpers = [Dumper(40, 10, 0), Dumper(40, 20, 0),
#                 Dumper(40, 10, 0), Dumper(40, 20, 0)]
