import numpy as np
import helper
from IPython import embed
NUM = 0


class TravelingGameAI:
    def __init__(self, nodes, edges, dumpers):
        self.nodes = nodes
        self.dumpers = dumpers
        self.edges = edges

        self.loading_nodes = []
        self.dumping_nodes = []
        self.parking_nodes = []

        for node in self.nodes:
            if node.class_id == 'loading':
                self.loading_nodes.append(node)
            elif node.class_id == 'dumping':
                self.dumping_nodes.append(node)
            elif node.class_id == 'parking':
                self.parking_nodes.append(node)

        self.reset()

    def reset(self, dumper_position='random'):
        # init the game state
        self.score = 0
        self.lol = False
        self.rest_dumping_mass = {}
        self.rest_loading_mass = {}
        for node in self.nodes:
            node.trigger()

            if node.class_id == 'dumping':
                mass_type = node.get_mass_type()
                if not mass_type in self.rest_dumping_mass:
                    self.rest_dumping_mass[mass_type] = {}

                self.rest_dumping_mass[mass_type][node] = node.get_max_capacity(
                )
            elif node.class_id == 'loading':
                self.rest_loading_mass[node] = node.get_max_capacity()

        for edge in self.edges:
            edge.trigger()

        self.max_loading_mass = self.rest_loading_mass
        self.max_dumping_mass = self.rest_dumping_mass

        # Do not pick up mass if it cannot be dumped somewhere

        # 1: place dumpers somewhere
        for dumper in self.dumpers:
            dumper.trigger(dumper_position)

        self.all_stuff_in_game = {0: {}}
        for dumper in self.dumpers:
            self.all_stuff_in_game[0][dumper] = dumper.get_current_node()

        # update starting place
        # 2: place nodes and edges
        # update lists of available_time, num_queue, queue

    # TODO: not used, if loaded -> it will have somewhere to dump
    # def get_rest_dumping_mass(self):
    #     return self.rest_dumping_mass

    def reserve_dump_mass(self, node, mass):
        self.rest_dumping_mass[node.get_mass_type()][node] -= mass

    def get_theoretical_max_loading(self, MAX_TIME):
        """
        Very limited. Loading rate 10 sek/ ton -> 40 ton dumpers, and mass_indpendent stuff
        Should be better if used for real. Loading

        """

        missed_loading = 0
        time_of_one_loading = (40 * 10 + 30 + 1)
        max_amount_of_loads = MAX_TIME/time_of_one_loading
        MAX_loading_jobs = 0
        for node in self.loading_nodes:
            MAX_loading_jobs += node.get_max_capacity()
            time_scheduling = node.get_time_scheduling()
            if len(time_scheduling) == 0:
                continue
            start_time = helper.get_second_from_time_formatted(
                list(time_scheduling.keys())[0])

            if start_time > (max_amount_of_loads % 1) * time_of_one_loading:
                missed_loading += 1

                start_time_after_first_failed = start_time - \
                    (max_amount_of_loads % 1) * time_of_one_loading

                missed_loading += start_time_after_first_failed // time_of_one_loading

        theoretical_max = (len(self.loading_nodes) *
                           int(max_amount_of_loads) - missed_loading) * 40

        return min(theoretical_max, MAX_loading_jobs)

    def get_rest_loading_mass(self):
        return np.sum(list(self.rest_loading_mass.values()))

    def reserve_load_mass(self, node, mass):
        self.rest_loading_mass[node] -= mass

    def print_information_of_current_game(self):
        print('\n\nINFO:')
        print('NODES:')
        for node in self.nodes:
            if node.class_id == 'dumping' or node.class_id == 'loading':
                print(
                    node, f'MAX: {node.get_max_capacity()}, REST: {node.get_rest_capacity()}')

    def get_loaders_waiting_time(self, dumpers, dumpers_parked, yeah=False):
        # LOADING PARKING
        time_dumpers_choose_parking = []
        list_of_dumpers_parking = np.array(
            dumpers)[np.array(dumpers_parked, dtype=bool)]
        for dumper in list_of_dumpers_parking:
            t = dumper.get_current_completed_route()[
                'start_end_time'][0]
            time_dumpers_choose_parking.append(t)
        waiting_loading_above_timer = np.zeros(
            len(time_dumpers_choose_parking))

        if len(self.parking_nodes) > 1:
            embed(header='need to improve for more parking nodes')

        # STARTING OF THE NODES
        # TODO: forcing people to wanna go to these nodes! - may be able to turn off and on.
        for loading_node in self.loading_nodes:
            # Pretending that the loader is idling to the agents
            if loading_node.start_active:
                _dist = self.parking_nodes[0].get_shortest_observed_distance_to_node(
                    loading_node)
                time_schedule = loading_node.get_time_scheduling()
                first_arrival_formatted, first_dumper = next(
                    iter(time_schedule.items()))
                first_arrival_sec = helper.get_second_from_time_formatted(
                    first_arrival_formatted)
                possible_first_arrival_sec = _dist/first_dumper.get_speed()

                time_loader_alone_start_episode = first_arrival_sec - \
                    possible_first_arrival_sec - first_dumper.starting_dumper

                for index, dumper in enumerate(list_of_dumpers_parking):
                    # If the current dumper could get there earlier - it will be added
                    my_possible_arrival_time = _dist/dumper.get_speed() + dumper.starting_dumper
                    if my_possible_arrival_time < first_arrival_sec:
                        waiting_loading_above_timer[index] += time_loader_alone_start_episode

        loader_alone = [0]*len(self.loading_nodes)

        for idx, node in enumerate(self.loading_nodes):
            end_time = 0
            time_scheduling = node.get_time_scheduling()

            for i, key in enumerate(time_scheduling):
                if i % 2 == 0:
                    start_time = helper.get_second_from_time_formatted(key)

                    if end_time != 0:
                        additional = (start_time - end_time - 1)
                        loader_alone[idx] += additional

                        # LOADING PARKING
                        for index in range(len(time_dumpers_choose_parking)):
                            if time_dumpers_choose_parking[index] < start_time:
                                waiting_loading_above_timer[index] += additional

                else:
                    end_time = helper.get_second_from_time_formatted(key)

        # LOADING PARKING
        waiting_loading_above_timer /= len(self.loading_nodes)
        self.waiting_loading_above_timer = waiting_loading_above_timer
        self.loader_alone = loader_alone

        return loader_alone

    def get_dumping_loading_info(self):

        dumping_nodes_dict = {}
        loading_nodes_dict = {}
        nodes_loading_or_dumping = []
        parking_nodes = []

        for node in self.nodes:
            special = False
            if node.class_id == 'dumping':
                special = True
                curr_dict = dumping_nodes_dict
            elif node.class_id == 'loading':
                special = True
                curr_dict = loading_nodes_dict

            if special:
                nodes_loading_or_dumping.append(node)
                if node.mass_type in curr_dict:
                    curr_dict[node.mass_type].append(node)
                else:
                    curr_dict[node.mass_type] = [node]

        return dumping_nodes_dict, loading_nodes_dict, list(self.rest_loading_mass.keys()), self.parking_nodes

    def get_mass_loaded(self):
        mass_loaded = 0
        for node in self.loading_nodes:
            mass_loaded += node.get_max_capacity() - node.get_rest_capacity()
        return mass_loaded

    def get_load_mass_jobs(self):
        load_mass_jobs = {}
        for node in self.loading_nodes:
            load_mass_jobs[f'{node}'] = node.get_rest_capacity()

        return load_mass_jobs

    def get_dump_mass_jobs(self):
        dump_mass_jobs = {}
        for node in self.dumping_nodes:
            dump_mass_jobs[f'{node}'] = node.get_rest_capacity()

        return dump_mass_jobs

    def get_mass_dumped(self):
        mass_dumped = 0
        for node in self.dumping_nodes:
            mass_dumped += node.get_max_capacity() - node.get_rest_capacity()
        return mass_dumped

        # return self.max_loading_mass - self.rest_loading_mass

    def get_total_waiting_time(self):
        """
        Return total waiting time for dumpers during the game
        """

        total_waiting_time = 0
        for dumper in self.dumpers:
            total_waiting_time += dumper.get_game_waiting_time()

        return total_waiting_time

    def get_total_distance(self):
        """
        Return total distance driven in the current game
        """

        total_distance = 0
        total_driving_time = 0
        for dumper in self.dumpers:
            dumper_total_distance, dumper_total_driving_time = dumper.get_game_total_distance(
                self.parking_nodes)
            total_distance += dumper_total_distance
            total_driving_time += dumper_total_driving_time

        return total_distance, total_driving_time

    def get_min_used_node(self):
        min_used_node = None
        min_used_num = 10000000
        for node in self.nodes:
            if node.class_id == 'dumping' or node.class_id == 'loading':
                if min_used_num > (node.get_max_capacity() - node.get_rest_capacity()):
                    min_used_num = (node.get_max_capacity() -
                                    node.get_rest_capacity())
                    min_used_node = node

        return min_used_node, min_used_num

    def play_step(self, action, dumper, time):
        """
        Will be called when ever something is starting to move, dump or loading
        """

        # TODO: add possibility of going to node that aren't dumping or loading
        game_over = False
        reward = 0

        # find the node, the plan is to go to
        to_node = dumper.get_current_node().get_edges()[
            np.argmax(action)].get_to_node()
        # embed(header='play_step - hvordan ser det ut')

        # if self.lol:
        #     embed(header='loolllz')

        # if len(dumper.get_current_completed_route()['actual_route']) > 234:
        #     if dumper.get_current_completed_route()['actual_route'][234] == self.nodes[25]:
        #         embed(header='fails here')
        #         self.lol = True

        # add the next node to the path (it is possible)
        dumper.set_next_node(to_node)
        # if not time in self.all_stuff_in_game:
        #     self.all_stuff_in_game[time] = {}

        # self.all_stuff_in_game[time][dumper] = dumper.to_node

        # embed(header='')

        # TODO: if not any of those, should be reward 0, and add it to the complete routes
        # if you can reach the next node
        if to_node == dumper.get_on_my_way_to():
            dumper.set_destination_finish(True)
            reward = 100  # improve

        dumper.add_node_to_actual_completed_routes(to_node, reward)
        dumper.set_direct_reward(reward)

        # 6: return reward, game_over, self.score
        return reward, game_over, self.score
