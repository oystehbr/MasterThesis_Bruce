import helper
import numpy as np
from IPython import embed
import copy
import time
from agent_node2 import AgentNode
# TODO: add nodes that aren't loading or dumping nodes

# TODO: make edge adding a method, and get_all_connected_edges
# right after adding all edges


class Node:
    """
    After adding all the edges to every node. You need to 
    call set_all_connected_nodes. Probably also calculate all 
    values inside each edge. 
    """
    class_id = 'node'

    node_counter = 0
    nodes = []

    def __init__(self, max_capacity=0, time=None, id=None, coordinates=None):
        self.node_id = Node.node_counter
        Node.node_counter += 1

        if id != None:
            self.node_id = id

        self.coordinates = coordinates
        self.max_capacity = max_capacity
        self.time = time
        self.edges_to_node = {}
        self.edges = []
        self.connected_nodes = []
        self.connected_node_ids = []
        self.shortest_observed_distance_to_special_node = {}
        self.n_games = -2
        self.last_time_resetting = 0
        self.start_active = False
        self.trigger()

    def trigger(self):
        self.rest_capacity = self.max_capacity
        self.reserved_capacity = 0
        # TODO: restart everything before new game

        self.used = False
        self.active = self.start_active
        self.available_time = None
        self.amount_currently_here = 0
        self.last_time_used = 0
        self.all_on_the_way_finished_used = 0
        self.queue = []
        self.dumpers_incoming = {}
        self.time_scheduling = {}
        self.n_games += 1

    def set_start_active(self, start_active):
        self.start_active = start_active

    def set_game(self, game):
        self.game = game

    def set_route_opt(self, route_opt):
        self.route_opt = route_opt

    def __str__(self):
        return f'N{self.node_id}'

    def __repr__(self):
        return f'N{self.node_id}'

    def add_edge(self, edge):
        to_node = edge.get_to_node()
        self.edges_to_node[to_node.get_id()] = edge
        self.connected_nodes.append(to_node)
        self.connected_node_ids.append(to_node.get_id())
        self.edges.append(edge)

    # TODO: delete old
    # def create_agent(self, loading_or_dumping_nodes):
    #     self.node_agent = AgentNode(
    #         self, Node.nodes, loading_or_dumping_nodes)

    def create_agent(self, loading_or_dumping_nodes):
        self.node_agent = AgentNode(
            self, Node.nodes, loading_or_dumping_nodes)

    def add_time_all_queue_finished(self, more_time):
        self.all_on_the_way_finished_used += more_time
        # TODO: maybe use this in the Route_opt_modified - function

    def get_time_scheduling(self):
        return self.time_scheduling

    def get_agent(self):
        return self.node_agent

    def get_coordinates(self):
        return self.coordinates

    def get_edges(self):
        return self.edges

    def is_isolated(self):
        return len(self.edges) == 0

    def get_edge_to_node(self, node):
        return self.edges_to_node[node.get_id()]

    def get_edge_distance_to_node(self, node):
        return self.get_edge_to_node(node).get_distance()

    def get_best_edge_to_node(self, to_node):
        return self.best_predictions[f'{to_node.get_id()}']

    def get_all_connected_nodes(self):
        return self.connected_nodes

    def add_dumper_incoming(self, dumper, start_time):
        pred_driving_time = dumper.get_current_node(
        ).get_shortest_observed_distance_to_node(self)/dumper.get_speed()
        pred_time = start_time + pred_driving_time

        self.dumpers_incoming[dumper] = [pred_time]

    # def add_predicted_finished_time_incoming(self, dumper, pred_end_time):
    #     self.dumpers_incoming[dumper].append(pred_end_time)

    def remove_dumper_incoming(self, dumper):
        self.dumpers_incoming.pop(dumper)

    def get_dumpers_incoming(self):
        return self.dumpers_incoming

    def get_all_connected_node_ids(self):
        return self.connected_node_ids

    def set_max_capacity(self, max_capacity):
        self.max_capacity = max_capacity

    def get_max_capacity(self):
        return self.max_capacity

    def set_available_time(self, available_time):
        """
        How long to next will be there 
        """
        self.available_time = available_time

    def initialize_shortest_observed_distance_to_node(self, start_dict):

        start_dict[f'{self.get_id()}'] = 0
        self.shortest_observed_distance_to_special_node = start_dict

        for node in self.connected_nodes:
            self.add_shortest_observed_distance_to_special_node(
                node, self.get_edge_distance_to_node(node))

        # for node in nodes:
        #     if node in self.connected_nodes:
        #         self.add_shortest_observed_distance_to_special_node(
        #             node, self.get_edge_distance_to_node(node))

        #     if node != self:
        #         self.add_shortest_observed_distance_to_special_node(
        #             node, distance)
        #     else:
        #         self.add_shortest_observed_distance_to_special_node(node, 0)

    def set_best_predictions(self):
        self.best_predictions = {}
        for key in self.shortest_observed_distance_to_special_node.keys():
            shortest = self.shortest_observed_distance_to_special_node[key]
            edges = self.get_edges()
            self.best_predictions[key] = [0]*len(edges)
            for idx, (edge, to_node) in enumerate(zip(edges, self.get_all_connected_nodes())):
                distance_to_node = edge.get_distance()
                distance_from_to_node = to_node.get_shortest_observed_distance_to_node2(
                    key)

                if shortest - distance_to_node == distance_from_to_node:
                    self.best_predictions[key][idx] = 1
                    break

    def get_shortest_observed_distance_to_node(self, node):
        """
        Return the shortest observed distance, if observed -> if not 
        use airdistance
        """
        return self.shortest_observed_distance_to_special_node[f'{node.get_id()}']

    def get_shortest_observed_time_to_node(self, node, dumper):
        return self.get_shortest_observed_distance_to_node(node)/dumper.get_speed()

    def get_shortest_observed_distance_to_node2(self, node_id):
        'Returns: Distance, next_node (or None)'

        return self.shortest_observed_distance_to_special_node[node_id]

    def add_shortest_observed_distance_to_special_node(self, node, distance):

        if f'{node.get_id()}' in self.shortest_observed_distance_to_special_node:
            if self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] > distance:
                self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] = int(
                    distance)

        else:
            self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] = int(
                distance)

    def add_shortest_observed_distance_to_special_node2(self, node_id, distance, next_node=None):
        'If knowing that the node id is already in the dictionary, and it is a better distance'

        self.shortest_observed_distance_to_special_node[node_id] = int(
            distance)

    def add_and_get_shortest_observed_distance_to_special_node(self, node, distance):

        if f'{node.get_id()}' in self.shortest_observed_distance_to_special_node:
            if self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] > distance:
                self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] = int(
                    distance)

                # TODO: may reset memory when new distance, if not bruces algorithm
                # self.get_agent().reset_memory()
                # self.last_time_resetting = self.n_games

        else:
            self.shortest_observed_distance_to_special_node[f'{node.get_id()}'] = int(
                distance)

        return self.shortest_observed_distance_to_special_node[f'{node.get_id()}']

    def add_queue(self, dumper):
        if not (dumper in self.queue):
            self.queue.append(dumper)

    def get_num_queue(self):
        return len(self.queue)

    def use(self, time, dumper):
        self.used = True
        self.time_start_using = time

        # Will not be turned off if not active
        if self.active:
            time_since_last_time_used = time - self.last_time_used
        else:
            time_since_last_time_used = 1

        self.time_scheduling[helper.get_time_formatted(time)] = dumper
        self.active = True
        return time_since_last_time_used

    def get_time_start_using(self):
        return self.time_start_using

    def set_time_finished_use(self, time_finished_use):
        self.time_finished_use = time_finished_use
        self.all_on_the_way_finished_used = time_finished_use

    def get_time_finished_use(self):
        return self.time_finished_use

    def finished_using(self, dumper, time_finished):
        self.used = False
        self.last_time_used = time_finished
        self.queue.remove(dumper)
        self.time_scheduling[helper.get_time_formatted(
            time_finished)] = f'FINISHED, queue: {self.queue}'

    def is_used(self):
        return self.used

    def get_rest_capacity(self):
        return self.rest_capacity

    def get_id(self):
        return self.node_id

    def distance_to_node(self, node):
        """
        Get the distance to the node, if connected. 
        Otherwise 0
        """
        node_id = node.get_id()
        if node_id in self.edges_to_node:
            return self.edges_to_node[node_id].get_distance()

        return 0


class Dumping(Node):
    class_id = 'dumping'

    def __init__(self, max_capacity, mass_type=None, time=None, id=None, coordinates=None):
        super().__init__(max_capacity, time, id, coordinates)
        self.mass_type = mass_type

        self.set_start_active(True)

    def __str__(self):
        return f'D{self.node_id}'

    def __repr__(self):
        return f'D{self.node_id}'

    def trigger(self):
        self.total_dumped = 0
        super().trigger()

    def reserve_mass(self, dumper):
        mass_reserved = min(self.max_capacity -
                            self.reserved_capacity, dumper.get_mass_capacity())
        self.reserved_capacity += mass_reserved

        self.game.reserve_dump_mass(self, mass_reserved)

    def get_mass_type(self):
        return self.mass_type

    def finished_using(self, dumper, time_finished):
        super().finished_using(dumper, time_finished)

        self.total_dumped += self.amount_dumped
        self.rest_capacity -= self.amount_dumped

    def dump_mass(self, amount):
        """
        return the amount that could not be dumped. If 
        0, it dumped all the mass. 
        """

        self.amount_dumped = min(amount, self.rest_capacity)
        amount_not_dumped = amount - self.amount_dumped

        return amount_not_dumped


class Loading(Node):
    class_id = 'loading'

    def __init__(self, max_capacity, mass_type=None, time=None, id=None, coordinates=None, loader=None):
        super().__init__(max_capacity, time, id, coordinates)
        self.loader = loader
        self.mass_type = mass_type

        if loader != None:
            self.loader.set_location_node(self)
            # TODO: can remove these, and call loader everytime, but worse efficient
            self.loading_rate = self.loader.get_loading_rate()  # TODO: s/ tonn
            self.mass_independent_loading_time = self.loader.get_mass_independent_loading_time()

        self.total_loaded = 0
        self.set_start_active(True)

    def __str__(self):
        return f'L{self.node_id}'

    def __repr__(self):
        return f'L{self.node_id}'

    def trigger(self):
        self.total_loaded = 0
        super().trigger()

    def reserve_mass(self, dumper):
        mass_reserved = min(self.max_capacity -
                            self.reserved_capacity, dumper.get_mass_capacity())
        self.reserved_capacity += mass_reserved

        self.game.reserve_load_mass(self, mass_reserved)

    def get_mass_type(self):
        return self.mass_type

    def get_loading_rate(self):
        return self.loading_rate

    def get_mass_independent_loading_time(self):
        return self.mass_independent_loading_time

    def finished_using(self, dumper, time_finished):
        super().finished_using(dumper, time_finished)

        self.total_loaded += self.amount_loaded
        self.rest_capacity -= self.amount_loaded

    def load_mass(self, max_load_amount):
        """
        return the amount that could be loaded into the dumper
        """

        # load max amount, or the rest that can be loaded
        self.amount_loaded = min(max_load_amount, self.get_rest_capacity())

        return self.amount_loaded


class Parking(Node):
    class_id = 'parking'

    def __init__(self, max_capacity=0, time=None, id=None, coordinates=None):
        super().__init__(max_capacity, time, id, coordinates)

    def __str__(self):
        return f'P{self.node_id}'

    def __repr__(self):
        return f'P{self.node_id}'

    def get_mass_type(self):
        return 'parking'


class Edge:
    # TODO: remove default value
    def __init__(self, from_node, to_node, id, distance=0, gps=None, reversed=False):
        self.edge_id = id
        self.from_node = from_node
        self.to_node = to_node
        self.distance = distance
        self.from_node.add_edge(self)
        self.gps = gps

        self.trigger()
        if not (gps is None):
            self.calculate_distance()
        if not reversed:
            self.create_reverse_edge()  # add an edge the other way

        # TODO: add statistics of route

    def __str__(self):
        return self.edge_id

    def __repr__(self):
        return self.edge_id

    def trigger(self):
        self.visitors = {}

    def get_from_node(self):
        return self.from_node

    def get_to_node(self):
        return self.to_node

    def use(self, dumper):
        # TODO: isn't called yet, maybe return how LONG time used?
        self.visitors[dumper] = dumper.get_amount_of_mass_on()

    def finish(self, dumper):
        # TODO: isn't called yet
        self.visitors.pop(dumper)

    def get_num_visit(self):
        return len(self.visitors)

    def set_statistics_of_edge(self):
        # TODO: create statistics, ready for fuel model, from gps data
        NotImplemented

    def get_statistics_of_edge(self):
        # TODO: give out list of statistics, 30 sec interval.
        NotImplemented

    def set_fuel_consumption(self):
        NotImplemented

    def get_fuel_consumption(self):
        NotImplemented

    def calculate_distance(self):
        self.distance = self.gps['distance_cumulative'].values[-1]

    def set_distance(self, distance):
        # Maybe do it for both ways.
        self.distance = distance

    def get_distance(self):
        return self.distance

    def set_reversed_edge(self, reversed_edge):
        self.reversed_edge = reversed_edge

    def get_reversed_edge(self):
        return self.reversed_edge

    def create_reverse_edge(self):
        # reversed_edge = copy.deepcopy(self)

        # reversed_edge.from_node = self.to_node
        # reversed_edge.to_node = self.from_node
        reversed_edge = Edge(self.to_node, self.from_node,
                             self.edge_id, self.distance, self.gps, reversed=True)

        # reversed_edge.gps = self.gps[::-1]
        # reversed_edge.set_statistics_of_edge()
        self.set_reversed_edge(reversed_edge)
        reversed_edge.set_reversed_edge(self)


# NODES = [Loading(10000, 40), Loading(10000, 40),
#          Dumping(10000, 10), Dumping(10000, 10)]
