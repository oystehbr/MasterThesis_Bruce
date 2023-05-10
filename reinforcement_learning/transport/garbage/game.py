import numpy as np
from IPython import embed
NUM = 0


class TravelingGameAI:
    def __init__(self, nodes, edges, dumpers):
        self.nodes = nodes
        self.dumpers = dumpers
        self.edges = edges

        self.reset()

    def reset(self):
        # init the game state
        self.score = 0
        self.rest_dumping_mass = 0
        self.rest_loading_mass = 0
        for node in self.nodes:
            node.trigger()
            if node.class_id == 'dumping':
                self.rest_dumping_mass += node.get_max_capacity()
            else:
                self.rest_loading_mass += node.get_max_capacity()

        for edge in self.edges:
            edge.trigger()

        self.max_loading_mass = self.rest_loading_mass
        self.max_dumping_mass = self.rest_dumping_mass

        # Do not pick up mass if it cannot be dumped somewhere

        # 1: place dumpers somewhere
        for dumper in self.dumpers:
            dumper.trigger()

        # update starting place
        # 2: place nodes and edges
        # update lists of available_time, num_queue, queue

    def get_rest_dumping_mass(self):
        return self.rest_dumping_mass

    def reserve_dump_mass(self, mass):
        self.rest_dumping_mass -= mass

    def get_rest_loading_mass(self):
        return self.rest_loading_mass

    def reserve_load_mass(self, mass):
        self.rest_loading_mass -= mass

    def get_mass_loaded(self):
        return self.max_loading_mass - self.rest_loading_mass

    def get_mass_dumped(self):
        return self.max_loading_mass - self.rest_loading_mass

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
        embed(header='play_step - hvordan ser det ut')
        to_node = self.nodes[np.argmax(action)]
        possible_move = True

        if dumper.get_distance_to_node(to_node) == 0:
            reward = -20

            self.score += reward
            possible_move = False
            dumper.set_direct_reward(reward)
            dumper.add_node_to_completed_routes(to_node)
            return reward, game_over, self.score, possible_move

        if dumper.is_node_used_this_route(to_node):
            reward = -15

            self.score += reward
            possible_move = False
            dumper.set_direct_reward(reward)
            dumper.add_node_to_completed_routes(to_node)
            return reward, game_over, self.score, possible_move

        # add the next node to the path (it is possible)
        dumper.set_next_node(to_node)

        # TODO: if not any of those, should be reward 0, and add it to the complete routes
        # if you can reach the next node
        if to_node.class_id == 'dumping':
            # TODO: also check if the dumping station has enough place
            if dumper.get_amount_of_mass_on() > 0:
                # TODO: when does the dumpers reach the place
                # self.score += dumper.get_amount_of_mass_on()

                dumper.set_destination_finish(True)
                reward = 110  # improve

            else:
                # TODO: also check if the loading station has enough place
                # if going to dumper station, empty
                dumper.set_destination_finish(False)
                reward = 0   # TODO: maybe improve

        if to_node.class_id == 'loading':
            if dumper.get_amount_of_mass_on() < dumper.get_mass_capacity():
                # TODO: when does the dumpers reach the place
                # self.score += dumper.get_amount_of_mass_on()
                dumper.set_destination_finish(True)

                # TODO: need to check when loading, how much improve
                reward = 100
            else:
                # if going to loading station, full
                dumper.set_destination_finish(False)
                reward = 0

        # If node id is just normal
        if reward < 0:
            embed(header='does this happen??')
            self.score += reward

        dumper.add_node_to_actual_completed_routes(to_node, reward)
        dumper.set_direct_reward(reward)

        # 2: move

        # 5: update UI maybe?

        # TODO: could do this smarter with the  rest_mass_dumping in the reset
        # sum_dump_mass = 0
        # sum_load_mass = 0
        # for node in self.nodes:
        #     if node.class_id == 'loading':
        #         sum_load_mass += node.get_rest_capacity()
        #     else:
        #         sum_dump_mass += node.get_rest_capacity()

        # # TODO: should be improved -
        # if sum_dump_mass == 0 or sum_load_mass == 0:
        #     game_over = True

        # 6: return reward, game_over, self.score
        return reward, game_over, self.score, possible_move
