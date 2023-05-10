import json
import torch
import sys
import time as timePackage

from agent_plan2 import AgentPlan, AgentPlanLoading, AgentPlanDumping
import agent_plan2
import agent_coffee_break
from nodes_edges import Node, Loading, Dumping, Edge
from agent_coffee_break import AgentCoffeeBreak
import model
from game_modified import TravelingGameAI
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import helper
import maps
import time
import constants
from tqdm import tqdm
import random


class RouteOptimization:
    def __init__(
        self,
        map,
        map_settings=None,
        num_dumpers=5,
        MAX_TIME=1000000000,
        output_name="output",
    ):
        agent_plan2.set_seed()
        agent_coffee_break.set_seed()

        ########################################################
        ############# Get information from the map #############
        ########################################################
        if map_settings is None:
            nodes, edges, loaders, dumpers, nodes_dict = map(print_connections=False)
        else:
            nodes, edges, loaders, dumpers, nodes_dict = map(
                num_dumpers=map_settings["num_dumpers"],
                rows=map_settings["rows"],
                cols=map_settings["cols"],
                num_loading_nodes=map_settings["num_loading_nodes"],
                loading_rates=map_settings["loading_rates"],
                num_dumping_nodes=map_settings["num_dumping_nodes"],
                pre_loading_nodes=map_settings["pre_loading_nodes"],
                pre_dumping_nodes=map_settings["pre_dumping_nodes"],
                pre_parking_nodes=map_settings["pre_parking_nodes"],
                mass_type=map_settings["mass_type"],
                max_capacity=map_settings["max_capacity"],
                edge_changes=map_settings["edge_changes"],
                print_connections=map_settings["print_connections"],
                seed=map_settings["seed"],
            )

        self.game = TravelingGameAI(nodes, edges, dumpers)
        self.map = map
        self.map_settings = map_settings

        self.all_dumpers = dumpers
        self.dumpers = dumpers
        self.edges = edges
        self.nodes = nodes
        self.node_ids = []
        for node in self.nodes:
            self.node_ids.append(f"{node.get_id()}")

        self.loaders = loaders

        self.len_nodes = len(nodes)

        self.output_name = output_name
        self.MAX_TIME = MAX_TIME

        model.set_route_opt(self)
        self.trigger()

    def trigger(self):
        ########################################################
        ############## Initialize plotting options #############
        ########################################################

        helper.start_plotter()
        self.plot_scores = []
        self.plot_scores_mean = []
        self.plot_mass_loaded = []
        self.plot_mass_loaded_mean = []
        self.plot_mass_dumped = []
        self.plot_mass_dumped_mean = []
        self.plot_mass_loaded_per_time = []
        self.plot_mass_loaded_per_time_mean = []
        self.plot_total_waiting_time = []
        self.plot_total_waiting_time_mean = []
        self.plot_total_distance = []
        self.plot_total_distance_mean = []
        self.plot_time_usage = []
        self.plot_time_usage_mean = []
        self.plot_loaders_waiting_time = []
        self.plot_loaders_waiting_time_mean = []
        self.plot_fuel = []
        self.plot_fuel_mean = []
        self.plot_cost = []
        self.plot_cost_mean = []

        self.plot_shower = "fuel"
        self.possible_plots = [
            "score",
            "mass_loaded",
            "mass_time",
            "mass_dumped",
            "waiting_time",
            "distance",
            "fuel",
            "cost",
            "time",
        ]

        ########################################################
        ############# Set information from the map #############
        ########################################################
        max_edge_distance = 0
        for edge in self.edges:
            max_edge_distance += max(edge.get_distance(), 0)

        # TODO: instead of max_edge_distance: set max.sysMAX??
        shortest_distances = {}
        for node_id in self.node_ids:
            shortest_distances[node_id] = max_edge_distance

        for node in self.nodes:
            node.initialize_shortest_observed_distance_to_node(
                shortest_distances.copy()
            )

        (
            self.dumping_nodes_dict,
            self.loading_nodes_dict,
            self.loading_nodes,
            self.parking_nodes,
        ) = self.game.get_dumping_loading_info()
        self.dumping_nodes = []

        for node in list(self.dumping_nodes_dict.values()):
            self.dumping_nodes += node

        self.loading_or_dumping_nodes = self.loading_nodes + self.dumping_nodes

        ########################################################
        ############# Give information to dumpers ##############
        ########################################################
        for dumper in self.dumpers:
            dumper.set_route_opt(self)

        ########################################################
        ################# Initialize agents ####################
        ########################################################
        self.dumping_plan_agents_dict = {}
        self.dumping_plan_agents_list = []
        self.plan_agents = []
        self.node_agents = []
        self.coffee_break_agent = AgentCoffeeBreak(
            self.nodes,
            self.edges,
            self.dumping_nodes,
            self.loading_nodes,
            self.parking_nodes,
            self.dumpers,
        )

        # AGENT NODES
        for node in self.nodes:
            node.create_agent(self.loading_or_dumping_nodes)
            self.node_agents.append(node.get_agent())
            node.set_game(self.game)
            node.set_route_opt(self)

        # AGENT LOADING - collective agent, dumper independent
        self.loading_plan_agent = AgentPlanLoading(
            self.nodes,
            self.edges,
            self.loading_or_dumping_nodes,
            self.dumping_nodes,
            self.loading_nodes,
            self.parking_nodes,
            self.dumpers,
        )

        self.plan_agents.append(self.loading_plan_agent)

        # AGENT DUMPING - dumper dependent, mass on will determine
        for mass_type in self.dumping_nodes_dict:
            self.dumping_plan_agents_dict[mass_type] = AgentPlanDumping(
                self.nodes,
                self.edges,
                self.loading_or_dumping_nodes,
                self.dumping_nodes_dict[mass_type],
                self.loading_nodes,
                self.dumpers,
                mass_type,
            )

            self.plan_agents.append(self.dumping_plan_agents_dict[mass_type])
            self.dumping_plan_agents_list.append(
                self.dumping_plan_agents_dict[mass_type]
            )

        ########################################################
        ######### Other stuff that needs initializing ##########
        ########################################################

        for agent in self.node_agents + self.plan_agents:
            agent.set_route_opt(self)

        self.start_time = None
        self.lol = False
        self.dumper_position = "parking"
        self.print_information_next_round = False
        self.map_identifier = f"{self.map.__name__}N{len(self.nodes)}E{len(self.edges)}D{len(self.dumping_nodes)}L{len(self.loading_nodes)}"
        self.map_path = f"models/{self.map_identifier}"
        helper.create_path(self.map_path)

        self.finished_training_node_agents = True
        self.parking_agent_active = True
        self.coffee_break_agent_active = True
        self.node_choice = "optimal"

        self.all_stuff_in_games = {}
        self.fixed_base_fuel = sys.maxsize
        self.fixed_base_cost = sys.maxsize
        self.n_games = 0
        self.n_games_plot_starter = 0
        self.last_game = -1
        self.timer_checker = 0
        self.last_game = -1
        self.fuel_benchmark2 = sys.maxsize
        self.time_benchmark2 = sys.maxsize
        self.framework_results_a = (
            [f"{self.map_settings['seed']}-a"] + [sys.maxsize] * 3 + [0]
        )
        self.framework_results_b = (
            [f"{self.map_settings['seed']}-b"] + [sys.maxsize] * 3 + [0]
        )
        self.reset_records()
        self.set_statistics_of_map_and_dumpers()

        self.output_file = open(f"outputs/{self.output_name}.txt", "w")
        self.output_file.close()
        self.error_file = open(f"outputs/errors.txt", "w")
        self.error_file.close()

    def set_statistics_of_map_and_dumpers(self):
        self.worst_loading_rate_node = None
        self.worst_loading_rate = 0
        self.worst_loading_ind_time = 0
        for loading_node in self.loading_nodes:
            self.worst_loading_ind_time = max(
                self.worst_loading_ind_time,
                loading_node.get_mass_independent_loading_time(),
            )

            if loading_node.get_loading_rate() > self.worst_loading_rate:
                self.worst_loading_rate = loading_node.get_loading_rate()
                self.worst_loading_rate_node = loading_node

        self.dumper_avg_speed = 0
        self.dumper_worst_speed = sys.maxsize
        self.dumper_best_speed = 0
        self.dumper_best_capacity = 0
        self.worst_waiting_time_loading = 0
        self.worst_waiting_time_dumping = 0

        for dumper in self.dumpers:
            self.dumper_avg_speed += dumper.get_speed()
            self.dumper_worst_speed = min(self.dumper_worst_speed, dumper.get_speed())
            self.dumper_best_speed = max(self.dumper_best_speed, dumper.get_speed())
            self.dumper_best_capacity = max(
                self.dumper_best_capacity, dumper.get_mass_capacity()
            )

            self.worst_waiting_time_loading += dumper.predict_loading_time(
                self.worst_loading_rate_node
            )
            self.worst_waiting_time_dumping += dumper.predict_dumping_time()

        self.dumper_avg_speed = self.dumper_avg_speed / len(self.dumpers)

        self.longest_distance = 0
        self.shortest_distance = sys.maxsize
        for dumping_node in self.dumping_nodes + self.parking_nodes:
            for loading_node in self.loading_nodes:
                _dist = dumping_node.get_shortest_observed_distance_to_node(
                    loading_node
                )
                self.longest_distance = max(self.longest_distance, _dist)
                self.shortest_distance = min(self.shortest_distance, _dist)

        self.shortest_time = self.shortest_distance / self.dumper_best_speed
        self.longest_time = self.longest_distance / self.dumper_worst_speed

        # TODO: set treshold
        # Driving the longest distance, and wait for 1.1 loadings, is treshold
        # TIME ONLY, should probably be included the time to the parking slot as negative inpact
        self.threshold_reward_load = (
            -1.1
            * (
                self.dumper_best_capacity * self.worst_loading_rate
                + self.worst_loading_ind_time
            )
        ) / 60

    def trigger_before_new_game(self):
        self.time = 0
        self.time_usage = self.MAX_TIME
        self.time_increment = 0
        self.time_tracker_1000 = 0
        self.total_score = 0
        self.control_score = 0
        self.fails = 0

        self.game.reset(self.dumper_position)
        self.all_stuff_in_game = {-1: {}}

        self.len_dumpers = len(self.dumpers)
        self.time_to_next_move = np.zeros(self.len_dumpers)
        self.dumpers_on_coffee_break = np.zeros(self.len_dumpers)
        self.dumpers_parked = np.ones(self.len_dumpers)
        self.state_old = [None] * self.len_dumpers
        self.done_old = [None] * self.len_dumpers
        self.reward_old = [None] * self.len_dumpers
        self.final_move_old = [None] * self.len_dumpers
        self.finished = [False] * self.len_dumpers
        self.parking_time = [None] * self.len_dumpers
        self.dumper_parked_from_start = [False] * self.len_dumpers
        self.game_over = False
        self.loading_plan_rewards = []
        self.parking_state_action_pairs = []
        self.parking_state_action_pairs_is_parking = []
        self.loading_plan_counter = 0

        # TODO: need to restart round memory for all agents.
        for agent_node in self.node_agents:
            agent_node.restart_round_memory()

        for agent_plan in self.plan_agents:
            agent_plan.restart_round_memory()

        self.coffee_break_agent.restart_round_memory()
        self.set_statistics_of_map_and_dumpers()

        # TODO: explain better maybe
        # Find information of dumpers location w.r.t. loading nodes
        shortest_distance_to_node = {}
        distances_to_node = {}
        dumper_distances_to_node = {}
        for loading_node in self.loading_nodes:
            distances_to_node[loading_node] = []

        for dumper in self.dumpers:
            # LOADING PARKING
            amount_of_dumper_in_simulation = sum(self.dumpers_parked == 0)
            state, _ = self.loading_plan_agent.get_state(
                dumper, self.time, amount_of_dumper_in_simulation
            )
            dumper_distances = state[-len(self.loading_nodes) :]
            dumper_distances_to_node[dumper] = dumper_distances
            min_distance = min(dumper_distances)
            shortest_distance_to_node[dumper] = min_distance

            for loading_node, distance in zip(self.loading_nodes, dumper_distances):
                distances_to_node[loading_node].append(distance)

        # sort the shortest distances to loading nodes for all nodes
        for node in distances_to_node:
            distances_to_node[node] = list(np.argsort(distances_to_node[node]))

        dumper_starting_order2 = []

        # Who get the plan first
        while True:
            current_list_final = [-1] * len(distances_to_node)
            index_not_taken = [i for i in range(len(current_list_final))]

            # AS all the position for each node isn't occupied, find new dumper
            while np.min(current_list_final) == -1:
                current_list = []
                for idx, node in enumerate(distances_to_node):
                    if idx in index_not_taken:
                        if len(distances_to_node[node]) > 0:
                            val = distances_to_node[node].pop(0)
                            current_list.append(val)

                unique_list = np.unique(current_list)

                distances_of_unique_list = []

                # Sort the unique list according to distance, so shortest will go first
                for dumper_num in unique_list:
                    distances_of_unique_list.append(
                        shortest_distance_to_node[self.dumpers[dumper_num]]
                    )

                unique_list_sorted = np.array(unique_list)[
                    np.argsort(distances_of_unique_list)
                ]

                dumper_starting_order2 += list(unique_list_sorted)

                # Let the dumper occupie a node
                for dumper_num in unique_list_sorted:
                    dumper_distances = dumper_distances_to_node[
                        self.dumpers[dumper_num]
                    ]

                    min_val_of_index_not_taken = np.min(
                        dumper_distances[index_not_taken]
                    )

                    list_of_possible_indeces = list(
                        min_val_of_index_not_taken == dumper_distances
                    )

                    for index in index_not_taken:
                        if list_of_possible_indeces[index]:
                            index_pred = index

                    current_list_final[index_pred] = dumper_num
                    index_not_taken.remove(index_pred)

                # remove all occurences of node that are used
                for node in distances_to_node:
                    for remove_node in unique_list:
                        if remove_node in distances_to_node[node]:
                            distances_to_node[node].remove(remove_node)

                rest_dumpers = len(distances_to_node[node])

                if rest_dumpers == 0:
                    break

            if rest_dumpers == 0:
                break

        # dumper_starting_order = np.argsort(
        #     np.array(list(shortest_distance_to_node.values())))
        dumper_starting_order_idx = np.array(dumper_starting_order2)
        dumper_starting_order = np.array(self.dumpers)[dumper_starting_order_idx]

        if self.dumper_position == "parking":
            dumper_starting_order = self.dumpers
            if len(self.parking_nodes) > 1:
                # TODO: go back to the earlier system
                embed(header="maybe need to check things out")

        for idx, dumper in enumerate(dumper_starting_order):
            # Choose a new plan for every dumper
            self.dumpers_parked[dumper.get_num()] = 0

            state_plan, finished = self.find_loading_node(dumper)
            if not finished:
                self.time_to_next_move[dumper.get_num()] = idx
                dumper.starting_dumper = idx
                dumper.add_plan_to_current_completed_route()
                dumper.add_new_info_to_current_completed_route("state_plan", state_plan)

            else:
                # If doesn't get a plan from the start => do nothing
                self.time_to_next_move[dumper.get_num()] = self.MAX_TIME + 1

        # One simulation
        self.failed = False
        self.failed_counter = 0

    def logfile_output(self, txt):
        self.output_file = open(f"outputs/{self.output_name}.txt", "a")
        self.output_file.write(txt + "\n")
        self.output_file.close()

    def logfile_error(self, txt):
        self.error_file = open(f"outputs/errors.txt", "a")
        if self.last_game != self.n_games:
            self.error_file.write(f"Game: {self.n_games: 3.0f}\n")

        self.error_file.write(txt + "\n")
        self.error_file.close()
        self.last_game = self.n_games

    def set_values_pre_simulation(self, values):
        self.values_pre_simulation = values
        self.save_model_filename = values["filename"]
        self.statistics_filename = values["statistics_filename"]
        self.num_games = values["num_games"]

        if values["load_model"]:
            self.load_models(values["filename"])

        self.bas(is_printing=False)
        self.set_exploration_num(*values["exploration"])

    def main(self, command_line_args=False):
        self.command_line_args = command_line_args
        while True:
            if not command_line_args:
                self.option_guide()

            self.iter = 0
            self.start_time = timePackage.time()
            if self.num_games > 0:
                for self.iter in tqdm(range(self.num_games)):
                    self.trigger_before_new_game()

                    #######################################
                    ########### Run a simulation ##########
                    #######################################
                    while (
                        self.time < self.MAX_TIME
                        and not self.game_over
                        and not self.failed
                    ):
                        self.simulate_all_next_moves()  # TODO: most time consuming
                        self.status_update_after_moves()

                    ##############################################
                    # Add time for dumper to get back to parking #
                    ##############################################
                    max_time_to_parking_node = 0
                    parking_node = self.parking_nodes[0]
                    if len(self.parking_nodes) > 1:
                        embed(header="need to check several parking ndoes")

                    for dumper in self.dumpers:
                        curr_node = dumper.get_current_node()
                        if curr_node.class_id != "parking":
                            time_to_parking = (
                                curr_node.get_shortest_observed_time_to_node(
                                    parking_node, dumper
                                )
                            )
                            max_time_to_parking_node = max(
                                time_to_parking, max_time_to_parking_node
                            )

                    self.time_usage += max_time_to_parking_node

                    # LOADING PARKING
                    if self.parking_agent_active:
                        # TODO: this should not be done if this is included
                        self.game.get_loaders_waiting_time(
                            self.dumpers, self.dumpers_parked
                        )

                        list_of_dumpers_parking = np.array(self.dumpers)[
                            np.array(self.dumpers_parked, dtype=bool)
                        ]
                        waiting_time_loading_above_timer = (
                            self.game.waiting_loading_above_timer
                        )

                        for index, dumper in enumerate(list_of_dumpers_parking):
                            route = dumper.get_current_completed_route()
                            try:
                                if dumper.get_current_node().class_id != "parking":
                                    continue
                            except:
                                embed(header="check it out 2")

                            # route['time_since_last_used_node'][0] = - \
                            #     waiting_time_loading_above_timer[index]

                            _min_dist = sys.maxsize
                            _max_dist = 0
                            for loading_node in self.loading_nodes:
                                start_node = route["actual_route"][0]
                                _dist = (
                                    start_node.get_shortest_observed_distance_to_node(
                                        loading_node
                                    )
                                )
                                _min_dist = min(_min_dist, _dist)
                                _max_dist = max(_max_dist, _dist)

                            _min_driving_time = _min_dist / dumper.get_speed()
                            _max_driving_time = _min_dist / dumper.get_speed()

                            try:
                                # loaders waiting is 3 times more costly
                                # route['waiting_time'][0] = _min_driving_time + \
                                #     waiting_time_loading_above_timer[index]
                                route["waiting_time"][
                                    0
                                ] = waiting_time_loading_above_timer[index]
                            except:
                                embed(header="check it out")
                            # _min_driving_time = 1

                            calculator = helper.Reward_Calculator(
                                dumper=dumper,
                                route=dumper.get_current_completed_route(),
                                time_of_task=None,  # doesn't do anything anyway
                                nodes=self.nodes,
                                shortest_time_LD=self.shortest_time,
                                longest_time_LD=self.longest_time,
                                longest_distance=self.longest_distance,
                                worst_waiting_time_loading=self.worst_waiting_time_loading,
                                worst_waiting_time_dumping=self.worst_waiting_time_dumping,
                            )

                            plan_reward = calculator.get_plan_reward(
                                larger_driving_time=_min_driving_time
                            )
                            route["plan_reward"] = [plan_reward]

                            state_plan = route["state_plan"][0]
                            sample = (
                                state_plan.get_state(),
                                state_plan.get_prediction(),
                                plan_reward,
                                # not used, so just setting the same
                                state_plan.get_state(),
                                False,  # not used, so just setting done parameter to False
                            )

                            self.loading_plan_agent.add_route_to_round_memory(sample)

                    self.process_info_of_round()
                    self.plot_evaluation()

                    #######################################
                    ####### Train/ restart agents #########
                    #######################################
                    if len(self.plot_cost) > 1:
                        if np.diff(self.plot_cost[-2:])[0] >= 0:
                            self.threshold_reward_load += np.random.normal(0.5)
                        else:
                            self.threshold_reward_load += np.random.normal(-0.5)

                    if self.coffee_break_agent_active:
                        self.coffee_break_agent.train_current_game()
                        self.coffee_break_agent.train_long_memory()

                    if not self.finished_training_node_agents:
                        for agent in self.node_agents:
                            agent.train_current_game()
                            agent.train_long_memory()

                    for agent_plan in self.plan_agents:
                        agent_plan.train_current_game(
                            self.plot_mass_loaded[-1], self.record_loading_mass
                        )
                        agent_plan.train_long_memory()

                if command_line_args:
                    if self.values_pre_simulation["save_model"]:
                        self.save_models(self.save_model_filename)

                    helper.print_statistics_overleaf(
                        framework_a=self.framework_results_a,
                        framework_b=self.framework_results_b,
                        benchmark1_a=self.benchmark1_a,
                        benchmark1_b=self.benchmark1_b,
                        benchmark2=self.benchmark2,
                        filename=self.statistics_filename,
                    )
                    break

    def process_info_of_round(self):
        #######################################
        ####### Improve/ restart agents #######
        #######################################
        self.n_games += 1
        for agent in self.node_agents + self.plan_agents + [self.coffee_break_agent]:
            agent.increase_game_number()

        self.loading_plan_agent.save_info()
        self.coffee_break_agent.save_coffee_break_info()

        #######################################
        ####### Save_all_stuff_in_game ########
        #######################################
        self.all_stuff_in_games[self.n_games] = self.all_stuff_in_game

        #######################################
        ####### Calculate some metrics ########
        #######################################

        self.plot_mass_loaded.append(self.game.get_mass_loaded())
        self.plot_mass_loaded_mean.append(np.mean(self.plot_mass_loaded[-20:]))

        self.plot_mass_loaded_per_time.append(self.plot_mass_loaded[-1] / self.MAX_TIME)
        self.plot_mass_loaded_per_time_mean.append(
            np.mean(self.plot_mass_loaded_per_time[-20:])
        )

        self.plot_mass_dumped.append(self.game.get_mass_dumped())
        self.plot_mass_dumped_mean.append(np.mean(self.plot_mass_dumped[-20:]))

        self.plot_total_waiting_time.append(self.game.get_total_waiting_time())
        self.plot_total_waiting_time_mean.append(
            np.mean(self.plot_total_waiting_time[-20:])
        )

        total_distance, total_driving_time = self.game.get_total_distance()
        self.plot_total_distance.append(total_distance / 1000)
        self.plot_total_distance_mean.append(np.mean(self.plot_total_distance[-20:]))

        self.plot_time_usage.append(self.time_usage)
        self.plot_time_usage_mean.append(np.mean(self.plot_time_usage[-20:]))

        self.plot_loaders_waiting_time.append(
            np.sum(
                self.game.get_loaders_waiting_time(self.dumpers, self.dumpers_parked)
            )
        )
        self.plot_loaders_waiting_time_mean.append(
            np.mean(self.plot_loaders_waiting_time[-20:])
        )

        current_time_usage = self.plot_time_usage[-1]
        current_loaders_waiting_time = self.plot_loaders_waiting_time[-1]
        current_waiting_time = self.plot_total_waiting_time[-1]
        current_distance = self.plot_total_distance[-1]

        current_fuel, current_cost = helper.calculate_fuel_and_cost_of_plan(
            dumpers=self.dumpers,
            parking_nodes=self.parking_nodes,
            distance=current_distance,  # in km
            driving_time=total_driving_time,
            idling_dumpers=current_waiting_time,  # in seconds
            idling_loaders=current_loaders_waiting_time,  # in seconds
            time=current_time_usage,
            dumper_position=self.dumper_position,
            dumpers_parking_time=self.parking_time,
            loading_nodes=self.loading_nodes,
            n_games=self.n_games,
        )

        self.plot_fuel.append(round(current_fuel, 2))
        self.plot_fuel_mean.append(np.mean(self.plot_fuel[-20:]))

        self.plot_cost.append(round(current_cost, 2))
        self.plot_cost_mean.append(np.mean(self.plot_cost[-20:]))

        ############################################
        ### SET RECORDS AND COLLECT GOOD RESULTS ###
        ############################################

        dumpers_parked_from_start = []
        for dumper in self.dumpers:
            total_distance, _ = dumper.get_game_total_distance(self.parking_nodes)
            dumpers_parked_from_start.append(total_distance == 0)

        current_dumpers_required = len(self.dumpers) - sum(dumpers_parked_from_start)

        add_current_to_records = True
        if current_fuel in self.fuels:
            if current_dumpers_required in self.fuels[current_fuel]:
                add_current_to_records = False
            else:
                self.fuels[current_fuel].append(current_dumpers_required)
        else:
            self.fuels[current_fuel] = [current_dumpers_required]

        self.dumpers_required.append(current_dumpers_required)

        if add_current_to_records:
            #################### Distance ####################
            if current_distance < max(self.record_distances):
                idx = np.argmax(self.record_distances)
                self.record_distances[idx] = current_distance
                self.record_distance_games[idx] = self.n_games
                if current_distance < self.record_distance:
                    self.record_distance = current_distance

            ################## Waiting time ##################
            if current_waiting_time < max(self.record_waiting_times):
                idx = np.argmax(self.record_waiting_times)
                self.record_waiting_times[idx] = current_waiting_time
                self.record_waiting_time_games[idx] = self.n_games
                if current_waiting_time < self.record_waiting_time:
                    self.record_waiting_time = current_waiting_time

            ############## Waiting time loaders ##############
            if current_loaders_waiting_time < max(self.record_loaders_waiting_times):
                idx = np.argmax(self.record_loaders_waiting_times)
                self.record_loaders_waiting_times[idx] = current_loaders_waiting_time
                self.record_loaders_waiting_time_games[idx] = self.n_games
                if current_loaders_waiting_time < self.record_loaders_waiting_time:
                    self.record_loaders_waiting_time = current_loaders_waiting_time

            ################### Time usage ###################
            if current_time_usage < max(self.record_time_usages):
                idx = np.argmax(self.record_time_usages)
                self.record_time_usages[idx] = current_time_usage
                self.record_time_usage_games[idx] = self.n_games
                if current_time_usage < self.record_time_usage:
                    self.record_time_usage = current_time_usage

            ###################### Fuel ######################
            if current_fuel < max(self.record_fuels):
                idx = np.argmax(self.record_fuels)
                self.record_fuels[idx] = current_fuel
                self.record_fuel_games[idx] = self.n_games
                if current_fuel < self.record_fuel:
                    self.record_fuel = current_fuel

            ###################### Cost ######################
            if current_cost < max(self.record_costs):
                idx = np.argmax(self.record_costs)
                self.record_costs[idx] = current_cost
                self.record_cost_games[idx] = self.n_games
                if current_cost < self.record_cost:
                    self.record_cost = current_cost

        ################## Loading amount ################
        _game_total_loaded = self.game.get_mass_loaded()
        if _game_total_loaded > self.record_loading_mass:
            self.record_loading_mass = _game_total_loaded

        # TODO: probably delete - better way of getting this informaiton -> txt (maybe included already)
        if self.print_information_next_round:
            self.print_information_next_round = False
            self.game.print_information_of_current_game()

        # TODO: probably delete or improve
        theoretical_max = self.game.get_theoretical_max_loading(self.MAX_TIME)

        min_used_node, min_used_num = self.game.get_min_used_node()

        self.logfile_output(
            f"Game: {self.n_games:3.0f}, "
            + f"Fuel (liter): {current_fuel:6.1f}, "
            + f"Record_fuel (liter): {self.record_fuel:5.1f}, "
            + f"Cost (pay): {current_cost:5.0f}, "
            + f"Record_cost (pay): {self.record_cost:5.0f}, "
            + f"Time (s): {current_time_usage:7.0f}, "
            + f"Record_time (s): {self.record_time_usage:7.0f}, "
            + f"Distance (km): {self.plot_total_distance[-1]:8.2f}, "
            + f"Record_distance (km): {self.record_distance:8.2f}, "
            + f"Loaders_waiting (s): {self.plot_loaders_waiting_time[-1]:5.0f}, "
            + f"Record_loaders_waiting (s): {self.record_loaders_waiting_time:5.0f}, "
            f"Dumpers_waiting (s): {self.plot_total_waiting_time[-1]:5.0f}, "
            + f"Record_dumpers_waiting (s): {self.record_waiting_time:5.0f}, "
            + f"Dumpers required: {current_dumpers_required}, "
            + f"MassL (ton): {self.plot_mass_loaded[-1]:4.0f}/{theoretical_max:4.0f}, "
            + f"Record_mass (ton): {self.record_loading_mass:5.0f}, "
            # f'Node min used: {min_used_node} ({min_used_num: 4.0f} ton), '
            # f'Fails: {self.fails: 4.0f}, ' +
            # f'Score: {score_actual : .0f}, '
            # f'Control_score: {self.control_score}, ' +
        )

        current_score = self.calculate_scoring_criteria(
            current_fuel, current_time_usage
        )

        if current_score > self.framework_results_a[-1]:
            self.framework_results_a[1] = current_dumpers_required
            self.framework_results_a[2] = round(current_fuel)
            self.framework_results_a[3] = round(current_time_usage)
            self.framework_results_a[4] = current_score

        if current_fuel < self.framework_results_b[2]:
            self.framework_results_b[1] = current_dumpers_required
            self.framework_results_b[2] = round(current_fuel)
            self.framework_results_b[3] = round(current_time_usage)
            self.framework_results_b[4] = current_score

    def plot_evaluation(self):
        # helper.plot(**choose_plotter[self.plot_shower])
        # TODO: above is dictionary with the different cases

        # choose_plotter = {
        #     'score': [self.plot_scores, self.plot_scores_mean, 'score'],
        #     'mass_loaded': [self.plot_mass_loaded, self.plot_mass_loaded_mean, 'mass_loaded'],
        #     'score': [self.plot_scores, self.plot_scores_mean, 'score'],
        #     'score': [self.plot_scores, self.plot_scores_mean, 'score'],
        #     'score': [self.plot_scores, self.plot_scores_mean, 'score'],
        # }

        if not self.command_line_args:
            if self.plot_shower == "score":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_scores,
                    self.plot_scores_mean,
                    ylabel="score",
                )
            elif self.plot_shower == "mass_loaded":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_mass_loaded,
                    self.plot_mass_loaded_mean,
                    labels=["loaded"],
                    ylabel="Mass of ton loaded",
                )
            elif self.plot_shower == "mass_dumped":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_mass_dumped,
                    self.plot_mass_dumped_mean,
                    labels=["dumped"],
                    ylabel="Mass of ton dumped",
                )

            elif self.plot_shower == "mass_time":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_mass_loaded_per_time,
                    self.plot_mass_loaded_per_time_mean,
                    ylabel="Mass_of_ton (dumped + moved) * 1/t",
                )

            elif self.plot_shower == "waiting_time":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_total_waiting_time,
                    self.plot_total_waiting_time_mean,
                    ylabel="Total waiting time (minutes)",
                )
            elif self.plot_shower == "distance":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_total_distance,
                    self.plot_total_distance_mean,
                    ylabel="Total distance (km)",
                )
            elif self.plot_shower == "fuel":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_fuel,
                    self.plot_fuel_mean,
                    ylabel="Fuel (L)",
                )
            elif self.plot_shower == "cost":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_cost,
                    self.plot_cost_mean,
                    ylabel="Cost (NOK)",
                )
            elif self.plot_shower == "time":
                helper.plot(
                    self.n_games_plot_starter,
                    self.plot_time_usage,
                    self.plot_time_usage_mean,
                    ylabel="Time (s)",
                )

    def simulate_all_next_moves(self):
        self.time_to_next_move -= self.time_increment

        activity_idx = np.where(self.time_to_next_move == 0)[0]
        for idx in activity_idx:
            dumper = self.dumpers[idx]

            # if self.time > 3442 and self.n_games == 1:
            #     if dumper.get_num() == 19:
            #         embed(header='why going backward, lolzan')

            # Testing coffee break version
            if dumper.get_coffee_break():
                finished_coffee = self.finish_coffee_break(dumper)

                if not finished_coffee:
                    continue

            # TODO: if dumper is ready for next move, if not just remove time on current task

            dumper.set_current_node(dumper.get_next_node())

            # TODO: next five line, only needed if animation # UNNECESSARY
            if not dumper.get_reported():
                if not self.time in self.all_stuff_in_game:
                    self.all_stuff_in_game[self.time] = {}

                self.all_stuff_in_game[self.time][dumper] = [
                    dumper.get_previous_node(),
                    dumper.get_current_node(),
                    dumper.get_amount_of_mass_on(),
                    dumper.get_on_my_way_to(),
                    self.game.get_load_mass_jobs(),
                    self.game.get_dump_mass_jobs(),
                ]
                dumper.set_reported(True)

            if (
                dumper.get_destination_finish()
            ):  # TODO: reached final destination (name)
                # ready to start dumping or loading

                # TODO: check if it should park here, or go further. NEW NEURAL NETWORK,
                # TODO: make new method in dumpers, state_info_choice_load_dump_further???
                if dumper.get_current_node().class_id == "parking":
                    self.dumper_finished_a_route(dumper)
                    self.finished[dumper.get_num()] = True
                    self.time_to_next_move[dumper.get_num()] = sys.maxsize
                    self.parking_time[dumper.get_num()] = self.time
                    continue

                wait = self.dumper_ready_for_loading_dumping(idx, dumper)

                # Should wait, so the first in first out works.
                if wait:
                    continue
            else:
                dumper.set_reported(False)

                # If ¸ dumping or loading, or visit all nodes - calculate the correct reward for path
                # MAY: but not necessary or len(dumper.get_current_completed_route()['actual_route']) > 1000:
                if (
                    dumper.is_loading()
                    or dumper.is_dumping()
                    or len(dumper.get_current_completed_route()["actual_route"]) % 100
                    == 0
                ):
                    self.dumper_finished_a_route(dumper)

                # check if finished: no mass on, and no mass left on jobs
                if (
                    dumper.get_amount_of_mass_on() == 0
                    and self.game.get_rest_loading_mass() == 0
                ):
                    if not dumper.get_on_my_way_to().class_id == "loading":
                        self.finished[idx] = True
                        self.time_to_next_move[idx] = self.MAX_TIME - self.time
                        # coffee
                        for i, coffe_breaker in enumerate(self.dumpers_on_coffee_break):
                            if coffe_breaker:
                                # self.dumpers_on_coffee_break[i] = 0
                                self.finished[i] = True
                                self.time_to_next_move[i] = self.MAX_TIME - self.time

                        # LOADING PARKING
                        for i, is_parked in enumerate(self.dumpers_parked):
                            if is_parked:
                                self.finished[i] = True
                                self.time_to_next_move[i] = self.MAX_TIME - self.time

                        if np.sum(self.finished) == len(self.finished):
                            self.time_usage = self.time

                # If finished or coffee break, do next one
                if self.finished[idx] or dumper.get_coffee_break():
                    continue

                # Find a possible move, negativ reward if not
                state, final_move = self.dumper_find_a_possible_move(dumper)

                dumper.add_state_info(helper.State(state, final_move))

                # Store values, can't predict next step before the step shall be taken.

                self.time_to_next_move[idx] = dumper.get_time_to_next_node()
                dumper.set_active()

    def dumper_ready_for_loading_dumping(self, idx, dumper):
        """
        Function that take care of loading and dumping of dumpers,
        it should return a boolean corresponding to if the dumper
        need to wait a second - such that it doesn't take a short cut.

        return:
            bool: True - need to wait for a second
                False - next time interaction is set
        """

        current_node = dumper.get_next_node()

        if self.failed_counter > 300:
            embed(header="check the fail")

        if dumper.get_num_in_queue() is None:
            # If there are someone in queue, that should go next - wait to make order correct
            if current_node.get_num_queue() > 0 and not current_node.is_used():
                # if someone in queue, let them move now
                self.time_to_next_move[idx] = 1
                return True

            # add to queue, and get order ticket
            current_node.add_queue(dumper)
            dumper.set_num_in_queue(current_node.get_num_queue() - 1)

        else:
            dumper.set_num_in_queue(dumper.get_num_in_queue() - 1)

        # if self.time == 3619 and self.n_games == 1:
        #     embed(header='why going backwards?1')
        # if self.time == 3633 and self.n_games == 1:
        #     embed(header='why going backwards?')

        # if it is not used, and first in line
        if not current_node.is_used() and dumper.get_num_in_queue() == 0:
            # TODO: could be in start_ - functions
            time_since_last_time_used = current_node.use(self.time, dumper)
            # embed(header='here111')
            if current_node.class_id == "loading":
                # TODO: check if it is possible to load mass here

                amount_loaded = dumper.start_loading(current_node, self.time)

                mass_fraction = amount_loaded / dumper.get_mass_capacity()
                next_time_interaction = dumper.get_loading_time()

            else:
                # TODO: check if it is possible to dump mass here
                amount_dumped = dumper.start_dumping(current_node, self.time)

                mass_fraction = amount_dumped / dumper.get_mass_capacity()
                next_time_interaction = dumper.get_dumping_time()

            max_reward = dumper.get_reward() * mass_fraction
            # if doing correct move, give reward if node hasn't been used for a while
            if max_reward > 0:
                max_reward += time_since_last_time_used * 20
                dumper.set_time_since_last_used_node(time_since_last_time_used)

            # Finished loading/ dumping, triggers:
            dumper.change_last_direct_reward(max_reward)
            dumper.set_destination_finish(False)
            dumper.set_num_in_queue(None)

            # TODO: find new plan
        else:
            # Need to wait to the node is ready to be used.
            time_finished = current_node.get_time_finished_use()

            # TODO: could try to add the correct time at the beginning
            next_time_interaction = (
                time_finished - self.time + dumper.get_num_in_queue()
            )  # TODO: 1 sec to switch, and first in first out

            if current_node.class_id == "loading":
                current_node.add_time_all_queue_finished(
                    dumper.predict_loading_time(current_node)
                )
            elif current_node.class_id == "dumping":
                current_node.add_time_all_queue_finished(dumper.predict_dumping_time())

            dumper.add_waiting_time(next_time_interaction)

        self.time_to_next_move[idx] = next_time_interaction

        return False

    def dumper_finished_a_route(self, dumper):
        # give some score everytime it loads mass
        self.control_score += 1

        time_of_task = 0

        # if dumper.get_current_completed_route()['actual_route'][-1].class_id == 'loading':
        #     embed(header='lolzz')
        what = ""
        if dumper.is_loading():
            time_of_task = dumper.get_loading_time()
            dumper.finish_loading(self.time)
            # embed(header='check the queue for loading')
            what = "loading"
            # TODO: is this correct
            dumper.get_current_node().remove_dumper_incoming(dumper)

        elif dumper.is_dumping():
            what = "dumping"
            time_of_task = dumper.get_dumping_time()
            dumper.finish_dumping(self.time)
            dumper.get_current_node().remove_dumper_incoming(dumper)

        elif dumper.is_parked():
            what = "parking"
            time_of_task = 1
            # return
            # embed(header='here11')
            # TODO: new action and train last action/ add it to memory

        # could end the route also - but give penalty
        if time_of_task != 0:
            # Say to dumpers on coffee, that new information is available
            if self.coffee_break_agent_active:
                for idx, coffee_break in enumerate(self.dumpers_on_coffee_break):
                    if coffee_break:
                        # Could be some wierd with order if zero
                        self.time_to_next_move[idx] = 1

            dumper.end_the_route(self.time)
            # TODO: remove, is at the bottom. dumper.init_new_route(time)
            # TODO: reached load/ dump, when nothing there - can be some penalty

            dumper.add_new_info_to_current_completed_route(
                "waiting_time", dumper.get_waiting_time()
            )

            dumper.add_new_info_to_current_completed_route(
                "time_since_last_used_node", dumper.get_time_since_last_used_node()
            )

            dumper.set_time_since_last_used_node(0)

            dumper.add_new_info_to_current_completed_route(
                "mass_end", dumper.get_amount_of_mass_on()
            )

            route = dumper.get_current_completed_route()
            calculator = helper.Reward_Calculator(
                dumper=dumper,
                route=route,
                time_of_task=time_of_task,
                nodes=self.nodes,
                shortest_time_LD=self.shortest_time,
                longest_time_LD=self.longest_time,
                longest_distance=self.longest_distance,
                worst_waiting_time_loading=self.worst_waiting_time_loading,
                worst_waiting_time_dumping=self.worst_waiting_time_dumping,
            )

            calculator.calculate_rewards_according_to_time()

            dumper.add_new_info_to_current_completed_route(
                "driving_time", calculator.get_driving_time()
            )
            route["actual_reward"] = calculator.get_updated_rewards()

            #################################################################
            ##################### Train the node agents #####################
            #################################################################

            # TODO: can make the code below much more simplified. With the new format

            if not self.finished_training_node_agents:
                completed_route = dumper.get_current_completed_route()
                indeces = helper.use_last_states_and_rewards1(
                    completed_route["actual_route"][:-1]
                )

                if len(indeces) > 0:
                    driving_rewards = calculator.get_driving_rewards_distance(indeces)
                    state_info = completed_route["state_info"]
                    node_path = completed_route["actual_route"]

                dict_nodes = {}
                for iter, idx in enumerate(indeces):
                    sample = (
                        state_info[idx].get_state(),
                        state_info[idx].get_prediction(),
                        driving_rewards[iter],
                        # not used, so just setting the same
                        state_info[idx].get_state(),
                        False,  # not used, so just setting done parameter to False
                    )

                    # the node will be route[idx] -
                    self.node = node_path[idx]
                    self.dumper = dumper
                    dict_nodes[f"{node_path[idx]}"] = state_info[idx]
                    node_path[idx].get_agent().add_route_to_round_memory(sample)

                # When doing a thing that isn't the last solution

                # if len(completed_route['actual_route'][:-1]) > 0:
                #     end_node = np.array(self.nodes)[np.where(
                #         state_info[0].get_state() == 1)][0]

                for idx, node in enumerate(completed_route["actual_route"][:-1]):
                    in_final_path = f"{node}" in dict_nodes
                    if in_final_path:
                        not_equal_pred = (
                            dict_nodes[f"{node}"].get_prediction()
                            != state_info[idx].get_prediction()
                        )

                    # TODO: check this out -> not_equal_pred should be set False in the beginning of loop?
                    if not in_final_path or not_equal_pred:
                        # next_node = completed_route['actual_route'][idx + 1]
                        # edge_distance = node.get_edge_to_node(
                        #     next_node).get_distance()

                        # shortest_observed_distance_next = next_node.get_shortest_observed_distance_to_node(
                        #     end_node)
                        # shortest_observed_distance = node.get_shortest_observed_distance_to_node(
                        #     end_node)

                        # alternative_reward = (
                        #     shortest_observed_distance - edge_distance) / shortest_observed_distance_next * 10

                        # if alternative_reward != 10:
                        #     embed(header='checker')

                        # TODO: should it be added to memory?
                        sample = (
                            state_info[idx].get_state(),
                            state_info[idx].get_prediction(),
                            0,  # instead of zero
                            # not used, so just setting the same
                            state_info[idx].get_state(),
                            False,  # not used, so just setting done parameter to False
                        )

                        # the node will be route[idx] -
                        self.node = node_path[idx]
                        self.dumper = dumper
                        node_path[idx].get_agent().add_route_to_round_memory(sample)
                        # print(2)

            if what == "dumping":
                #####################################################################
                ##################### Train dumping plan agent ######################
                #####################################################################
                plan_reward = calculator.get_plan_reward()
                dumper.add_new_info_to_current_completed_route(
                    "plan_reward", plan_reward
                )
                state_plan = route["state_plan"][0]
                sample = (
                    state_plan.get_state(),
                    state_plan.get_prediction(),
                    plan_reward,
                    # not used, so just setting the same
                    state_plan.get_state(),
                    False,  # not used, so just setting done parameter to False
                )

                _dumping_plan_agent = self.dumping_plan_agents_dict[
                    dumper.get_current_node().get_mass_type()
                ]
                _dumping_plan_agent
                _dumping_plan_agent.add_route_to_round_memory(sample)

                # Choose a new plan for the dumper -> new loading point
                state_plan, finished = self.find_loading_node(dumper)
                if finished:
                    return

            elif what == "loading":
                #####################################################################
                ##################### Train loading plan agent ######################
                #####################################################################

                plan_reward = calculator.get_plan_reward()

                # (array([ 0.  ,  1.  , 10.25, 10.12]), [1, 0], 30.0, array([ 0.  ,  1.  , 10.25, 10.12]), False)

                dumper.add_new_info_to_current_completed_route(
                    "plan_reward", plan_reward
                )
                state_plan = route["state_plan"][0]
                sample = (
                    state_plan.get_state(),
                    state_plan.get_prediction(),
                    plan_reward,
                    # not used, so just setting the same
                    state_plan.get_state(),
                    False,  # not used, so just setting done parameter to False
                )

                # if list(state_plan.get_state()) == [0, 1, 10.25, 10.12]:
                #     if plan_reward == 30:
                #         embed(header='what happpend here___')
                self.loading_plan_agent.add_route_to_round_memory(sample)
                self.loading_plan_rewards.append(plan_reward)

                #####################################################################
                ##################### Train coffee break agent ######################
                #####################################################################

                if self.coffee_break_agent_active and dumper.get_used_coffee_agent():
                    dumper.add_new_info_to_current_completed_route(
                        "coffee_state", dumper.coffee_states[-1]
                    )
                    dumper.add_new_info_to_current_completed_route(
                        "coffee_pred", dumper.coffee_preds[-1]
                    )

                    coffee_break_rewards = calculator.get_plan_reward_coffee_break(
                        dumper.coffee_times
                    )

                    dumper.add_waiting_time_coffee()

                    for coffee_i in range(len(coffee_break_rewards)):
                        sample_coffee = (
                            dumper.coffee_states[coffee_i],
                            dumper.coffee_actions[coffee_i],
                            coffee_break_rewards[coffee_i],
                            # not used, so just setting the same
                            dumper.coffee_states[coffee_i],
                            False,  # not used, so just setting done parameter to False
                        )

                        dumper.add_coffee_reward(coffee_break_rewards[coffee_i])

                        self.coffee_break_agent.add_route_to_round_memory(sample_coffee)

                    # if len(dumper.coffee_states) > 1:
                    #     embed(header='heree1111')

                    dumper.set_used_coffee_agent(False)

                # New plan/ dumper point - with correct mass_type
                state_plan = self.find_dumping_node(dumper)
            elif what == "parking":
                return
            else:
                embed(header="what happend")

            dumper.init_new_route(self.time)
            dumper.add_new_info_to_current_completed_route("state_plan", state_plan)
            dumper.add_plan_to_current_completed_route()
            dumper.reset_waiting_time()

        else:
            completed_route = dumper.get_current_completed_route()
            last_part = completed_route["actual_route"][-50:]
            circular, num = helper.check_if_circular(last_part)

            ac_route = completed_route["actual_route"]
            state_info = completed_route["state_info"]

            ac_route[-3].get_agent().model(
                torch.tensor(state_info[-2].get_state(), dtype=torch.float)
            )
            if circular:
                self.logfile_error("circular")
                iter = 0
                # embed(header='lollz')
                # if node is just connected with one other node

                # Train model to prediction will be something else
                idx = 0
                new_pred = False
                while not new_pred:
                    try:
                        node_in_path = ac_route[-(idx % num + 2)]
                    except IndexError:
                        embed(header="what happend")
                    loop_state_info = state_info[-(idx % num + 1)]
                    # embed(header='yeah')

                    sample = (
                        loop_state_info.get_state(),
                        loop_state_info.get_prediction(),
                        -1,
                        # not used, so just setting the same
                        loop_state_info.get_state(),
                        False,  # not used, so just setting done parameter to False
                    )

                    self.lol = False
                    self.dumper = dumper
                    self.node_in_path = node_in_path

                    node_in_path.get_agent().add_route_to_round_memory(sample)
                    new_pred = (
                        loop_state_info.get_prediction()
                        != node_in_path.get_agent().get_action(
                            loop_state_info.get_state(), dumper, choice=self.node_choice
                        )
                    )

                    idx += 1
                    # embed(header='yeah1')

                self.lol = False

    def dumper_find_a_possible_move(self, dumper):
        # Find a possible move, negativ reward if not
        node_agent = dumper.get_current_node().get_agent()
        state = node_agent.get_state(dumper)
        final_move = node_agent.get_action(state, dumper, choice=self.node_choice)

        reward, done, score = self.game.play_step(
            final_move, dumper, self.time
        )  # TODO: make play_step

        # TODO: could make an active function inside dumper, so it should have maked one move first.
        # TOOD: maybe remove: - used for debugging

        return state, final_move

    def start_coffee_break(self, dumper):
        dumper.set_coffee_break(True, self.time)
        self.dumpers_on_coffee_break[dumper.get_num()] = True

        # Coffee break to after next have done a move
        # minimum_to_next_move = np.min(
        #     self.time_to_next_move[self.dumpers_on_coffee_break == 0])
        # self.time_to_next_move[dumper.get_num()] = minimum_to_next_move + 1
        self.time_to_next_move[dumper.get_num()] = sys.maxsize

    def finish_coffee_break(self, dumper):
        # try to find another loading node
        state_plan, no_plan = self.find_loading_node(dumper)

        # still coffee break
        if no_plan:
            # minimum_to_next_move = np.min(
            #     self.time_to_next_move[self.dumpers_on_coffee_break == 0])
            # self.time_to_next_move[dumper.get_num()] = minimum_to_next_move + 1
            self.time_to_next_move[dumper.get_num()] = sys.maxsize

            return False

        self.dumpers_on_coffee_break[dumper.get_num()] = False

        dumper.init_new_route(self.time)
        dumper.add_new_info_to_current_completed_route("state_plan", state_plan)
        dumper.add_plan_to_current_completed_route()
        dumper.reset_waiting_time()
        dumper.set_coffee_break(False, self.time)

        return True

    def find_loading_node(self, dumper):
        # Find the loading nodes that have more mass to move
        more_mass_vec = np.array(list(self.game.rest_loading_mass.values())) > 1

        # If finished all jobs, end the dumper
        if np.sum(more_mass_vec) == 0:
            return 0, True

        # If active and some other dumper is driving
        # LOADING PARKING
        if self.parking_agent_active and np.sum(self.dumpers_parked == 0) > 1:
            more_mass_vec = np.append(more_mass_vec, True)
        else:
            more_mass_vec = np.append(more_mass_vec, False)

        # LOADING PARKING
        amount_of_dumper_in_simulation = sum(self.dumpers_parked == 0)
        loading_state, parking_node = self.loading_plan_agent.get_state(
            dumper, self.time, amount_of_dumper_in_simulation
        )
        final_move, pred = self.loading_plan_agent.get_action(
            loading_state, dumper, more_mass_vec
        )

        # LOADING PARKING
        num = np.argmax(final_move)
        going_parking = False
        if num == len(self.loading_nodes):
            going_parking = True
            if self.time == 0:
                self.dumper_parked_from_start[dumper.get_num()] = True
        else:
            loading_node = self.loading_nodes[num]

        # Is this information necessary?? -> should probably train the coffee_agent afterwards anyway

        if going_parking:
            state_plan = helper.State(loading_state, final_move)
            dumper.set_plan(parking_node, self.time)
            if dumper.get_current_node() == dumper.get_on_my_way_to():
                dumper.set_destination_finish(True)

            dumper.set_parked(True)
            self.dumpers_parked[dumper.get_num()] = 1
            return state_plan, False

        dumper_parked_or_coffee_break = (
            self.dumpers_on_coffee_break + self.dumpers_parked
        )

        ###############################
        #######  Coffee break   #######
        ###############################
        if self.coffee_break_agent_active:
            # If there are more than 1 dumpers driving
            if (
                np.sum(dumper_parked_or_coffee_break == 0) > 1
                or dumper.get_coffee_break()
            ):
                forced_finish_coffee = False
                if np.sum(dumper_parked_or_coffee_break == 0) == 0:
                    # On coffee break, but the remaining are parked, force starting
                    forced_finish_coffee = True

                coffee_break_now = self.call_coffee_break_agent(
                    dumper, pred, final_move, loading_node, forced_finish_coffee
                )

                if coffee_break_now and forced_finish_coffee:
                    embed(header="here something is wrong")

                if coffee_break_now:
                    return 0, True

            else:
                dummy = ""

        # Is actually following the plan
        self.loading_plan_counter += 1
        dumper.set_plan(loading_node, self.time)

        # If already at the plan, set finished its destination
        if dumper.get_current_node() == dumper.get_on_my_way_to():
            dumper.set_destination_finish(True)
        state_plan = helper.State(loading_state, final_move)

        return state_plan, False

    def call_coffee_break_agent(
        self, dumper, loading_pred, final_move, loading_node, forced_finish_coffee=False
    ):
        """

        Return: if dumper should take coffee_break or not
        """

        dumper.set_used_coffee_agent(True)
        pred_reward = loading_pred[final_move.index(1)].item()

        try:
            state = self.coffee_break_agent.get_state(
                dumper, self.time, loading_node, pred_reward
            )  # , pred_reward)
        except Exception as e:
            print(e)
            embed(header="failing in calling coffee break")
            a = ""
        finally:
            # embed(header='after calling coffee_agent')
            a = ""

        try:
            final_move_coffee, pred = self.coffee_break_agent.get_action(state, dumper)
            # embed(header='look at action of coffee break')
        except:
            a = ""
            embed(header="failing in calling coffee break1")
        finally:
            # embed(header='after calling coffee_agent')
            a = ""
        # if self.n_games == 105:
        #     embed(header='withing call_coffee')
        if forced_finish_coffee:
            final_move_coffee = [0, 1]
            # if self.n_games == 31:
            #     embed(header='need to check this')

        dumper.set_coffee_info(
            state, final_move_coffee, self.time, pred, self.loading_plan_counter
        )
        if final_move_coffee.index(1) == 0:
            if not dumper.get_coffee_break():
                # TODO: inside start_coffee -> attach state onto the dumper
                # TODO: if average from here and out -> add index of interest
                # Make count: I have chosen a loading plan -> and keep all rewards when training the
                self.start_coffee_break(dumper)

            return True

        return False

    def find_dumping_node(self, dumper):
        mass_type = dumper.get_current_node().get_mass_type()
        dumping_plan_agent = self.dumping_plan_agents_dict[mass_type]

        more_mass_vec = (
            np.array(list(self.game.rest_dumping_mass[mass_type].values())) > 1
        )

        state = dumping_plan_agent.get_state(dumper, self.time)
        final_move, pred = dumping_plan_agent.get_action(state, dumper, more_mass_vec)
        num = np.argmax(final_move)

        # TODO: add mass type into self.dumping_nodes
        dumping_node = self.dumping_nodes_dict[mass_type][num]
        dumper.set_plan(dumping_node, self.time)

        state_plan = helper.State(state, final_move)

        return state_plan

    def status_update_after_moves(self):
        # If all are finished, game_over
        self.game_over = len(self.finished) == np.sum(self.finished)

        self.time_increment = np.min(self.time_to_next_move)
        self.time += self.time_increment

        if self.time // 1000 > self.time_tracker_1000:
            self.time_tracker_1000 += 2

        if self.time_increment == 0:
            self.failed_counter += 1
            if self.failed_counter == 400:
                self.failed = True
            if self.failed_counter == 400:
                embed(header="time_increment")
                # TODO: driving back and forth -> takes 0 seconds.
        else:
            self.failed_counter = 0

    def save_models(self, foldername):
        foldername = f"{self.map_path}/{foldername}"

        for agent in self.node_agents:
            agent.save_model(foldername)

        for agent_plan in self.plan_agents:
            agent_plan.save_model(foldername)

        self.coffee_break_agent.save_model(foldername)

        ##################################################
        ########  save shortest observed route ###########
        ##################################################

        list_of_dicts = []
        for node in self.nodes:
            list_of_dicts.append(node.shortest_observed_distance_to_special_node)

        with open(f"{foldername}/dicts_shortest_observed_route.json", "w") as fout:
            json.dump(list_of_dicts, fout)

        ##################################################
        ########  save lowest fuel value found ###########
        ##################################################
        with open(f"{foldername}/minimum_fuel.txt", "w") as fout:
            try:
                min_fuel = min(self.plot_fuel)
            except:
                # If want to change name of saved models
                min_fuel = (self.fixed_base_fuel) / 2
            fout.write(f"{min_fuel}")

        with open(f"{foldername}/minimum_cost.txt", "w") as fout:
            try:
                min_cost = min(self.plot_cost)
            except:
                # If want to change name of saved models
                min_cost = (self.fixed_base_cost) / 2
            fout.write(f"{min_cost}")

    def load_models(self, foldername=None):
        # TODO: in load and save model -> need a list of all dumping agents.

        if foldername is None:
            foldername = self.loading_plan_agent.load_model(map_path=self.map_path)
        else:
            self.loading_plan_agent.load_model(
                map_path=self.map_path, foldername=foldername
            )

        # To be able to go back from load models
        if foldername == "NO":
            return

        for agent in self.node_agents:
            agent.load_model(map_path=self.map_path, foldername=foldername)

        for mass_type in self.dumping_plan_agents_dict:
            agent = self.dumping_plan_agents_dict[mass_type]
            agent.load_model(map_path=self.map_path, foldername=foldername)

        self.coffee_break_agent.load_model(
            map_path=self.map_path, foldername=foldername
        )

        ##################################################
        ########  add shortest observed route ############
        ##################################################

        with open(
            f"{self.map_path}/{foldername}/dicts_shortest_observed_route.json", "r"
        ) as fin:
            list_of_dicts = eval(fin.read())

        for idx, node in enumerate(self.nodes):
            node.shortest_observed_distance_to_special_node = list_of_dicts[idx]

        for node in self.nodes:
            node.set_best_predictions()

        ##################################################
        ########  load lowest fuel value found ###########
        ##################################################
        with open(f"{self.map_path}/{foldername}/minimum_fuel.txt", "r") as fin:
            self.fixed_base_fuel = float(fin.readline()) * 2

        with open(f"{self.map_path}/{foldername}/minimum_cost.txt", "r") as fin:
            self.fixed_base_cost = float(fin.readline()) * 2

        self.coffee_break_agent.create_time_converter_constant()
        self.set_exploration_num(0)
        self.set_random_choice_prob(0)

    def reset_records(self):
        # reset records:
        self.last_reset_n_games = self.n_games
        self.record = 0
        self.record_loading_mass = 0
        self.record_waiting_time = sys.maxsize
        self.record_distance = sys.maxsize
        self.record_time_usage = sys.maxsize
        self.record_loaders_waiting_time = sys.maxsize
        self.record_cost = sys.maxsize
        self.record_fuel = sys.maxsize

        save_amount = 10
        self.record_distances = [sys.maxsize] * save_amount
        self.record_distance_games = [0] * save_amount
        self.record_waiting_times = [sys.maxsize] * save_amount
        self.record_waiting_time_games = [0] * save_amount
        self.record_loaders_waiting_times = [sys.maxsize] * save_amount
        self.record_loaders_waiting_time_games = [0] * save_amount
        self.record_time_usages = [sys.maxsize] * save_amount
        self.record_time_usage_games = [0] * save_amount
        self.record_costs = [sys.maxsize] * save_amount
        self.record_cost_games = [sys.maxsize] * save_amount
        self.record_fuels = [sys.maxsize] * save_amount
        self.record_fuel_games = [sys.maxsize] * save_amount
        self.fuels = {}
        self.dumpers_required = []

    def change_num_dumpers(self, num):
        self.dumpers = self.all_dumpers[: len(self.dumpers) + num]
        self.reset_records()

    def set_alpha_in_agents(self, num, type):
        done = False
        if type in ["nodes", None]:
            done = True
            for agent in self.node_agents:
                agent.trainer.set_alpha(num)

        if type in ["plans", None]:
            done = True
            for agent_plan in self.plan_agents:
                agent_plan.trainer.set_alpha(num)

        if not done:
            print(f'\n\nType: "{type}" will not work ...\n\n')

    def re_initialize_networks(self, type=None):
        """
        Reinitialize policy of either or both of node - and plan agents

        Args:
            type [None, 'nodes', 'plans']:
                what type the probability should correspond to.
                None: both
        """

        done = False
        if type in ["nodes", None]:
            done = True
            for agent in self.node_agents:
                agent.initialize_network()

        if type in ["plans", None]:
            done = True
            for agent_plan in self.plan_agents:
                agent_plan.initialize_network()

        if not done:
            print(f'\n\nType: "{type}" will not work ...\n\n')

    def set_random_choice_prob(self, prob, type=None):
        """
        Set the random choice probability of either or both of node - and plan agents

        Args:
            prob (int):
                probability of random choice
            type [None, 'nodes', 'plans']:
                what type the probability should correspond to.
                None: both
        """

        done = False
        if type in ["nodes", None]:
            done = True
            for agent in self.node_agents:
                agent.set_random_choice_prob(prob)

        if type in ["plans", None]:
            done = True
            for agent_plan in self.plan_agents:
                agent_plan.set_random_choice_prob(prob)

        if type in ["coffee", None]:
            done = True
            self.coffee_break_agent.set_random_choice_prob(prob)

        if not done:
            print(f'\n\nType: "{type}" will not work ...\n\n')

    def set_exploration_num(self, num, times=1, exploration_break=5, type=None):
        """
        Set the exploration num of either or both of node - and plan agents

        Args:
            num (int):
                exploration number
            type [None, 'nodes', 'plans']:
                what type the exploration num should correspond to.
                None: both
        """

        done = False
        if type in ["nodes", None]:
            done = True
            for agent in self.node_agents:
                agent.set_exploration_num(num, times, exploration_break)

        if type in ["plans", None]:
            done = True
            for agent_plan in self.plan_agents:
                agent_plan.set_exploration_num(num, times, exploration_break)

        if type in ["coffee", None]:
            done = True
            self.coffee_break_agent.set_exploration_num(num, times, exploration_break)

        if not done:
            print(f'\n\nType: "{type}" will not work ...\n\n')

    def get_best_games(self, best=3, criterions_count=3, rating_system="2"):
        # Write out the best games
        # TODO: maybe mark record holder

        importance = {}
        min_fuel, min_time = helper.print_minimum_fuel_and_time(
            self.dumper_best_speed,
            self.dumper_best_capacity,
            self.loading_nodes,
            self.dumping_nodes_dict,
            self.parking_nodes,
            is_printing=False,
        )

        # Can't find more than the number of runs
        best = min(best, np.sum(np.array(self.record_distance_games) != 0))
        record_criterium = -1

        # FUEL
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_fuel = np.array(self.record_fuel_games)[
                np.argsort(self.record_fuels)[:best]
            ]
            for idx, game in enumerate(importance_fuel):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # COST
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_cost = np.array(self.record_cost_games)[
                np.argsort(self.record_costs)[:best]
            ]
            for idx, game in enumerate(importance_cost):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # TIME USAGE
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_time_usage = np.array(self.record_time_usage_games)[
                np.argsort(self.record_time_usages)[:best]
            ]
            for idx, game in enumerate(importance_time_usage):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # DISTANCE
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_distance = np.array(self.record_distance_games)[
                np.argsort(self.record_distances)[:best]
            ]
            for idx, game in enumerate(importance_distance):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # LOADERS WAITING TIME
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_loaders_waiting_time = np.array(
                self.record_loaders_waiting_time_games
            )[np.argsort(self.record_loaders_waiting_times)[:best]]
            for idx, game in enumerate(importance_loaders_waiting_time):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # WAITING TIME
        record_criterium += 1
        if record_criterium < criterions_count:
            importance_waiting_time = np.array(self.record_waiting_time_games)[
                np.argsort(self.record_waiting_times)[:best]
            ]
            for idx, game in enumerate(importance_waiting_time):
                if not game in importance:
                    importance[game] = "0" * criterions_count

                importance[game] = (
                    importance[game][:record_criterium]
                    + f"{idx+1}"
                    + importance[game][(record_criterium + 1) :]
                )

        # def sort_dict(d):
        # return sorted(d.items(), key=lambda x: (x[1].count('1'), x[1].count('2'), x[1].count('3'), -list(d.keys()).index(x[0])), reverse=True)

        def sort_dict(d):
            return sorted(
                d.items(),
                key=lambda x: (
                    -x[1].count("1"),
                    -x[1].count("2"),
                    -x[1].count("3"),
                    -x[1].count("4"),
                    -x[1].count("5"),
                    -x[1].count("6"),
                    -x[1].count("7"),
                    -x[1].count("8"),
                    -x[1].count("9"),
                    list(d.keys()).index(x[0]),
                ),
            )

        sorted_importance = dict(sort_dict(importance))
        ordered_list_keys = list(sorted_importance.keys())

        # embed(header='here11')

        output = [0] * len(ordered_list_keys)
        infile = open("outputs/output.txt", "r")
        another_ranking = []
        another_ranking_strings = []
        for idx, line in enumerate(infile):
            game_num = idx + 1
            if game_num in importance:
                measures = line.split(",")[1::2][:-1]
                fuel_percent = min_fuel / float(measures[0].split()[-1]) * 100
                time_percent = min_time / int(measures[2].split()[-1]) * 100
                total_percent = 4 / 5 * fuel_percent + 1 / 5 * time_percent

                part_string = f"(Sum:{total_percent:3.1f}%, F:{fuel_percent:3.1f}%, T:{time_percent:3.1f}%, Rating: {importance[game_num]}, Game: {game_num:3.0f}) | "
                for measure in measures:
                    part_string += measure.strip() + " | "

                another_ranking.append(total_percent)
                another_ranking_strings.append(part_string)
                output[ordered_list_keys.index(game_num)] = part_string

        infile.close()

        outfile = open(
            f'models/{self.map_identifier}/ID{self.map_settings["id"]}#dumper{len(self.dumpers)}.txt',
            "w",
        )
        outfile.write(f"Last reset, Game: {self.last_reset_n_games}\n")
        outfile.write(f"Current, Game: {self.n_games}\n\n")
        try:
            outfile.write("\n".join(output))
        except:
            embed(header="try1")
        outfile.close()

        another_ranking_sorted_strings = list(
            np.array(another_ranking_strings)[np.argsort(another_ranking)[::-1]]
        )
        outfile2 = open(
            f'models/{self.map_identifier}/ID{self.map_settings["id"]}#dumper{len(self.dumpers)}_V2.txt',
            "w",
        )
        outfile2.write(f"Last reset, Game: {self.last_reset_n_games}\n")
        outfile2.write(f"Current, Game: {self.n_games}\n\n")
        try:
            outfile2.write("\n".join(another_ranking_sorted_strings))
        except:
            embed(header="try2")
        outfile2.close()

        another_ranking_sorted_strings = list(
            np.array(another_ranking_strings)[np.argsort(another_ranking)[::-1]]
        )

        # Finding the best rankings weighted fuel and time as well (percent)
        infile = open("outputs/output.txt", "r")
        scores = [0] * 10
        game_nums = [-1] * 10

        for idx, line in enumerate(infile):
            game_num = idx + 1

            measures = line.split(",")[1::2][:-1]
            fuel_percent = min_fuel / float(measures[0].split()[-1]) * 100
            time_percent = min_time / int(measures[2].split()[-1]) * 100
            total_percent = 4 / 5 * fuel_percent + 1 / 5 * time_percent

            if total_percent > min(scores):
                idx_of_change = scores.index(min(scores))
                scores[idx_of_change] = total_percent
                game_nums[idx_of_change] = game_num
        infile.close()

        outfile3 = open(
            f'models/{self.map_identifier}/ID{self.map_settings["id"]}#dumper{len(self.dumpers)}_V3.txt',
            "w",
        )
        sorting = np.argsort(scores)[::-1]
        scores = np.array(scores)[sorting]
        game_nums = np.array(game_nums)[sorting]

        for score, game in zip(scores, game_nums):
            outfile3.write(f"Game:{game:5.0f}, score: {score:3.1f}\n")

        outfile3.close()

    def bruces_algorithm(self, random_search=False):
        """
        A way to find the shortest paths between all pairs of nodes in an edge-weighted graph

        :param random_search (bool):
            NOTIMPLEMENTED
        """

        room_for_improvement_counter = 0

        for start_node, start_node_id in zip(self.nodes, self.node_ids):
            remaining_node_ids = self.node_ids.copy()
            remaining_nodes = self.nodes.copy()

            remaining_node_ids.remove(start_node_id)
            remaining_nodes.remove(start_node)

            # Check if neighboor have better shortest distance
            for edge in start_node.get_edges():
                edge_distance = edge.get_distance()
                to_node = edge.get_to_node()

                for final_node, final_node_id in zip(
                    remaining_nodes, remaining_node_ids
                ):
                    # Finding the current shortest distance to final node
                    shortest_distance = (
                        start_node.get_shortest_observed_distance_to_node2(
                            final_node_id
                        )
                    )

                    # Check if better distance from end -> start, and update
                    shortest_distance_from_final_node = (
                        final_node.get_shortest_observed_distance_to_node2(
                            start_node_id
                        )
                    )

                    if shortest_distance_from_final_node < shortest_distance:
                        start_node.add_shortest_observed_distance_to_special_node2(
                            final_node_id, shortest_distance_from_final_node
                        )
                        shortest_distance = shortest_distance_from_final_node
                        room_for_improvement_counter += 1

                    shortest_distance_from_to_node = (
                        to_node.get_shortest_observed_distance_to_node2(final_node_id)
                    )
                    test_shortest_distance = (
                        shortest_distance_from_to_node + edge_distance
                    )
                    if test_shortest_distance < shortest_distance:
                        start_node.add_shortest_observed_distance_to_special_node2(
                            final_node_id, test_shortest_distance
                        )

                        room_for_improvement_counter += 1

        return room_for_improvement_counter

    def bas(self, is_printing=True):
        """
        TODO: need to have different algorithms if it is a directed graph -> only change,
            cannot set minimal distance to be the same for back-forth way
        """

        start_time = time.time()
        # Get the dictionaries more without calling node class
        nodes_edges = {}
        shortest_observed = {}
        edge_count_node_id = {}
        # embed(header='here111')

        # Restart the shortest distances (if load_model with shortest distance, but wrong)
        shortest_distances = {}
        for node_id in self.node_ids:
            shortest_distances[node_id] = sys.maxsize

        for node in self.nodes:
            node.initialize_shortest_observed_distance_to_node(
                shortest_distances.copy()
            )

        for node, node_id in zip(self.nodes, self.node_ids):
            shortest_observed[node_id] = node.shortest_observed_distance_to_special_node
            nodes_edges[node_id] = []
            edges = node.get_edges()

            if len(edges) in edge_count_node_id:
                edge_count_node_id[len(edges)].append(node_id)
            else:
                edge_count_node_id[len(edges)] = [node_id]

            for edge in edges:
                nodes_edges[node_id].append(
                    [f"{edge.get_to_node().get_id()}", edge.get_distance()]
                )

        if is_printing:
            print("Precise search:")
        more = True
        iter = 1
        while more:
            part_start_time = time.time()
            more, shortest_observed, nodes_edges = self.bruces_algorithm2(
                self.node_ids, shortest_observed, nodes_edges
            )

            if is_printing:
                print(
                    f"{iter:3.0f}) Part time: {time.time() - part_start_time: 3.1f}s, Fixed {more} best choices"
                )
            iter += 1

        if is_printing:
            print(f"Found optimal way in {time.time() - start_time: .1f}s")

        for node in tqdm(self.nodes, disable=not is_printing):
            node.set_best_predictions()

        self.coffee_break_agent.create_time_converter_constant()

        # Calculating benchmarks benchmarks for the map
        self.calculate_benchmarks()

        # self.add_best_choice_to_node_agents()

    def calculate_benchmarks(self):
        min_fuel, min_time = helper.print_minimum_fuel_and_time(
            self.dumper_best_speed,
            self.dumper_best_capacity,
            self.loading_nodes,
            self.dumping_nodes_dict,
            self.parking_nodes,
            is_printing=False,
        )
        self.fuel_benchmark2 = round(min_fuel)
        self.time_benchmark2 = round(min_time)

        self.benchmark2 = [self.fuel_benchmark2, self.time_benchmark2]

        fuel_dumpers = []
        time_dumpers = []
        num_dumpers = []
        score_dumpers = []
        for _num_dumpers in range(len(self.loading_nodes), len(self.dumpers) + 1):
            num_dumpers.append(_num_dumpers)
            _fuel, _time = helper.print_reference_fuel_and_time(
                self.dumpers,
                self.loading_nodes,
                self.dumping_nodes,
                self.dumping_nodes_dict,
                self.parking_nodes,
                num_dumpers=_num_dumpers,
                is_printing=False,
            )

            score_dumpers.append(self.calculate_scoring_criteria(_fuel, _time))
            fuel_dumpers.append(round(_fuel))
            time_dumpers.append(round(_time))

        self.benchmark1_a = []  # highest percentage
        indx_max_score = score_dumpers.index(max(score_dumpers))

        self.benchmark1_a.append(num_dumpers[indx_max_score])
        self.benchmark1_a.append(fuel_dumpers[indx_max_score])
        self.benchmark1_a.append(time_dumpers[indx_max_score])
        self.benchmark1_a.append(score_dumpers[indx_max_score])

        self.benchmark1_b = []  # lowest fuel
        indx_min_fuel = fuel_dumpers.index(min(fuel_dumpers))
        self.benchmark1_b.append(num_dumpers[indx_min_fuel])
        self.benchmark1_b.append(fuel_dumpers[indx_min_fuel])
        self.benchmark1_b.append(time_dumpers[indx_min_fuel])
        self.benchmark1_b.append(score_dumpers[indx_min_fuel])

    def calculate_scoring_criteria(self, other_fuel, other_time):
        return round(
            (
                0.8 * (self.fuel_benchmark2 / other_fuel)
                + 0.2 * (self.time_benchmark2 / other_time)
            )
            * 100,
            1,
        )

    def bruces_algorithm2(
        self, node_ids, shortest_observed, nodes_edges, random_search=False
    ):
        """
        A way to find the shortest paths between all pairs of nodes in an edge-weighted graph

        :param random_search (bool):
            NOTIMPLEMENTED
        """

        room_for_improvement_counter = 0

        for idx, start_node_id in enumerate(node_ids):
            remaining_node_ids = node_ids.copy()
            remaining_node_ids.remove(start_node_id)

            # Check if neighboor have better shortest distance
            for final_node_id in remaining_node_ids:
                # Finding the current shortest distance to final node, (both ways)
                shortest_distance = shortest_observed[start_node_id][final_node_id]
                # shortest_distance_back = shortest_observed[final_node_id][start_node_id]

                for to_node_id, edge_distance in nodes_edges[start_node_id]:
                    # Find the shortest distance via edge.to_node
                    shortest_distance_from_to_node = shortest_observed[to_node_id][
                        final_node_id
                    ]
                    test_shortest_distance = (
                        shortest_distance_from_to_node + edge_distance
                    )

                    if test_shortest_distance < shortest_distance:
                        shortest_observed[start_node_id][
                            final_node_id
                        ] = test_shortest_distance
                        # TODO: directed graph - remove underneath
                        shortest_observed[final_node_id][
                            start_node_id
                        ] = test_shortest_distance
                        shortest_distance = test_shortest_distance
                        room_for_improvement_counter += 1

        return room_for_improvement_counter, shortest_observed, nodes_edges

    def add_best_choice_to_node_agents(self):
        """
        After running Bruces algorithm, add best memory to the node agents
        """

        pbar = tqdm(total=len(self.nodes))
        for start_node, start_node_id in zip(self.nodes, self.node_ids):
            remaining_nodes = self.nodes.copy()
            remaining_node_ids = self.node_ids.copy()

            remaining_node_ids.remove(start_node_id)
            remaining_nodes.remove(start_node)

            for final_node, final_node_id in zip(remaining_nodes, remaining_node_ids):
                state = np.zeros(len(self.nodes))
                state[final_node.get_id()] = 1
                prediction = np.zeros(len(start_node.get_edges()))

                shortest_distance = start_node.get_shortest_observed_distance_to_node2(
                    final_node_id
                )

                if True:
                    for idx, edge in enumerate(start_node.get_edges()):
                        edge_distance = edge.get_distance()
                        to_node = edge.get_to_node()

                        shortest_distance_from_to_node = (
                            to_node.get_shortest_observed_distance_to_node2(
                                final_node_id
                            )
                        )

                        next_shortest_distance = (
                            shortest_distance_from_to_node + edge_distance
                        )

                        # Saving the state, and add best prediction
                        if next_shortest_distance == shortest_distance:
                            # start_node.add_shortest_observed_distance_to_special_node2(

                            prediction = np.zeros(len(start_node.get_edges()))
                            prediction[idx] = 1

                            sample = (state, prediction, 10, state, False)

                            start_node.get_agent().add_route_to_round_memory(
                                sample, train=False
                            )
                            break
            pbar.update(1)

    def option_guide(self):
        """
        The option guide of the code, when running the code interactively
        """

        if self.last_game != self.n_games:
            if not (self.start_time is None):
                stop_time = timePackage.time()
                time_usage_simulations = stop_time - self.start_time
                print(
                    f"### Time used for {self.num_games} games: {time_usage_simulations : .2f} seconds ### \n\n"
                )

            print(constants.OPTIONS_MAIN + "\n\n" + constants.OPTIONS_OTHER)

        self.num_games = 0
        self.last_game = self.n_games

        answer = input("Answer: ")
        input_list = answer.lower().split()

        try:
            input_lower_first = input_list[0]
        except IndexError:
            input_lower_first = "hmmmmm_are_you_dumb"  # TODO: change

        if input_lower_first == "stop":
            plt.close()
            exit(0)
        elif input_lower_first == "embed":
            embed()
        elif input_lower_first == "random_prob":
            try:
                _random_prob = int(input_list[1])
                _type = None
                if len(input_list) > 2:
                    _type = input_list[2]

                self.set_random_choice_prob(_random_prob, _type)
            except ValueError:
                print(input_list[1] + " is not an int")
        elif input_lower_first == "plot_shower":
            if input_list[1] in self.possible_plots:
                self.plot_shower = input_list[1]
        elif input_lower_first in ["options", "o"]:
            print(constants.OPTIONS_MAIN + "\n\n" + constants.OPTIONS_OTHER)
        elif input_lower_first == "save_model":
            if len(input_list) > 1:
                filename = input_list[1]
                self.save_models(filename)
            else:
                print("ERROR: need to provide foldername")
        elif input_lower_first in ["load_model", "lm"]:
            self.load_models()

        elif input_lower_first in ["re_init_networks", "rin"]:
            _type = None
            if len(input_list) > 1:
                _type = input_list[1]

            self.re_initialize_networks(_type)
        elif input_lower_first == "exploration":
            try:
                _exploration_num = int(input_list[1])
                _times = 1
                _exploration_break = 5
                _type = None
                if len(input_list) > 2:
                    _times = int(input_list[2])
                if len(input_list) > 3:
                    _exploration_break = int(input_list[3])
                if len(input_list) > 4:
                    _type = input_list[4]

                self.set_exploration_num(
                    _exploration_num, _times, _exploration_break, _type
                )
            except ValueError:
                print(input_list[1] + "is not an int")
            except IndexError:
                print("Missed some Input")
        elif input_lower_first in ["print_information", "pf"]:
            self.print_information_next_round = True
            input_ = "1"
        elif input_lower_first == "max_time":
            try:
                self.MAX_TIME = int(input_list[1])
            except Exception as e:
                print(e)
                print("\n\nCould not set new MAX_TIME\n\n")
        elif input_lower_first == "dumper":
            try:
                if input_list[1].lower() == "fixed":
                    print("\n\nDumper position is set fixed\n\n")
                    self.dumper_position = "fixed"
                elif input_list[1].lower() == "random":
                    print("\n\nDumper position is set random\n\n")
                    self.dumper_position = "random"
                elif input_list[1].lower() == "parking":
                    print("\n\nDumper position is set to parking spot\n\n")
                    self.dumper_position = "parking"
            except Exception as e:
                print(e)
                print("Could not change the dumper position")
        elif input_lower_first in ["print_minimum_fuel_and_time", "pmfat"]:
            helper.print_minimum_fuel_and_time(
                self.dumper_best_speed,
                self.dumper_best_capacity,
                self.loading_nodes,
                self.dumping_nodes_dict,
                self.parking_nodes,
            )
        elif input_lower_first in ["print_reference_fuel_and_time", "prfat"]:
            try:
                min_num_dumpers = int(input_list[1])
            except:
                min_num_dumpers = len(self.dumpers)
            finally:
                print(f"\nReference measures:")
                for _num_dumpers in range(min_num_dumpers, len(self.dumpers) + 1):
                    try:
                        helper.print_reference_fuel_and_time(
                            self.dumpers,
                            self.loading_nodes,
                            self.dumping_nodes,
                            self.dumping_nodes_dict,
                            self.parking_nodes,
                            num_dumpers=_num_dumpers,
                        )
                    except:
                        _dummy = "haha"
                print()
        elif input_lower_first in ["print_statistics_overleaf", "pso"]:
            helper.print_statistics_overleaf(
                framework_a=self.framework_results_a,
                framework_b=self.framework_results_b,
                benchmark1_a=self.benchmark1_a,
                benchmark1_b=self.benchmark1_b,
                benchmark2=self.benchmark2,
            )
        elif input_lower_first in ["finished_training_node_agents", "ftna"]:
            if len(input_list) > 1:
                if input_list[1].lower() == "true":
                    self.finished_training_node_agents = True
                elif input_list[1].lower() == "false":
                    self.finished_training_node_agents = False
                else:
                    print(f"\n\nDo not understand: {input_list[1]}\n\n")
        elif input_lower_first in ["coffee_break", "cb"]:
            if len(input_list) > 1:
                if input_list[1].lower() == "true":
                    self.coffee_break_agent_active = True
                elif input_list[1].lower() == "false":
                    self.coffee_break_agent_active = False
                else:
                    print(f"\n\nDo not understand: {input_list[1]}\n\n")
        elif input_lower_first in ["parking"]:
            if len(input_list) > 1:
                if input_list[1].lower() == "true":
                    self.parking_agent_active = True
                elif input_list[1].lower() == "false":
                    self.parking_agent_active = False
                else:
                    print(f"\n\nDo not understand: {input_list[1]}\n\n")
        elif input_lower_first in ["parking_set_fuel_base", "psfb"]:
            if len(input_list) > 1:
                if input_list[1].lower() == "min":
                    try:
                        self.fixed_base_fuel = min(self.plot_fuel) * 2
                        self.fixed_base_cost = min(self.plot_cost) * 2
                    except:
                        print(f"\n\nNo fuel value found \n\n")
                else:
                    try:
                        self.fixed_base_fuel = int(input_list[1])
                        self.fixed_base_cost = int(input_list[1])
                    except:
                        print(f"\n\nDo not understand: {input_list[1]} \n\n")
            else:
                print(f"\n\nYou need to make a choice \n\n")
        elif input_lower_first in ["alpha", "a"]:
            try:
                _alpha = float(input_list[1])
                _type = None
                if len(input_list) > 2:
                    _type = input_list[2]

                self.set_alpha_in_agents(_alpha, _type)
            except ValueError:
                print(input_list[1] + "is not an float")
            except IndexError:
                print("Missed some Input")
        elif input_lower_first in ["change_dumpers", "cd"]:
            try:
                _num = int(input_list[1])
                self.change_num_dumpers(_num)
            except Exception as e:
                print(e)
                print("\n\nCould not add/ remove amount of dupers\n\n")
        elif input_lower_first in ["find_prediction", "fp"]:
            state = np.zeros(len(self.nodes))
            try:
                state[int(input_list[2])] = 1
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.nodes[int(input_list[1])].get_agent().model(state0)
                print(prediction)
            except:
                print(f"Failed")
        elif input_lower_first in ["clear_plot", "cp"]:
            try:
                self.n_games_plot_starter = int(input_list[1])
            except:
                self.n_games_plot_starter = self.n_games

        elif input_lower_first in ["bruces_algorithm", "ba"]:
            self.bruces_algorithm2()

        elif input_lower_first in ["bas"]:
            self.bas()
        elif input_lower_first in ["get_best_games", "gbg"]:
            try:
                self.get_best_games(int(input_list[1]), int(input_list[2]))
            except:
                try:
                    self.get_best_games(int(input_list[1]))
                except:
                    self.get_best_games()

        elif input_lower_first in ["node_choice", "nc"]:
            try:
                if input_list[1] in ["policy", "optimal"]:
                    self.node_choice = input_list[1]
                else:
                    print(f"Something wrong. node_choice: {self.node_choice}")
            except:
                print(f"Something wrong. node_choice: {self.node_choice}")

        elif input_lower_first in ["add_best_choice_to_node_agents", "abctna"]:
            self.add_best_choice_to_node_agents()

        elif input_lower_first in ["train_best_memory", "tbm"]:
            try:
                n = int(input_list[1])
            except:
                n = 10

            for i in tqdm(range(n)):
                for node_agent in self.node_agents:
                    node_agent.train_long_memory()

        elif input_lower_first in ["save_animation"]:
            try:
                name = input_list[1]
                try:
                    schedule = self.all_stuff_in_games[int(input_list[2])]
                except:
                    print(f"Goes with last schedule")
                    schedule = self.all_stuff_in_game

                helper.create_graph_visualization_advanced3(
                    self.nodes, self.edges, schedule=schedule, animation_name=name
                )
            except:
                print("Could not make animation")
        elif input_lower_first in ["save_map_image"]:
            try:
                name = input_list[1]
                helper.create_graph_visualization(
                    self.nodes, self.edges, image_name=name
                )
            except:
                print("Could not make an image of map")
        else:
            try:
                self.num_games = int(input_lower_first)
            except:
                print("You did nothing")
                self.num_games = 0


if __name__ == "__main__":
    # NODES, EDGES, LOADERS, DUMPERS, NODES_DICT = maps.map3(False)
    # tester = RouteOptimization(maps.map3_without_intersection_small_jobs)

    edge_changes = {
        "edge_0_1": 175,
        "edge_0_3": 10000,
        "edge_1_2": 75,
        "edge_1_4": 25,
        "edge_2_5": 10000,
        "edge_3_4": 10000,
        "edge_3_6": 10000,
        "edge_4_5": 10000,
        "edge_4_7": 25,
        "edge_5_8": 10000,
        "edge_6_7": 75,
        "edge_7_8": 175,
    }
    # Benchmark:
    # loaders_waiting = 0
    # dumpers_waiting = 0 -> before end game (probably dumpers should park now)
    # time: 190 * 5 + 50 + 30 = 1030

    map_settings_simple = {
        "num_dumpers": 4,
        "rows": 3,
        "cols": 3,
        "num_loading_nodes": 2,
        "num_dumping_nodes": 2,
        "pre_dumping_nodes": [2, 8],
        "pre_loading_nodes": [0, 6],
        "pre_parking_nodes": None,
        "mass_type": ["0"],
        "max_capacity": 200,
        "edge_changes": edge_changes,
        "print_connections": False,
        "seed": 7,
        "id": 0
        # 'seed': 7   # NORMAL
    }

    edge_changes = {
        "edge_0_1": 250,
        "edge_0_3": 10000,
        "edge_1_2": 150,
        "edge_1_4": 10000,
        "edge_2_5": 10000,
        "edge_3_4": 10000,
        "edge_3_6": 10000,
        "edge_4_5": 10000,
        "edge_4_7": 10000,
        "edge_5_8": 10000,
        "edge_6_7": 120,
        "edge_7_8": 80,
    }

    # Benchmark:
    # loaders_waiting = 0
    # dumpers_waiting = 0 -> before end game (probably dumpers should park now)
    # time: 190 * 5 + 50 + 30 = 1030

    map_settings_simple_new = {
        "num_dumpers": 4,
        "rows": 3,
        "cols": 3,
        "num_loading_nodes": 2,
        "loading_rates": [4, 2],
        "num_dumping_nodes": 2,
        "pre_dumping_nodes": [2, 8],
        "pre_loading_nodes": [0, 6],
        "pre_parking_nodes": [4],
        "mass_type": ["0"],
        "max_capacity": 200,
        "edge_changes": edge_changes,
        "print_connections": False,
        "seed": 7,
        "id": 0
        # 'seed': 7   # NORMAL
    }

    map_settings_V1 = {
        "num_dumpers": 25,
        "rows": 10,
        "cols": 10,
        "num_loading_nodes": 6,
        "loading_rates": [4] * 6,
        "num_dumping_nodes": 3,
        "pre_dumping_nodes": None,
        "pre_loading_nodes": None,
        "pre_parking_nodes": None,
        "mass_type": ["0"],
        "max_capacity": 400,
        "edge_changes": {},
        "print_connections": False,
        "seed": 1,  # map i
        "id": 0
        # 'seed': 7   # NORMAL
    }

    edge_changes = {
        "edge_0_1": 100,
        "edge_1_2": 100,
        "edge_0_3": 40,
        "edge_1_4": 40,
        "edge_2_5": 40,
        "edge_3_4": 40,
        "edge_3_5": 40,
    }

    map_settings_parking_agent = {
        "num_dumpers": 5,
        "rows": 2,
        "cols": 3,
        "num_loading_nodes": 1,
        "loading_rates": [4],
        "num_dumping_nodes": 1,
        "pre_dumping_nodes": [2],
        "pre_loading_nodes": [0],
        "pre_parking_nodes": [5],
        "mass_type": ["0"],
        "max_capacity": 600,
        "edge_changes": edge_changes,
        "print_connections": False,
        "seed": 7,
        "id": 0
        # 'seed': 7   # NORMAL
    }

    map_settings_abstract = {
        "num_dumpers": 4,
        "rows": 6,
        "cols": 6,
        "num_loading_nodes": 2,
        "loading_rates": [4] * 3,
        "num_dumping_nodes": 2,
        "pre_dumping_nodes": None,
        "pre_loading_nodes": None,
        "pre_parking_nodes": None,
        "mass_type": ["0"],
        "max_capacity": 400,
        "edge_changes": {},
        "print_connections": False,
        "seed": 7,
        "id": 5
        # 'seed': 7   # NORMAL
    }

    map_settings_method = {
        "num_dumpers": 4,
        "rows": 5,
        "cols": 5,
        "num_loading_nodes": 2,
        "loading_rates": [4] * 2,
        "num_dumping_nodes": 2,
        "pre_dumping_nodes": None,
        "pre_loading_nodes": None,
        "pre_parking_nodes": None,
        "mass_type": ["0"],
        "max_capacity": 400,
        "edge_changes": {},
        "print_connections": False,
        "seed": 7,
        "id": 5
        # 'seed': 7   # NORMAL
    }

    map_settings_V1["seed"] = 1

    # tester = RouteOptimization(maps.map_generator, map_settings_method)
    # tester = RouteOptimization(maps.map_generator, map_settings_simple_new)
    tester = RouteOptimization(maps.map_generator, map_settings_V1)

    values = {
        "save_model": True,
        "load_model": False,
        "filename": "auto21",
        "exploration": [20, 10, 10],
        "num_games": 360,
        "statistics_filename": "statistics_latex_format/scenario21",
    }

    with open(values["statistics_filename"] + ".txt", "w"):
        print()
    with open(values["statistics_filename"] + "_latex.txt", "w"):
        print()

    run_values = True
    print(f"Map {map_settings_V1['seed']}: Training")
    if run_values:
        tester.set_values_pre_simulation(values)
    tester.main(run_values)

    ################################################################
    ##  Try run more rounds at the same time and save statistics  ##
    ################################################################

    values["exploration"] = [5, 20, 20]
    values["num_games"] = 510
    # values['exploration'] = [5, 20, 20]
    # values['exploration'] = [10, 10, 20]
    # values['num_games'] = 510
    values["save_model"] = False
    values["load_model"] = True

    for seed_num in range(1, 21):
        print(f"\nMap {seed_num}:")

        map_settings_V1["seed"] = seed_num
        tester = RouteOptimization(maps.map_generator, map_settings_V1)

        tester.set_values_pre_simulation(values)
        tester.main(run_values)
