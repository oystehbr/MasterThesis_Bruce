import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython import display, embed
import road_data_manipulation as rdm
import pandas as pd
import time as timePackage
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt
from IPython import embed
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D, offset_copy
from tqdm import tqdm
import constants


class State:
    def __init__(self, state, final_move):
        self.state = state
        self.final_move = final_move

    def get_state(self):
        return self.state

    def get_prediction(self):
        return self.final_move


class Reward_Calculator:
    """
    When finished a route, the calculation of the rewards for each action
    will be done instead this class
    """

    # TODO: If reward is negativ, do not discount it

    def __init__(self, dumper, route, time_of_task, nodes, shortest_time_LD, longest_time_LD, longest_distance, worst_waiting_time_loading, worst_waiting_time_dumping):
        self.dumper = dumper
        self.route = route
        self.nodes = nodes
        self.shortest_time_LD = shortest_time_LD
        self.longest_time_LD = longest_time_LD
        self.longest_distance = longest_distance
        self.worst_waiting_time_loading = worst_waiting_time_loading
        self.worst_waiting_time_dumping = worst_waiting_time_dumping
        self.actual_route = route['actual_route']
        self.rewards = route['actual_reward']
        self.waiting_time = route['waiting_time'][0]
        self.time_of_task = time_of_task
        self.updated_rewards = []
        self.calculate_driving_time()
        self.calculate_driving_distance()

    def calculate_driving_time(self):
        self.driving_time = 0
        self.driving_time_list = [0]

        for idx in range(len(self.actual_route) - 1):
            distance = self.actual_route[idx].distance_to_node(
                self.actual_route[idx + 1])
            self.driving_time += distance/self.dumper.get_speed()
            self.driving_time_list.append(self.driving_time)

    def calculate_driving_distance(self):
        self.driving_distance = 0
        self.driving_distance_list = [0]

        for idx in range(len(self.actual_route) - 1):
            distance = self.actual_route[idx].distance_to_node(
                self.actual_route[idx + 1])
            self.driving_distance += distance
            self.driving_distance_list.append(self.driving_distance)

        self.driving_distance_list = np.array(self.driving_distance_list)

    def calculate_fuel_consumption(self):
        NotImplemented

    def calculate_rewards_according_to_time(self):
        if len(self.rewards) != 0:

            self.final_reward = self.rewards[-1]
            # discount if positive
            if self.rewards[-1] > 0:
                try:
                    self.final_reward /= (self.waiting_time +
                                          self.get_driving_time())  # 3 times worse to drive than standing still
                except RuntimeWarning:
                    embed(header='runtime')

            self.updated_rewards = np.array(self.rewards, dtype=np.float64)
            self.updated_rewards[:-1] += self.final_reward

            # TODO: only need to set the same number ^ above is case
            # dumping some and then all in next move. SPLIT ON EVERY DUMP/LOAD
            self.updated_rewards[-1] = self.final_reward
            self.updated_rewards = list(self.updated_rewards)

    # def calculate_rewards_according_to_time(self):
    #     if len(self.rewards) != 0:

    #         self.final_reward = self.rewards[-1]
    #         # discount if positive
    #         if self.rewards[-1] > 0:
    #             self.final_reward /= (self.waiting_time +
    #                                   self.get_driving_time())  # 3 times worse to drive than standing still

    #         self.updated_rewards = np.array(self.rewards, dtype=np.float64)
    #         self.updated_rewards[:-1] += self.final_reward

    #         # TODO: only need to set the same number ^ above is case
    #         # dumping some and then all in next move. SPLIT ON EVERY DUMP/LOAD
    #         self.updated_rewards[-1] = self.final_reward
    #         self.updated_rewards = list(self.updated_rewards)

    def get_driving_rewards_time(self):
        # TODO: CAN modify reward => if same node is visited more times -> do something.

        embed(header='not finished')
        # closer to final destination the more important
        frac = np.array(self.driving_time_list) / self.driving_time
        # TODO: need to find better function -> depend on the driving time

        return 10 * frac

    def get_driving_rewards_distance(self, indeces):
        """
        Alternative reward, will fail on switching edge_distances:
            Will think that the 'next best' route is going the shortest distance
            to another node, and then go back to current node.

        TODO: write this into the master's thesis, view insights of multiple ways of rewarding
            the node agents.
            It will probably be better if we have a static map: since we could do fake training and
            optimize in that way. But static isn't that cool

        'Normal' reward will not have this problem.
        """

        # TODO: CAN modify reward => if same node is visited more times -> do something.

        # closer to final destination the more important
        frac = 1 - (self.driving_distance - self.driving_distance_list)/10000
        # TODO: need to find better function -> depend on the driving time
        self.time_to_goal = self.driving_distance - self.driving_distance_list
        end_node = self.actual_route[-1]
        try:
            important_nodes = np.array(self.actual_route)[indeces]
        except IndexError:
            embed(header='indxxx')

        new_distance = 0
        new_distances = [0]
        for idx in indeces:
            new_distance += self.actual_route[idx].get_edge_to_node(
                self.actual_route[idx + 1]).get_distance()

            new_distances.append(new_distance)

        new_distances = np.array(new_distances)

        try:
            distances_to_goal = (new_distance - new_distances)[:-1]
        except TypeError:
            embed(header='typerror')

        shortest_observed_distances_to_goal = []
        for node, distance in zip(important_nodes, distances_to_goal):
            shortest_observed_distance_to_goal = node.add_and_get_shortest_observed_distance_to_special_node(
                end_node, distance)
            shortest_observed_distances_to_goal.append(
                shortest_observed_distance_to_goal)

        #######################################################################
        # Train parts of the route to be good on other than just the end node #
        #######################################################################
        distances_between_nodes = abs(np.diff(distances_to_goal))
        for i, part_end_node in enumerate(important_nodes[1:]):
            # Constructing an input and prediction to train on
            fake_state = np.array(
                part_end_node == np.array(self.nodes), dtype=int)

            current_distance = sum(distances_between_nodes[:i+1])
            for j, part_start_node in enumerate(important_nodes[:i+1]):
                the_edge = part_start_node.get_edge_to_node(
                    important_nodes[j+1])
                fake_prediction = np.array(
                    np.array(part_start_node.get_edges()) == the_edge, dtype=int)

                shortest_observed_distance = part_start_node.add_and_get_shortest_observed_distance_to_special_node(
                    part_end_node, current_distance)

                fake_reward = (shortest_observed_distance /
                               current_distance) * 10

                # fake_alternative_reward
                next_distance = current_distance - distances_between_nodes[j]
                if j < i:
                    part_next_node = important_nodes[j+1]
                    shortest_observed_distance_next = part_next_node.add_and_get_shortest_observed_distance_to_special_node(
                        part_end_node, next_distance)
                    fake_alternative_reward = (
                        (shortest_observed_distance - distances_between_nodes[j]) / shortest_observed_distance_next) * 10
                    # More nodes to pass
                else:
                    # Directly
                    fake_alternative_reward = fake_reward

                # if fake_reward == 10 or fake_alternative_reward == 10:
                #     if fake_reward != fake_alternative_reward:
                #         embed(header='hereee1')

                sample = (
                    fake_state,
                    fake_prediction,
                    fake_reward,
                    fake_state,  # not used
                    False  # not used
                )

                # state = np.zeros(25)
                # state[12] = 1

                # if abs(fake_alternative_reward - 1) < 0.1:
                #     if np.sum(state == fake_state) == len(state):
                #         if f'{part_start_node}' == 'L4':
                #             embed(header='check1')

                part_start_node.get_agent().add_route_to_round_memory(sample)
                current_distance = next_distance

        # TODO: train the agent

        # air_distance_to_goal = []
        # final_x = int(end_node.get_coordinates()['x'])
        # final_y = int(end_node.get_coordinates()['y'])
        # for node in important_nodes:
        #     curr_x = int(node.get_coordinates()['x'])
        #     curr_y = int(node.get_coordinates()['y'])

        #     air_distance = np.sqrt((curr_x - final_x) **
        #                            2 + (curr_y - final_y)**2)
        #     air_distance_to_goal.append(air_distance)

        # if abs(air_distance_to_goal[-1]) < 0.001:
        #     air_distance_to_goal[-1] = important_nodes[-1].get_edge_to_node(
        #         end_node).get_distance()

        # rewards = (air_distance_to_goal / distances_to_goal) * 10

        shortest_observed_distances_to_goal = np.array(
            shortest_observed_distances_to_goal)

        alternative_rewards = list(
            (shortest_observed_distances_to_goal[:-1] - distances_between_nodes)/(shortest_observed_distances_to_goal[1:]) * 10)
        rewards = (shortest_observed_distances_to_goal /
                   distances_to_goal) * 10

        alternative_rewards.append(rewards[-1])

        return rewards

        embed(header='looolz')

        # end_node = self.actual_route[-1]

        # important_rewards = []
        # for time_to_goal, node in zip(important_time_to_goal, important_nodes):
        #     shortest_observed_time = node.add_and_get_shortest_observed_distance_to_special_node(
        #         end_node, time_to_goal)
        #     important_rewards(10 * shortest_observed_time / time_to_goal)

        # embed(header='loolzzz1')
        l = self.driving_distance - self.driving_distance_list
        return 10000 / l[:-1]

        return 10000/(self.driving_distance + 1) + self.driving_distance_list * 0

        return 10 * frac

    def get_plan_reward(self, coffee_break=0, larger_driving_time=None):
        # prev_route = self.dumper.get_prev_completed_route()

        # self.max_plan_reward = 100 + \
        #     20 * (self.route['time_since_last_used_node'][0] +
        #           prev_route['time_since_last_used_node'][0]
        #         )

        # TODO: if dumping is long time ago => not increase the rewad of that reason.
        # TODO: NO REASON TO GIVE MUCH REWARDS FOR DUMPING ON A DUMPING STATION THAT HASN'T BEEN DUMPED ON FOR A WHILE, or?
        # ^^ MAYBE it is, because you should finish all the jobs approximately at the same time.

        # TODO: 1000, could be changed such that shortest air-line, with no waiting time, gives 20
        # if self.dumper.get_on_my_way_to().class_id == 'parking':
        #     embed(header='here111')

        node_waiting_time = (self.route['time_since_last_used_node'][0] - 1)
        if not (larger_driving_time is None):
            self.driving_time = max(self.driving_time, larger_driving_time)

        # setting cap of node_waiting_time
        # self.max_plan_reward = 1000 + 5 * node_waiting_time

        # self.penalty1 = 4*self.waiting_time + 8*self.driving_time
        # TODO: coffee break adding to the waiting time for the

        # self.penalty1 = ((self.waiting_time + coffee_break)
        #                  * 5) + self.driving_time
        # self.penalty1 = self.waiting_time + coffee_break + 4 * self.driving_time

        # # TODO: testing out neew -> WORKING AMAZING
        # self.max_plan_reward = (
        #     50 * self.shortest_time_LD * 4) + 8 * node_waiting_time  # maybe 4*, same as self.driving_time
        # self.penalty1 = max(self.waiting_time +
        #                     coffee_break + 4 * self.driving_time, 1)

        # Does this happen - do not think so, if mapped to parking slots
        # if self.driving_time == 0:
        #     embed(header='what happend')
        #     return 100 / (coffee_break + 1)

        # self.max_plan_reward = (
        #     50 * self.shortest_time_LD * 8) + 3 * node_waiting_time  # maybe 4*, same as self.driving_time
        # self.penalty1 = max(self.waiting_time +
        #                     coffee_break + 8 * self.driving_time, 1)

        # TODO: should be something else for the dumping nodes

        # maybe 4*, same as self.driving_time
        self.max_plan_reward = 50 * self.shortest_time_LD * 5
        # need to cost more than node_waiting_time ... -> or else you will wait for a period just to let the loader be standing still.
        coffee_break *= 4

        if self.actual_route[-1].class_id == 'dumping':
            # operating time loader -> operating time dumper: 6 times less, maybe add that
            action_idx = self.route['state_plan'][0].final_move.index(1)
            more_mass_runs = self.dumper.knowledge_mass_differences
            # 1 diff = 1 min driving
            penalty_of_differences_mass = more_mass_runs[action_idx] * 60 * 8

            self.max_plan_reward = 50 * self.shortest_time_LD * 8
            self.penalty1 = max(
                penalty_of_differences_mass + 8 * self.driving_time, 1)
        elif self.actual_route[-1].class_id == 'loading':
            # LOADING AND PARKING
            self.penalty1 = max(self.waiting_time +
                                coffee_break - 3*node_waiting_time + 8*self.driving_time, 1)
        elif self.actual_route[-1].class_id == 'parking':
            self.penalty1 = max(self.waiting_time +
                                coffee_break - 3*node_waiting_time + 8*self.driving_time, 1)

        # TODO: maybe set cap of node_waiting_time
        return min(self.max_plan_reward/self.penalty1, 80)

        # node_waiting_time = max(10 * 60, node_waiting_time)
        # TODO: need to do something else for the dumping nodes
        fuel_loader = constants.FUEL_info['loader_fuel_idling'] * \
            node_waiting_time
        fuel_dumper = constants.FUEL_info['dumper_fuel_idling'] * \
            (self.waiting_time + coffee_break)
        fuel_driving = constants.FUEL_info['driving_fuel_per_second'] * \
            self.driving_time

        if self.driving_time == 0:
            return 100 / (coffee_break + 1)

        # FUEL thinkers
        # return 50 + fuel_loader - fuel_dumper - fuel_driving

        # TIME thinkers
        # time_reward = self.longest_time_LD + node_waiting_time - \
        #     (self.waiting_time + coffee_break) - self.driving_time

        # return min(time_reward/60, 80)  # in minuits

    def get_plan_reward_coffee_break(self, times):
        if 'coffee_break' in self.route:
            coffee_break_rewards = []

            for coffee_break_time in np.cumsum(np.diff(times)[::-1])[::-1]:
                coffee_break_rewards.append(
                    self.get_plan_reward(coffee_break_time))

            return coffee_break_rewards + [self.get_plan_reward(0)]
        else:
            return [self.get_plan_reward(0)]

    # def get_plan_reward(self):
    #     prev_route = self.dumper.get_prev_completed_route()

    #     # self.max_plan_reward = 100 + \
    #     #     20 * (self.route['time_since_last_used_node'][0] +
    #     #           prev_route['time_since_last_used_node'][0]
    #     #         )

    #     # TODO: if dumping is long time ago => not increase the rewad of that reason.

    #     self.max_plan_reward = 100 + 20 * \
    #         prev_route['time_since_last_used_node'][0]

    #     self.penalty1 = self.waiting_time + self.driving_time*3
    #     self.penalty2 = prev_route['waiting_time'][0] + \
    #         prev_route['driving_time'][0]*3

    #     return self.max_plan_reward/(self.penalty1 + self.penalty2)

    def get_driving_time(self):
        return self.driving_time

    def get_fuel_consumption(self):
        NotImplemented

    def get_updated_rewards(self):
        return self.updated_rewards


def start_plotter():
    plt.ion()


def plot(start, scores, mean_scores, labels=[''], ylabel='', several=False):
    display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel(ylabel)
    if several:
        for idx, (score, mean_score) in enumerate(zip(scores, mean_scores)):
            plt.plot(score, label=labels[idx])
            plt.plot(mean_score, label=labels[idx])
            plt.text(len(score)-1, score[-1], str(score[-1]))
            plt.text(len(mean_score)-1, mean_score[-1], str(mean_score[-1]))

        plt.legend()
    else:
        plt.plot(
            range(start, start + len(scores[start:])), scores[start:], label='current')
        plt.plot(
            range(start, start + len(scores[start:])), mean_scores[start:], label='mean')
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.legend()

    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(.1)


def check_if_circular(node_path, num=5):
    node_path = np.array(node_path)
    for i in range(2, num + 1):
        count = 0
        for j in range(i):
            sequence = node_path[j::i]
            if len(sequence) == np.sum(sequence == node_path[j]):
                count += 1
        if count == i:
            return True, i

    return False, -1


def get_informative_coords_df(coords, origin=None, is_route=False):
    """
    coords should be a nested list of [[Longitude, Latitude, Altitude values], [...]]


    """

    df = pd.DataFrame(coords)
    df.columns = ['Longitude', 'Latitude', 'Altitude']

    df, proj_info = rdm.add_meter_columns(df, origin)

    if is_route:
        df['x_diff'] = df['x'].diff()
        df['y_diff'] = df['y'].diff()
        df['distance_diff'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
        df['distance_cumulative'] = df.distance_diff.cumsum()

    return df


def get_informative_coords_df_xy(df, is_route=False):
    """
    coords should be a nested list of [[Longitude, Latitude, Altitude values], [...]]


    """

    if is_route:
        df['x_diff'] = df['x'].diff()
        df['y_diff'] = df['y'].diff()
        df['distance_diff'] = np.sqrt(df['x_diff']**2 + df['y_diff']**2)
        df['distance_cumulative'] = df.distance_diff.cumsum()

    return df


def get_time_formatted(time):
    positive_sign = time >= 0
    time = abs(time)

    HH = '00'
    MM = '00'
    SS = '00'
    hours = int(time // 3600)
    if hours > 0:
        HH = (2 - len(str(hours))) * '0' + str(hours)
        time -= (time // 3600) * 3600

    minutes = int(time // 60)
    if minutes > 0:
        MM = (2 - len(str(minutes))) * '0' + str(minutes)
        time -= minutes * 60

    time = int(time)
    SS = (2 - len(str(time))) * '0' + str(time)
    if positive_sign:
        return f'{HH}:{MM}:{SS}'
    else:
        return f'-{HH}:{MM}:{SS}'


def get_second_from_time_formatted(time_formatted):
    time_list = time_formatted.split(':')
    seconds = int(time_list[0])*3600 + \
        int(time_list[1])*60 + int(float(time_list[2]))
    return seconds


def use_last_states_and_rewards(node_path, n=20):
    lol = node_path.copy()
    node_path.reverse()
    indeces = np.array([node_path.index(x)
                        for x in list(dict.fromkeys(node_path))])

    node_path.reverse()
    indeces = len(node_path) - indeces - 1

    return indeces


def use_last_states_and_rewards1(node_path, n=20):
    lol = node_path.copy()
    node_path.reverse()
    indeces = np.array([node_path.index(x)
                        for x in list(dict.fromkeys(node_path))])

    node_path.reverse()
    indeces = len(node_path) - indeces - 1

    idx = 0
    iter = 0

    while idx + 1 < len(lol) and iter < 2 * len(lol):
        for i in range(1, len(lol) - idx):
            if lol[idx] == lol[-i]:
                lol = lol[:idx] + lol[-i:]
                # embed(header='here')
                break

        idx += 1
        iter += 1

    if iter > len(lol)*3/2:
        embed(header='is something worng')

    node_path.reverse()
    indeces1 = np.array([node_path.index(x) for x in lol])
    indeces1 = len(node_path) - indeces1 - 1
    node_path.reverse()

    return indeces1


# def calculate_distances_to_special_nodes(special_nodes, all_nodes):
#     for special_node in special_nodes:
#         a = time.time()
#         for node in all_nodes:
#             if not node.is_isolated():
#                 print(node.get_shortest_distance_to_special_node(special_node))

#         print(time.time() - a)


def estimate_fuel_consumption_of_a_route(fully_route, XGBmodel_name, X_feat, group_size=30, altitude_threshold=0.01):
    """
    Sample trip pieces from df.
    :param fully_route: Formatted dataframe with Ditio and Fuel data for one route
    :param XGBmodel_name: path to the file including the XGG model
    # TODO: can make a json file to include this information, given a xgb_model
    :param X_feat: the features the model is using
    :param group_size: the amount of datapoints that should create one statistics point and be sent into the model
    :altitude_threshold:The granularity in height difference in meters per bin used to define uphill/downhill movement

    :return: the fuel prediction of the given route
    """

    fully_route['part'] = fully_route // group_size

    parts_of_route = [fully_route[fully_route['part'] == i]
                      for i in range(0, int(fully_route.index.values[-1])//group_size)]

    # TODO: maybe do something if one part is 1 second or something
    # length_last_part = len(parts_of_route[-1])

    # compute statistics and predict fuel values for the route
    statdata = pd.DataFrame()
    for i in range(len(parts_of_route)):
        trip = parts_of_route[i]

        statdata_row = pd.Series(dtype='object')
        statdata_row['LengthDistance'] = trip['DistanceCumsum'].max(
        ) - trip['DistanceCumsum'].min()
        statdata_row['Quantity'] = trip.iloc[0].Quantity
        # Assign the type that is most frequent

        trip['AltitudeDiff'] = trip['Altitude'].diff().fillna(0)
        index_uphill = trip['AltitudeDiff'] > altitude_threshold
        index_downhill = trip['AltitudeDiff'] < -altitude_threshold
        statdata_row['AltitudeLoss'] = trip.loc[index_downhill,
                                                'AltitudeDiff'].sum()
        statdata_row['AltitudeGain'] = trip.loc[index_uphill,
                                                'AltitudeDiff'].sum()

        statdata_row['AltitudeDeltaEndStart'] = trip['Altitude'].iloc[-1] - \
            trip['Altitude'].iloc[0]

        statdata_row['AltitudeChange'] = statdata_row['AltitudeGain'] + \
            abs(statdata_row['AltitudeLoss'])

        statdata_row['AltitudeDeltaMaxMin'] = trip['Altitude'].max() - \
            trip['Altitude'].min()

        # TODO: this will be a litt bit wrong with same model?? - less time
        statdata_row['SpeedMean'] = statdata_row['LengthDistance'] / \
            len(trip)

        # Average inclinations
        average_downInclination_part = [0]
        average_upInclination_part = [0]

        for part_i in range(int(len(trip)/2)):
            part_incline = (
                trip['AltitudeDiff']/trip['DistanceCumsumDiff'])[part_i: (15 + part_i)]
            average_downInclination_part.append(
                part_incline[part_incline < 0].mean())
            average_upInclination_part.append(
                part_incline[part_incline > 0].mean())

        statdata_row['DownInclination'] = 0
        statdata_row['DownInclinationPart'] = 0
        if np.sum(index_downhill) > 0:
            statdata_row['DownInclination'] = statdata_row['AltitudeLoss'] / \
                trip.loc[index_downhill, 'DistanceCumsumDiff'].sum()
            statdata_row['DownInclinationPart'] = min(
                average_downInclination_part)

        statdata_row['UpInclination'] = 0
        statdata_row['UpInclinationPart'] = 0
        if np.sum(index_uphill) > 0:
            statdata_row['UpInclination'] = statdata_row['AltitudeGain'] / \
                trip.loc[index_uphill, 'DistanceCumsumDiff'].sum()
            statdata_row['UpInclinationPart'] = max(average_upInclination_part)

        trip['DiffCourse'] = trip.Course.diff().abs()
        statdata_row['SumRotation'] = trip['DiffCourse'][trip['DiffCourse'] < 200].sum()

        statdata_row['Sum_RotationXAltitudeDiffPos'] = (trip['DiffCourse'][(trip['DiffCourse'] < 200) & (index_uphill)]
                                                        * trip['AltitudeDiff'][(trip['DiffCourse'] < 200) & (index_uphill)]).sum()
        statdata_row['Sum_RotationXAltitudeDiffNeg'] = (trip['DiffCourse'][(trip['DiffCourse'] < 200) & (index_downhill)]
                                                        * trip['AltitudeDiff'][(trip['DiffCourse'] < 200) & (index_downhill)]).sum()

        altitudeLossPart = [0]
        altitudeGainPart = [0]
        for part_i in range(int(len(trip)/2)):
            part_altitude_change = trip['AltitudeDiff'][part_i: (15 + part_i)]
            altitudeLossPart.append(
                part_altitude_change[part_altitude_change < 0].sum())
            altitudeGainPart.append(
                part_altitude_change[part_altitude_change > 0].sum())

        statdata_row['AltitudeLossPart'] = min(altitudeLossPart)
        statdata_row['AltitudeGainPart'] = max(altitudeGainPart)

        statdata_row = statdata_row[X_feat]

        statdata = statdata.append(statdata_row, ignore_index=True)

    try:
        model_xgb_loaded = pickle.load(open(XGBmodel_name,  'rb'))
    except Exception as e:
        print(f"#### ERROR1 #### {e}")
        try:
            print(os.getcwd() + '/' + XGBmodel_name)
            model_xgb_loaded = pickle.load(
                open(os.getcwd() + '/' + XGBmodel_name,  'rb'))
        except Exception as e:
            print(f"#### ERROR2 #### {e}")
            exit()

    predicted_fuel_parts = model_xgb_loaded.predict(statdata.values)

    # scale last prediction value according to its length (in sec) - if it isn't perfectly divided
    if (len(fully_route) % group_size) != 0:
        predicted_fuel_parts[-1] = predicted_fuel_parts[-1] * \
            ((len(fully_route) % group_size)/group_size)

    predicted_fuel_fully_route = sum(predicted_fuel_parts)

    return predicted_fuel_fully_route, parts_of_route, predicted_fuel_parts


def find_course(coords):

    x = coords['x_diff'].values
    y = coords['y_diff'].values
    vals = []
    for _x, _y in zip(x[1:], y[1:]):
        if _x > 0:
            if _y > 0:
                val = np.arctan(_y/_x) * 180 / np.pi

            else:
                val = 270 + 90 + (np.arctan(_y/_x)) * 180 / np.pi

        else:
            if _y > 0:
                val = 90 + 90 + np.arctan(_y/_x) * 180 / np.pi
            else:
                val = 180 + np.arctan(_y/_x) * 180 / np.pi

        vals.append(val)

    return vals


def create_path(path):
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(
                1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def create_graph_visualization(nodes, edges, image_name=None):
    # Checking if the edges are of equal edge-distance or not
    one_edge_value = True
    for edge_num in range(0, len(edges), 2):
        if edges[edge_num].get_distance() != edges[edge_num + 1].get_distance():
            one_edge_value = False
            break

    G = nx.DiGraph()

    for node in nodes:
        coords = node.get_coordinates()
        x = coords['x'].item()
        y = coords['y'].item()
        G.add_node(f'{node}', label=f'{node}', pos=(x, y))

    for idx, edge in enumerate(edges):
        if one_edge_value:
            if idx % 2 != 0:
                continue

        from_node = edge.get_from_node()
        to_node = edge.get_to_node()
        distance = edge.get_distance()
        G.add_edge(f'{from_node}', f'{to_node}', label=f'{distance:.0f}')

    # pos = nx.spring_layout(G, seed=7)
    # pos = nx.spectral_layout(G)
    pos = nx.get_node_attributes(G, "pos")
    # pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()

    node_colors = []
    node_labels = nx.get_node_attributes(G, "label")
    for label in node_labels.values():
        if label.startswith("L"):
            node_colors.append("#228B22")
        elif label.startswith("D"):
            node_colors.append("#ff0000")
        elif label.startswith("P"):
            node_colors.append("#007fff")
        else:
            node_colors.append("#87ceeb")

    # Draw the graph with labeled nodes and edges
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax)
    # nx.draw(G, pos, labels=node_labels, node_color=node_colors,
    #         node_size=500)
    # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

    curved_edges = [edge for edge in G.edges() if edge[::-1] in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    if one_edge_value:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=straight_edges, arrows=False)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)

    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
                           connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(G, "label")
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {
        edge: edge_weights[edge] for edge in straight_edges}
    my_draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

    # Show the plot
    if image_name:
        plt.savefig(f'maps/{image_name}.png')
        plt.close()
    else:
        plt.show()


def create_graph_visualization_advanced(nodes, edges):
    # Checking if the edges are of equal edge-distance or not
    one_edge_value = True
    for edge_num in range(0, len(edges), 2):
        if edges[edge_num].get_distance() != edges[edge_num + 1].get_distance():
            one_edge_value = False
            break

    G = nx.DiGraph()

    for node in nodes:
        coords = node.get_coordinates()
        x = coords['x'].item()
        y = coords['y'].item()
        G.add_node(f'{node}', label=f'{node}', pos=(x, y))

    for idx, edge in enumerate(edges):
        if one_edge_value:
            if idx % 2 != 0:
                continue

        from_node = edge.get_from_node()
        to_node = edge.get_to_node()
        distance = edge.get_distance()
        G.add_edge(f'{from_node}', f'{to_node}', label=f'{distance:.0f}')

    # pos = nx.spring_layout(G, seed=7)
    # pos = nx.spectral_layout(G)
    pos = nx.get_node_attributes(G, "pos")
    # pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots()

    node_colors = []
    node_labels = nx.get_node_attributes(G, "label")
    for label in node_labels.values():
        if label.startswith("L"):
            node_colors.append("green")
        elif label.startswith("D"):
            node_colors.append("blue")
        else:
            node_colors.append("yellow")
    red_node = 'N99'

    embed(header='yeah')
    counter = 0
    finished = 20
    plt.ion()
    while counter < finished:
        display.clear_output(wait=True)
        counter += 1

        for i, color in enumerate(node_colors):
            if list(node_labels.keys())[i] == red_node:
                if color == 'red':
                    node_colors[i] = 'yellow'
                else:
                    node_colors[i] = 'red'

        print(f'hei_{counter}')
        # Draw the graph with labeled nodes and edges
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax)
        # nx.draw(G, pos, labels=node_labels, node_color=node_colors,
        #         node_size=500)
        # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

        curved_edges = [edge for edge in G.edges() if edge[::-1] in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))

        if one_edge_value:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=straight_edges, arrows=False)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)

        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
                               connectionstyle=f'arc3, rad = {arc_rad}')

        edge_weights = nx.get_edge_attributes(G, "label")
        curved_edge_labels = {
            edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {
            edge: edge_weights[edge] for edge in straight_edges}
        my_draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)

        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

        # Show the plot
        # plt.draw()
        plt.show(block=False)
        plt.pause(0.2)

    plt.show()


def create_graph_visualization_advanced2(nodes, edges, animation_name='ani2'):
    # Checking if the edges are of equal edge-distance or not
    one_edge_value = True
    for edge_num in range(0, len(edges), 2):
        if edges[edge_num].get_distance() != edges[edge_num + 1].get_distance():
            one_edge_value = False
            break

    G = nx.DiGraph()

    for node in nodes:
        coords = node.get_coordinates()
        x = coords['x'].item()
        y = coords['y'].item()
        G.add_node(f'{node}', label=f'{node}', pos=(x, y))

    for idx, edge in enumerate(edges):
        if one_edge_value:
            if idx % 2 != 0:
                continue

        from_node = edge.get_from_node()
        to_node = edge.get_to_node()
        distance = edge.get_distance()
        G.add_edge(f'{from_node}', f'{to_node}', label=f'{distance:.0f}')

    # pos = nx.spring_layout(G, seed=7)
    # pos = nx.spectral_layout(G)
    pos = nx.get_node_attributes(G, "pos")
    # pos = nx.kamada_kawai_layout(G)

    node_colors = []
    node_labels = nx.get_node_attributes(G, "label")
    for label in node_labels.values():
        if label.startswith("L"):
            node_colors.append("#228B22")
        elif label.startswith("D"):
            node_colors.append("#ff0000")
        else:
            node_colors.append("#87ceeb")
    red_node = 'N99'

    fig, ax = plt.subplots(figsize=(12, 8))

    def update(num, node_colors, red_node, node_labels):
        for i, color in enumerate(node_colors):
            if list(node_labels.keys())[i] == red_node:
                if num % 2 == 0:
                    node_colors[i] = 'yellow'
                else:
                    node_colors[i] = 'red'

        # Draw the graph with labeled nodes and edges
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(G, pos, ax=ax)
        # nx.draw(G, pos, labels=node_labels, node_color=node_colors,
        #         node_size=500)
        # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

        curved_edges = [edge for edge in G.edges() if edge[::-1] in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))

        if one_edge_value:
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=straight_edges, arrows=False)
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)

        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
                               connectionstyle=f'arc3, rad = {arc_rad}')

        edge_weights = nx.get_edge_attributes(G, "label")
        curved_edge_labels = {
            edge: edge_weights[edge] for edge in curved_edges}
        straight_edge_labels = {
            edge: edge_weights[edge] for edge in straight_edges}
        my_draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)

        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

    ani = animation.FuncAnimation(fig, update, frames=range(10),
                                  fargs=(node_colors, red_node,
                                         node_labels),
                                  interval=1000)

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showmaximized()
    ani.save(f'{animation_name}.gif', writer='Pillow')


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    fig = ax.figure
    canvas = fig.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        # Convert window extent from pixels to inches
        # to avoid issues displaying at different dpi
        ex = fig.dpi_scale_trans.inverted().transform_bbox(ex)

        if orientation == 'horizontal':
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=ex.width, y=0)
        else:
            t = text.get_transform() + \
                offset_copy(Affine2D(), fig=fig, x=0, y=ex.height)


# pairwise_color = [
#     ['rosybrown', 'maroon'],  # BROWN OK
#     ['#ffccff', '#ff66ff'],  # PINK OK
#     ['peachpuff', 'darkorange'],  # ORANGE OK
#     ['#bfbfbf', '#808080'],  # GRAY OK
#     ['#b366ff', '#8000ff'],  # PURPLE OK
#     ['#b3ffb3', '#33ff33'],  # GREEN OK
#     ['lightgray', 'darkgray'],
#     ['lemonchiffon', 'darkgoldenrod'],
#     ['fuchsia', 'darkmagenta'],
# ]

pairwise_color = [
    ['rosybrown', 'maroon'],  # BROWN OK
    ['#ffccff', '#ff66ff'],  # PINK OK
    ['peachpuff', 'darkorange'],  # ORANGE OK
    ['#bfbfbf', '#808080'],  # GRAY OK
    ['#cc99ff', '#8000ff'],  # PURPLE OK
    ['#b3ffb3', '#33ff33'],  # GREEN OK
    ['lightgray', 'darkgray'],
    ['lemonchiffon', 'darkgoldenrod'],
    ['fuchsia', 'darkmagenta'],
]

pairwise_color_explain = ['brown', 'pink', 'orange', 'gray', 'purple', 'green']


def create_graph_visualization_advanced3(nodes, edges, schedule=None, animation_name='ani_test', frames=None):
    # TODO: improvements: TEXT for how much mass should be moved.
    # TODO: improvements: remove the color of green and red, if it is finished used.
    # TODO: improvements: split the node into several colors, if overlapping dumpers.

    # Checking if the edges are of equal edge-distance or not
    one_edge_value = True
    for edge_num in range(0, len(edges), 2):
        if edges[edge_num].get_distance() != edges[edge_num + 1].get_distance():
            one_edge_value = False
            break

    G = nx.DiGraph()

    max_x = 0
    max_y = 0
    for node in nodes:
        coords = node.get_coordinates()
        x = coords['x'].item()
        y = coords['y'].item()

        if x > max_x:
            max_x = x

        if y > max_y:
            max_y = y

        G.add_node(f'{node}', label=f'{node}', pos=(x, y))

    for idx, edge in enumerate(edges):
        if one_edge_value:
            if idx % 2 != 0:
                continue

        from_node = edge.get_from_node()
        to_node = edge.get_to_node()
        distance = edge.get_distance()
        G.add_edge(f'{from_node}', f'{to_node}', label=f'{distance:.0f}')

    # pos = nx.spring_layout(G, seed=7)
    # pos = nx.spectral_layout(G)
    pos = nx.get_node_attributes(G, "pos")
    # pos = nx.kamada_kawai_layout(G)

    node_colors = []
    node_labels = nx.get_node_attributes(G, "label")
    for label in node_labels.values():
        if label.startswith("L"):
            node_colors.append("#228B22")
        elif label.startswith("D"):
            node_colors.append("#ff0000")
        elif label.startswith("P"):
            node_colors.append("#007fff")
        else:
            node_colors.append("#87ceeb")

    start_colors = node_colors.copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    if not frames is None:
        schedule_ = {}
        for i, key in enumerate(schedule):
            if i > frames:
                print(key)
                break
            schedule_[key] = schedule[key].copy()
        schedule = schedule_

    time_slots = list(schedule.keys())
    node_labels_keys = list(node_labels.keys())
    node_situation = {}
    for i, key in enumerate(node_labels_keys):
        node_situation[key] = [node_colors[i]]

    # Draw the graph with labeled nodes and edges
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax)
    # nx.draw(G, pos, labels=node_labels, node_color=node_colors,
    #         node_size=500)
    # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

    curved_edges = [edge for edge in G.edges() if edge[::-1] in G.edges()]
    straight_edges = list(set(G.edges()) - set(curved_edges))

    if one_edge_value:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=straight_edges, arrows=False)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)

    arc_rad = 0.25
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
                           connectionstyle=f'arc3, rad = {arc_rad}')

    edge_weights = nx.get_edge_attributes(G, "label")
    curved_edge_labels = {
        edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {
        edge: edge_weights[edge] for edge in straight_edges}
    my_draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)

    nx.draw_networkx_edge_labels(
        G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

    def init():
        pass

    pbar = tqdm(total=len(time_slots))
    dumper_on_the_way_to = {}

    def update(num, start_colors, node_colors, node_labels_keys, schedule, time_slots, situation):
        if num > 0:
            # +1 cause title 'Plans: \n'
            del ax.texts[-(len(dumper_on_the_way_to) + 1):]

        for dumper in schedule[time_slots[num]]:
            values = schedule[time_slots[num]][dumper]
            prev_node = values[0]
            next_node = values[1]
            amount_of_mass = values[2]
            end_node = values[3]

            # remove from previous, and add to new
            # idx = node_labels_keys.index(f'{next_node}')
            # different with or without mass
            if amount_of_mass > 0:
                color = pairwise_color[dumper.dumper_num][1]
            else:
                color = pairwise_color[dumper.dumper_num][0]
            situation[f'{next_node}'].insert(0, color)

            dumper_on_the_way_to[pairwise_color_explain[dumper.dumper_num]] = [
                end_node, color]

            if prev_node:
                for color in pairwise_color[dumper.dumper_num]:
                    if color in situation[f'{prev_node}']:
                        situation[f'{prev_node}'].remove(color)

        for i, node_key in enumerate(situation):
            node_colors[i] = situation[node_key][0]

        # # Draw the graph with labeled nodes and edges
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=500)
        # nx.draw_networkx_labels(G, pos, ax=ax)
        # # nx.draw(G, pos, labels=node_labels, node_color=node_colors,
        # #         node_size=500)
        # # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->")

        # curved_edges = [edge for edge in G.edges() if edge[::-1] in G.edges()]
        # straight_edges = list(set(G.edges()) - set(curved_edges))

        # if one_edge_value:
        #     nx.draw_networkx_edges(
        #         G, pos, ax=ax, edgelist=straight_edges, arrows=False)
        # else:
        #     nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)

        # arc_rad = 0.25
        # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges,
        #                        connectionstyle=f'arc3, rad = {arc_rad}')

        # edge_weights = nx.get_edge_attributes(G, "label")
        # curved_edge_labels = {
        #     edge: edge_weights[edge] for edge in curved_edges}
        # straight_edge_labels = {
        #     edge: edge_weights[edge] for edge in straight_edges}
        # my_draw_networkx_edge_labels(
        #     G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)

        # nx.draw_networkx_edge_labels(
        #     G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

        try:
            final_happening = schedule[time_slots[num]
                                       ][list(schedule[time_slots[num]].keys())[-1]]
        except IndexError:
            final_happening = schedule[time_slots[num+1]
                                       ][list(schedule[time_slots[num+1]].keys())[0]]

        loading_text = 'Remaining mass to be loaded (ton): '
        for loading_node in final_happening[4]:
            loading_text += f'{loading_node:>4}: {final_happening[4][loading_node]:4}, '
        loading_text = loading_text[:-2]

        dumping_text = f'Remaining mass to be dumped (ton): '
        for dumping_node in final_happening[5]:
            dumping_text += f'{dumping_node:>4}: {final_happening[5][dumping_node]:4}, '
        dumping_text = dumping_text[:-2]

        title_font = {'family': 'monospace', 'weight': 'bold'}
        ax.set_title(
            f'Frame {num}   Time: {get_time_formatted(time_slots[num])}' +
            f'\n{loading_text}\n{dumping_text}', fontdict=title_font)

        text = ''
        colors = ['black']
        for i, key_color in enumerate(dumper_on_the_way_to):
            text += f'{key_color.upper()}: {dumper_on_the_way_to[key_color][0]}'
            colors.append(dumper_on_the_way_to[key_color][1])

            if i < len(dumper_on_the_way_to) - 1:
                text += ' | '

        rainbow_text(-0.9 * (max_x + 1)/10, -1.5 * (max_y + 1)/10, ['Plans: \n'] +
                     text.split('|'), colors, ax=ax)

        # text = 'Plans: \n '
        # for i, key_color in enumerate(dumper_on_the_way_to):
        #     text += f'{key_color.upper()}: {dumper_on_the_way_to[key_color]} | '
        # fig.text(.13, 0.07, text, ha='left')
        pbar.update(1)

    ani = animation.FuncAnimation(fig, update, frames=range(len(time_slots)),
                                  fargs=(start_colors, node_colors,
                                         node_labels_keys, schedule, time_slots, node_situation),
                                  init_func=init, interval=1000)

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showmaximized()
    _ = ani.save(f'animations/{animation_name}.gif', writer='pillow')
    del _
    plt.close()


def calculate_fuel_and_cost_of_plan(dumpers, parking_nodes, distance, driving_time, idling_dumpers, idling_loaders, time, dumper_position, dumpers_parking_time, loading_nodes, n_games):
    """
    Fuel * price
    Dumper while not parking * salary
    Loaders while jobbing (first task to last task)
    """

    ################################################################
    # CALCULATE FUEL VALUES
    ################################################################

    fuel_distance = driving_time * \
        constants.FUEL_info['driving_fuel_per_second']
    fuel_dumpers = idling_dumpers * constants.FUEL_info['dumper_fuel_idling']
    fuel_loaders = idling_loaders * constants.FUEL_info['loader_fuel_idling']

    total_fuel = fuel_distance + fuel_loaders + fuel_dumpers

    ################################################################
    # FIND HOW LONG THE LOADERS WAS WORKING:
    ################################################################
    total_working_time_loaders = 0
    for loading_node in loading_nodes:
        times = list(loading_node.get_time_scheduling().keys())
        start_time = get_second_from_time_formatted(times[0])
        end_time = get_second_from_time_formatted(times[-1])

        total_working_time_loaders += (end_time - start_time)

    ################################################################
    # CALCULATE COST
    ################################################################
    cost_fuel = total_fuel * constants.COST_info['fuel_price']
    cost_dumpers = (driving_time + idling_dumpers) * \
        constants.COST_info['salary_dumper']
    cost_loaders = total_working_time_loaders * \
        constants.COST_info['salary_loader']

    total_cost = cost_fuel + cost_dumpers + cost_loaders

    return total_fuel, total_cost


def train_minimum_reward(fuels, min_rewards, learning_rate):
    # Set min_reward to zero, and it will do as before
    # Oke dersom det ikke forandrer seg så mye

    return min_rewards[np.argmin(fuels)] + np.random.uniform(), learning_rate

    embed(header='test')
    if len(fuels) > 1:
        # if len(fuels) > 100:
        #     embed(header='inside training')

        # if (fuels[-1] - np.min(fuels)) / np.min(fuels) > 0.02:
        if (fuels[-1] - fuels[-2]) / fuels[-2] > 0:
            # if np.diff(min_rewards[-2:]) > 0:
            #     learning_rate = learning_rate / 2

            return min_rewards[-1] - 1 * np.random.uniform(), learning_rate
            # return min_rewards[-1] - learning_rate * np.random.uniform(), learning_rate

        else:
            # if np.diff(min_rewards[-2:]) < 0:
            #     learning_rate = learning_rate / 2

            return min_rewards[-1] + 1 * np.random.uniform(), learning_rate
            return min_rewards[-1] + learning_rate * np.random.uniform(), learning_rate

    return min_rewards[-1], learning_rate


def is_1_before_or_after(a):
    """
    Check if there is a 1 before or after or at the current index. 
    """

    return [(x == 1) or (a[max(i-1, 0)] == 1)
            or (a[min(i+1, len(a) - 1)] == 1) for i, x in enumerate(a)]


def print_minimum_fuel_and_time(dumper_best_speed, dumper_best_capacity, loading_nodes, dumping_nodes_dict, parking_nodes, is_printing=True):
    """
    All loading rates are equal
    All dumping rates are equal
    All dumper speeds are equal
    All dumpers capacity are equal

    Min fuel (not achieveable):
        Start parking:
        One dumper is responsible for each loading node:
            - choosing the nearest dumping node at all times
            - loader doesn't idle while it is gone. 
            - when finished go back to parking spot

    Min time:
        Start parking
        Finds the shortest time possible doing the above. 
        And then in the final transportation
            -   go via dumping node, such that the way to parking node
                is the shortest afterwards.  
    """

    min_fuel = 0
    min_time = 0
    min_distance = 0

    parking_node = parking_nodes[0]
    if len(parking_nodes) > 1:
        embed(header='several parking nodes is not optimized')

    for loading_node in loading_nodes:
        mass_type = loading_node.get_mass_type()

        distances = []

        # Find the shortest way from loading node via dumping -> parking
        dumping_nodes = dumping_nodes_dict[mass_type]
        _shortest_distance_LDP = sys.maxsize
        for dumping_node in dumping_nodes:
            _distance_LD = loading_node.get_shortest_observed_distance_to_node(
                dumping_node)
            distances.append(_distance_LD)

            _distance_DP = dumping_node.get_shortest_observed_distance_to_node(
                parking_node)
            _distance_LDP = _distance_LD + _distance_DP

            if _distance_LDP < _shortest_distance_LDP:
                _shortest_distance_LDP = _distance_LDP

        my_mass = loading_node.get_max_capacity()
        total_trips_LD = np.ceil(my_mass/dumper_best_capacity)
        # this is probably not the best, as you can go through other loading nodes, to do better
        distance_PL = parking_node.get_shortest_observed_distance_to_node(
            loading_node)

        # Parking -> Loading
        time_of_job = distance_PL / dumper_best_speed

        # Time -> all
        time_of_job += my_mass * loading_node.loading_rate + \
            total_trips_LD * loading_node.get_mass_independent_loading_time()

        # Loading -> Dumping -> Parking
        time_of_job += _shortest_distance_LDP / dumper_best_speed
        time_of_job += 30   # dumping rate

        if time_of_job > min_time:
            min_time = time_of_job

        _min_distance = distance_PL

        for distance_idx in np.argsort(distances):
            dumping_node = dumping_nodes[distance_idx]
            dumping_mass = dumping_node.get_max_capacity()

            if my_mass <= dumping_mass:
                # last one is going to the parking spot, therefore -1
                _min_distance += distances[distance_idx] * \
                    (2*total_trips_LD - 1)
                _min_distance += dumping_node.get_shortest_observed_distance_to_node(
                    parking_node)
                break
            else:
                _number_of_trips = dumping_mass/dumper_best_capacity
                # both ways
                _min_distance += distances[distance_idx] * _number_of_trips * 2
                my_mass -= dumping_mass

        min_distance += _min_distance  # in meters

    min_total_driving_time = min_distance/dumper_best_speed
    min_fuel = constants.FUEL_info['driving_fuel_per_second'] * \
        min_total_driving_time

    if is_printing:
        print(f'Minimum (possible not achieveable) measures (Parking -> Jobs -> Parking):')
        print(
            f'Minimum fuel: {min_fuel: 6.2f} | Minimum time: {min_time: 5.0f}')
    else:
        return min_fuel, min_time


def print_reference_fuel_and_time(dumpers, loading_nodes, dumping_nodes, dumping_nodes_dict, parking_nodes, num_dumpers=None, is_printing=True):
    """
    DOESN'T WORK IF YOU SHOULD SERVE SEVERAL DUMPING NODES
    SO IT SHOULD JUST BE one D for each L.

    NUMBER OF TASKS > NUMBER OF DUMPERS ON EACH LOADING NODE


    All loading rates are equal
    All dumping rates are equal
    All dumper speeds are equal
    All dumpers capacity are equal

    Min fuel (not achieveable):
        Start parking:
        One dumper is responsible for each loading node:
            - choosing the nearest dumping node at all times
            - loader doesn't idle while it is gone. 
            - when finished go back to parking spot

    Min time:
        Start parking
        Finds the shortest time possible doing the above. 
        And then in the final transportation
            -   go via dumping node, such that the way to parking node
                is the shortest afterwards.  
    """

    reference_fuel = 0
    reference_time = 0

    if num_dumpers is not None:
        dumpers = dumpers[:num_dumpers]

    dumper = dumpers[0]
    dumper_speed = dumper.get_speed()
    dumper_capacity = dumper.get_mass_capacity()

    parking_node = parking_nodes[0]
    if len(parking_nodes) > 1:
        embed(header='several parking nodes is not optimized')

    dumping_rest_capacity = []
    for dumping_node in dumping_nodes:
        dumping_rest_capacity.append(dumping_node.get_max_capacity())

    # Finds the one that needs that has the worst possible dumping node, and rank accordingly
    longest_distance_LD = [0]*len(loading_nodes)
    total_distances_to_dumping_nodes = []
    for idx, loading_node in enumerate(loading_nodes):
        _mass_type = loading_node.get_mass_type()
        distances_to_dumping_nodes = []
        for dumping_node in dumping_nodes_dict[_mass_type]:
            _dist = loading_node.get_shortest_observed_distance_to_node(
                dumping_node)

            distances_to_dumping_nodes.append(_dist)
            if _dist > longest_distance_LD[idx]:
                longest_distance_LD[idx] = _dist

        total_distances_to_dumping_nodes.append(distances_to_dumping_nodes)

    rank_loading_node = np.argsort(longest_distance_LD)[::-1]

    # Finds out which dumping node to send to
    plans = {}
    for idx in rank_loading_node:
        loading_node = loading_nodes[idx]
        _mass_type = loading_node.get_mass_type()
        plans[loading_node] = {}
        remaining_mass_to_move = loading_node.get_max_capacity()

        ranking_index_of_dumping_nodes = np.argsort(
            total_distances_to_dumping_nodes[idx])
        ranked_dumping_nodes = np.array(dumping_nodes_dict[_mass_type])[
            ranking_index_of_dumping_nodes]

        for dumping_node in ranked_dumping_nodes:
            idx_dumping_node = dumping_nodes.index(dumping_node)
            remaining_mass_to_dump = dumping_rest_capacity[idx_dumping_node]
            mass_amount = min(remaining_mass_to_dump, remaining_mass_to_move)

            if mass_amount > 0:
                plans[loading_node][dumping_node] = mass_amount

            dumping_rest_capacity[idx_dumping_node] -= mass_amount
            remaining_mass_to_move -= mass_amount

            if remaining_mass_to_move == 0:
                break

    # Find the amount of dumpers needed on every Loading node
    max_amount_of_dumpers_needed_to_L = [0]*len(loading_nodes)
    dumpers_provided_to_L = [0]*len(loading_nodes)
    for idx, loading_node in enumerate(loading_nodes):
        time_to_load = dumper.predict_loading_time(loading_nodes[0])
        for dumping_node in plans[loading_node]:

            time_to_drive = (
                2 * loading_node.get_shortest_observed_distance_to_node(dumping_node))/dumper_speed
            time_to_dump = dumper.predict_dumping_time()

            _max_num_of_dumpers = (
                time_to_drive + time_to_dump)/time_to_load + 1
            if _max_num_of_dumpers > max_amount_of_dumpers_needed_to_L[idx]:
                max_amount_of_dumpers_needed_to_L[idx] = _max_num_of_dumpers
                dumpers_provided_to_L[idx] = int(
                    _max_num_of_dumpers)

    # Find the amount of dumpers that will be assinged to every loading node
    max_amount_of_dumpers_needed_to_L = np.array(
        max_amount_of_dumpers_needed_to_L)
    dumpers_provided_to_L = np.array(dumpers_provided_to_L)

    while np.sum(dumpers_provided_to_L) > len(dumpers):
        max_val = np.max(dumpers_provided_to_L)
        max_indexes = np.where(dumpers_provided_to_L == max_val)[0]

        if len(max_indexes) == 1:
            dumpers_provided_to_L[max_indexes[0]] -= 1
        else:
            min_b_index = max_indexes[np.argmin(
                max_amount_of_dumpers_needed_to_L[max_indexes])]
            dumpers_provided_to_L[min_b_index] -= 1

    # Calculate the time of doing all the different tasks, save distances, also

    # Ventetid for loaderne vil skje i perioder: og vil vare
    # (max_amount_of_dumpers_needed_to_L - dumpers_provided_to_L) * loading_time -> lengde
    # Hvis man skal frakt 9 turer, og har 4 stykk så vil ventetiden_Loader være * 2
    #   --- fordi de er pause mellom 4-5, 8-9. int(antall_turer/antall_stykk)
    # Kostnad - alle dumpere kjører parking -> loading node
    # Kostnad - loading-dumping, dumping-loading
    # Kostnad - alle dumpere kjører dumping node -> parking

    total_distance = 0
    total_waiting_time_loaders = 0
    reference_time = 0

    for idx_a, loading_node in enumerate(loading_nodes):
        number_of_dumpers = dumpers_provided_to_L[idx_a]
        number_of_dumpers_no_waiting_time_L = max_amount_of_dumpers_needed_to_L[idx_a]
        curr_total_distance = 0

        number_of_trips = loading_node.get_max_capacity() / dumper_capacity

        # Parking -> Loading
        _dist_PL = parking_node.get_shortest_observed_distance_to_node(
            loading_node)
        curr_total_distance += _dist_PL * number_of_dumpers

        # TODO
        dumping_node = list(plans[loading_node].keys())[0]
        _dist_LD = loading_node.get_shortest_observed_distance_to_node(
            dumping_node)

        # Loading -> Dumping
        curr_total_distance += _dist_LD * number_of_trips

        # Dumping -> Loading
        curr_total_distance += _dist_LD * (number_of_trips - number_of_dumpers)

        # Dumping -> Parking
        _dist_DP = dumping_node.get_shortest_observed_distance_to_node(
            parking_node)
        curr_total_distance += _dist_DP * number_of_dumpers

        # Calculate waiting time for Loaders
        waiting_time_L = (number_of_dumpers_no_waiting_time_L -
                          number_of_dumpers) * dumper.predict_loading_time(loading_node)
        curr_waiting_time_loader = waiting_time_L * \
            (np.ceil(number_of_trips/number_of_dumpers) - 1)

        # Calculate the total time, after the first tour L->D, no queue at L
        curr_time = 0

        # P -> L
        curr_time += _dist_PL / dumper_speed

        # The amount of loadings before the one of interest
        curr_time += dumper.predict_loading_time(loading_node) * (
            (number_of_trips - 1) % number_of_dumpers)

        # The amount of loading of the dumper of interest
        amount_of_LD_for_the_ending_dumper = np.ceil(
            number_of_trips/number_of_dumpers)

        curr_time += dumper.predict_loading_time(
            loading_node) * amount_of_LD_for_the_ending_dumper

        # The amount of dumping for the dumper of interest
        curr_time += dumper.predict_dumping_time() * amount_of_LD_for_the_ending_dumper

        # The dumper of interest driving (L -> D, D -> L, D -> P)
        curr_time += (amount_of_LD_for_the_ending_dumper *
                      _dist_LD)/dumper_speed
        curr_time += ((amount_of_LD_for_the_ending_dumper - 1)
                      * _dist_LD)/dumper_speed
        curr_time += _dist_DP/dumper_speed

        # Add to the total
        total_waiting_time_loaders += curr_waiting_time_loader
        total_distance += curr_total_distance

        if curr_time > reference_time:
            reference_time = curr_time

    total_driving_time = total_distance / dumper_speed  # m -> s
    reference_fuel = total_driving_time * constants.FUEL_info['driving_fuel_per_second'] + \
        total_waiting_time_loaders * constants.FUEL_info['loader_fuel_idling']

    min_fuel, min_time = print_minimum_fuel_and_time(
        dumper_speed, dumper_capacity, loading_nodes, dumping_nodes_dict, parking_nodes, is_printing=False)

    fuel_percent = min_fuel/reference_fuel * 100
    time_percent = min_time/reference_time * 100

    total_percent = 4/5 * fuel_percent + 1/5 * time_percent

    if is_printing:
        print(f'# Dumpers:{sum(dumpers_provided_to_L):3.0f} |', end=' ')
        print(
            f'Sum%:{total_percent:4.1f} | Fuel: {reference_fuel:6.2f} | Time:{reference_time:6.0f} | Distance:{total_distance/1000:5.0f} | Waiting_L:{total_waiting_time_loaders:6.0f}')
    else:
        return reference_fuel, reference_time


def print_statistics_overleaf(framework_a, framework_b, benchmark1_a, benchmark1_b, benchmark2, filename=None):
    framework_a = [str(a) for a in framework_a]
    framework_b = [str(a) for a in framework_b]
    benchmark1_a = [str(a) for a in benchmark1_a]
    benchmark1_b = [str(a) for a in benchmark1_b]
    benchmark2 = [str(a) for a in benchmark2]

    a_str = " & ".join(framework_a) + " & " + \
        " & ".join(benchmark1_a) + " & " + " & ".join(benchmark2)
    b_str = " & ".join(framework_b) + " & " + \
        " & ".join(benchmark1_b) + " & " + \
        " & ".join(benchmark2)

    if filename is None:
        print(a_str)
        print(b_str)
    else:

        # latex
        with open(filename + '_latex.txt', mode='a') as file:
            file.write(a_str + " \\\ \n")
            file.write(b_str + " \\\ \hline \n")

        # normal, easy to pandas
        with open(filename + '.txt', mode='a') as file:
            file.write(a_str + "\n")
            file.write(b_str + "\n")
