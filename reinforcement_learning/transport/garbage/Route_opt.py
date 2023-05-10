import time as timePackage
from agent import Agent
import model
from game import TravelingGameAI
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np
import helper
import time
from tqdm import tqdm


class RouteOptimization():
    def __init__(self, map, MAX_TIME=10000, output_name='output'):
        nodes, edges, loaders, dumpers, nodes_dict = map(False)

        self.agent = Agent(nodes, edges, dumpers)
        self.game = TravelingGameAI(nodes, edges, dumpers)

        self.dumpers = dumpers
        self.edges = edges
        self.nodes = nodes
        self.loaders = loaders
        self.len_dumpers = len(dumpers)
        self.len_nodes = len(nodes)
        self.output_name = output_name
        self.MAX_TIME = MAX_TIME

        self.trigger()

    def trigger(self):
        self.plot_scores = []
        self.plot_scores_mean = []
        self.plot_mass_loaded = []
        self.plot_mass_loaded_mean = []
        self.plot_mass_loaded_per_time = []
        self.plot_mass_loaded_per_time_mean = []

        self.start_time = None
        self.print_information_next_round = False
        self.record = 0
        self.record_loading_mass = 0
        self.plot_shower = 'mass'
        self.output_file = open(f'outputs/{self.output_name}.txt', 'w')
        self.output_file.close()

        model.set_agent(self.agent)

    def trigger_before_new_game(self):
        self.time = 0
        self.time_increment = 0
        self.time_tracker_1000 = 0
        self.total_score = 0
        self.control_score = 0
        self.fails = 0

        self.time_to_next_move = np.zeros(self.len_dumpers)
        self.state_old = [None]*self.len_dumpers
        self.done_old = [None] * self.len_dumpers
        self.reward_old = [None]*self.len_dumpers
        self.final_move_old = [None]*self.len_dumpers
        self.finished = [False]*self.len_dumpers
        self.game_over = False

        self.agent.restart_round_memory()

        # if first node is load, start loading
        for dumper in self.dumpers:
            if dumper.get_current_node().class_id == 'loading':
                dumper.set_destination_finish(True)

        # TODO: if you are at loading in the first iteration - try load.

        # One simulation
        self.failed = False
        self.failed_counter = 0

    def logfile_output(self, txt):
        self.output_file = open(f'outputs/{self.output_name}.txt', 'a')
        self.output_file.write(txt + '\n')
        self.output_file.close()

    def set_values_from_commandline(self, values):
        try:
            self.num_games = int(values[0])
            self.MAX_TIME = int(values[1])
            self.agent.set_exploration_num(int(values[2]))
            self.agent.set_random_choice_prob(int(values[3]))
            self.save_model_filename = values[4]
            self.output_name = self.save_model_filename
        except:
            print('Something was wrong with the commandline args')

    def main(self, command_line_args=False):
        self.command_line_args = command_line_args
        while True:
            if not command_line_args:
                self.option_guide()
                
            self.iter = 0
            self.start_time = timePackage.time()
            for self.iter in tqdm(range(self.num_games)):
                self.trigger_before_new_game()

                while self.time < self.MAX_TIME and not self.game_over and not self.failed:
                    self.simulate_all_next_moves()  # TODO: most time consuming
                    self.status_update_after_moves()

                self.process_info_of_round()
                self.agent.train_current_game()
                self.agent.restart_round_memory()
                self.agent.train_long_memory()
                self.game.reset()

            if command_line_args:
                self.agent.save_model(self.save_model_filename)
                break

    def process_info_of_round(self):

        self.agent.n_games += 1
        self.agent.save_info()

        score_actual = 0
        for dumper in self.agent.dumpers:
            for i, route_key in enumerate(dumper.completed_routes):
                if i != 0:
                    try:
                        score_actual += dumper.completed_routes[route_key]['actual_reward'][0]
                    except IndexError:
                        # TODO: is needed?
                        NotImplemented
                    except KeyError:
                        # TODO: is needed?
                        NotImplemented

        if score_actual > self.record:
            self.record = score_actual

        if self.game.get_mass_loaded() > self.record_loading_mass:
            self.record_loading_mass = self.game.get_mass_loaded()
            # TODO: save the model with the best mass
            # agent.model.save()

        self.plot_scores.append(int(score_actual))
        self.plot_scores_mean.append(np.mean(self.plot_scores[-20:]))

        self.plot_mass_loaded.append(self.game.get_mass_loaded())
        self.plot_mass_loaded_mean.append(
            np.mean(self.plot_mass_loaded[-20:]))

        if self.print_information_next_round:
            self.print_information_next_round = False
            print('\n\nINFO:')
            print('NODES:')
            for node in self.nodes:
                if node.class_id == 'dumping' or node.class_id == 'loading':
                    print(
                        node, f'MAX: {node.get_max_capacity()}, REST: {node.get_rest_capacity()}')

        min_used_node, min_used_num = self.game.get_min_used_node()

        self.logfile_output(
            f'Game: {self.agent.n_games: 3.0f}, ' +
            f'Mass: {self.plot_mass_loaded[-1]: 5.0f}, ' +
            f'Record_mass: {self.record_loading_mass: 4.0f}, ' +
            f'Fails: {self.fails: 4.0f}, ' +
            f'Node min used: {min_used_node} ({min_used_num: 4.0f} ton), ' +
            f'Score: {score_actual : .0f}, '
            # f'Control_score: {self.control_score}, ' +
        )

        self.plot_mass_loaded_per_time.append(
            self.plot_mass_loaded[-1]/self.MAX_TIME)
        self.plot_mass_loaded_per_time_mean.append(
            np.mean(self.plot_mass_loaded_per_time[-20:])
        )

        if not self.command_line_args:
            if self.plot_shower == 'score':
                helper.plot(self.plot_scores,
                            self.plot_scores_mean,
                            ylabel='score')
            elif self.plot_shower == 'mass':
                helper.plot(self.plot_mass_loaded,
                            self.plot_mass_loaded_mean,
                            ylabel='Mass of ton (dumped + moved)')
            elif self.plot_shower == 'mass_time':
                helper.plot(self.plot_mass_loaded_per_time,
                            self.plot_mass_loaded_per_time_mean,
                            ylabel='Mass_of_ton (dumped + moved) * 1/t')

    def simulate_all_next_moves(self):
        self.time_to_next_move -= self.time_increment
        activity_idx = np.where(self.time_to_next_move == 0)[0]
        for idx in activity_idx:
            dumper = self.dumpers[idx]
            # TODO: if dumper is ready for next move, if not just remove time on current task
            dumper.set_current_node(dumper.get_next_node())

            if dumper.get_destination_finish():  # TODO: reached final destination (name)
                # ready to start dumping or loading

                # TODO: check if it should park here, or go further. NEW NEURAL NETWORK,
                # TODO: make new method in dumpers, state_info_choice_load_dump_further???
                wait = self.dumper_ready_for_loading_dumping(idx, dumper)

                # Should wait, so the first in first out works.
                if wait:
                    continue

            else:
                # If finished dumping or loading, or visit all nodes - calculate the correct reward for path
                if dumper.is_loading() or dumper.is_dumping() or dumper.is_all_possible_nodes_used():
                    self.dumper_finished_a_route(dumper)

                # check if finished
                if dumper.get_amount_of_mass_on() == 0:
                    if self.game.get_rest_dumping_mass() == 0 or self.game.get_rest_loading_mass() == 0:
                        self.finished[idx] = True
                        self.time_to_next_move[idx] = self.MAX_TIME - self.time

                if self.finished[idx]:
                    continue

                # Find a possible move, negativ reward if not
                state, final_move = self.dumper_find_a_possible_move(dumper)

                dumper.add_state_info(helper.State(state, final_move))

                # Store values, can't predict next step before the step shall be taken.

                self.time_to_next_move[idx] = dumper.get_time_to_next_node(
                )
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

        if dumper.get_num_in_queue() is None:
            # If there are someone in queue, that should go next - wait a sec to make order correct
            if current_node.get_num_queue() > 0 and not current_node.is_used():
                self.time_to_next_move[idx] = 1
                return True

            current_node.add_queue(dumper)
            dumper.set_num_in_queue(
                current_node.get_num_queue())
        else:
            dumper.set_num_in_queue(
                dumper.get_num_in_queue() - 1)

        if not current_node.is_used() and dumper.get_num_in_queue() == 1:
            mass_before = dumper.get_amount_of_mass_on()

            # TODO: could be in start_ - functions
            time_since_last_time_used = current_node.use(self.time)
            if current_node.class_id == 'loading':
                # TODO: check if it is possible to load mass here

                amount_loaded = dumper.start_loading(
                    current_node, self.time, self.game)

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
        else:

            # Need to wait to the node is ready to be used.
            time_finished = current_node.get_time_finished_use()
            next_time_interaction = time_finished - self.time + \
                1 + dumper.get_num_in_queue()  # TODO: 1 sec to switch, and first in first out

            dumper.add_waiting_time(next_time_interaction)

        self.time_to_next_move[idx] = next_time_interaction

        return False

    def dumper_finished_a_route(self, dumper):
        # give some score everytime it loads mass
        self.control_score += 1

        time_of_task = 0

        # if dumper.get_current_completed_route()['actual_route'][-1].class_id == 'loading':
        #     embed(header='lolzz')

        if dumper.is_loading():
            time_of_task = dumper.get_loading_time()
            dumper.finish_loading(self.time)

        elif dumper.is_dumping():
            time_of_task = dumper.get_dumping_time()
            dumper.finish_dumping(self.time)

        # used all - and last used was not success dumping or loading
        if dumper.is_all_possible_nodes_used() and time_of_task == 0:
            # TODO: if negativ reward, to not discount it
            dumper.change_last_direct_reward(-10)
            self.fails += 1
            time_of_task = 1

        # could end the route also - but give penalty
        if time_of_task != 0:
            dumper.end_the_route(self.time, self.agent)
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

            route['actual_reward'] = calculator.get_updated_rewards(
            )

            # TODO: need to be previous
            dumper.add_new_info_to_current_completed_route(
                'waiting_time', dumper.get_waiting_time())

            dumper.add_new_info_to_current_completed_route(
                'driving_time', calculator.get_driving_time())

            dumper.add_new_info_to_current_completed_route(
                'time_since_last_used_node', dumper.get_time_since_last_used_node())

            dumper.set_time_since_last_used_node(0)

            dumper.add_new_info_to_current_completed_route(
                'mass_end', dumper.get_amount_of_mass_on())

            self.agent.add_route_to_round_memory(route)

            # and train memory
            dumper.init_new_route(self.time)
            dumper.reset_waiting_time()

    def dumper_find_a_possible_move(self, dumper):
        # Find a possible move, negativ reward if not
        possible_move = False
        state = self.agent.get_state(self.game, dumper)
        not_possible_move_counter = 0
        while not possible_move:    # TODO: make it bad if you go backwards
            # get move
            final_move = self.agent.get_action(state, dumper)

            # perform move and get new state    # TODO: remember amount of mass on dumper etc -> special case
            reward, done, score, possible_move = self.game.play_step(
                final_move, dumper, self.time)   # TODO: make play_step

            # TODO: could make an active function inside dumper, so it should have maked one move first.

            # If it tries to move to a location that isn't connected
            if not possible_move:
                not_possible_move_counter += 1
                self.agent.train_short_memory(
                    state, final_move, reward, state, False)

                # TODO: maybe?
                # agent.remember(state, final_move,
                #                reward, state, False)

            if (not_possible_move_counter + 1) % 500 == 0:
                model.set_dumper(dumper)
                self.agent.set_random_choice_prob(
                    self.agent.get_random_choice_prob() + 1)
                # model.set_now(True)

                if (not_possible_move_counter + 1) % 1500 == 0:
                    embed(header='long time')

        return state, final_move

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
                embed(header='time_increment')
        else:
            self.failed_counter = 0

    def option_guide(self):
        if not(self.start_time is None):
            stop_time = timePackage.time()
            time_usage = stop_time - self.start_time
            print(
                f'### Time used for {self.num_games} games: {time_usage : .2f} seconds ### \n\n')

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
        print(
            ">> MAX_TIME t (SIMULATION LENGTH t in seconds, default = 3600    ### 1 hour)")
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
                self.agent.set_random_choice_prob(int(input_lower.split()[1]))
            except ValueError:
                print(input_lower.split()[1] + ' is not an int')
        elif input_lower_first == 'plot_shower':
            if input_lower.split()[1] == 'score':
                self.plot_shower = 'score'
            elif input_lower.split()[1] == 'mass':
                self.plot_shower = 'mass'
            elif input_lower.split()[1] == 'mass_time':
                self.plot_shower = 'mass_time'
        elif input_lower_first == 'save_model':
            filename = input_.split()[1]
            self.agent.save_model(filename)
        elif input_lower_first == 'load_model':
            self.agent.load_model()
        elif input_lower_first == 'exploration':
            try:
                self.agent.set_exploration_num(int(input_lower.split()[1]))
            except ValueError:
                print(input_lower.split()[1] + 'is not an int')
        elif input_lower_first in ['print_information', 'pf']:
            self.print_information_next_round = True
            input_ = '1'
        elif input_lower_first == 'max_time':
            try:
                self.MAX_TIME = int(input_lower.split()[1])
            except Exception as e:
                print(e)
                print('Could not set new MAX_TIME')

        try:
            self.num_games = int(input_)
        except:
            self.num_games = 0
