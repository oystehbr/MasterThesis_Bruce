OPTIONS_MAIN = f"OPTIONS_MAIN:\
    \n>> bruces_algorithm (FIND SHORTEST DISTANCE ALL-ALL) \
    \n>> change_dumpers num (ADD/ REMOVE num DUMPERS) \
    \n>> coffee_break [TRUE, FALSE] \
    \n>> parking [TRUE, FALSE] \
    \n>> dumper[random, fixed, parking](START LOCATION) \
    \n>> exploration m [times] [break_count] [plans, nodes, coffee] \
    \n>> embed(DEBUGGING)\
    \n>> find_prediction from to(SHOWS BEST EDGE) \
    \n>> get_best_games(ACCORDING TO RECORDS) \
    \n>> load_model \
    \n>> n(RUN n GAMES)\
    \n>> node_choice[policy, optimal] \
    \n>> random_prob p[plans, nodes, coffee](PROBABILITY p, [0, 100], to make a random action(correct))\
    \n>> save_animation name game_number \
    \n>> save_map_image name \
    \n>> save_model filename\
    \n>> train_best_memory num \
    "
OPTIONS_OTHER = f"OPTIONS_OTHER: \
    \n>> add_best_choice_to_node_agents \
    \n>> alpha num (ALPHA IN BELLMAN EQUATION, RANGE: [0, 1]) \
    \n>> clear_plot \
    \n>> finished_training_node_agents bool (TRAIN NODES OR NOT)\
    \n>> MAX_TIME t (simulation length) \
    \n>> plot_shower [score, mass_loaded, mass_time, waiting_time, distance] \
    \n>> print_minimum_fuel_and_time \
    \n>> print_reference_fuel_and_time \
    \n>> re_init_networks [plans, nodes] (REINITIALIZE POLICY) \
    \n>> stop (STOP PROGRAM)\
    \n>> (DUMMY) print_information (SOME INFO FROM GAME) \
    \n>> (NOT IN USE) parking_set_fuel_base [MIN, int] \
    \n>> (NOT IMPLEMENTED) model.train(), model.eval() \
    \n>> (NOT IMPLEMENTED) reset memory \
    \n>> (NOT IMPLEMENTED) reset records \
    "


# loader: https://www.volvoce.com/norge/nb-no/volvo-maskin-as/products/excavators/ec380e/?gclid=CjwKCAiArY2fBhB9EiwAWqHK6i8wxUXDoSWXe7agoQiSsU1js_jnZsb1aMnwNkWcXu5iebKFOdD3rhoCKPAQAvD_BwE#specifications
# chatGPT: 10 - 13 liter/h
# dumper: https://www.volvoce.com/europe/en/products/articulated-haulers/a45g/
# chatGPT: 5.7 - 9.5 liter/h

# TODO: should make the reward function by utilizing these information
FUEL_info = {
    "loader_fuel_idling": 3.785 * 3 / 3600,  # L/s
    "dumper_fuel_idling": 3.785 * 1 / 3600,  # L/s
    # 'dumper_fuel_idling': 13 / 3600,  # L/s
    "driving_fuel_per_second": 3.785 * 8 / 3600,  # L/s,
    # TODO: should be changed - dumper should have control of this
    # L/km = (L/s) / (m/s) * 1000
    "distance_to_fuel": (3.785 * 8 / 3600) / 5 * 1000,
}


# 0.5 L/km * 36 km/h = 18L/h

# 0.3 L / 30 seconds.
# 0.3 * 2 * 60 L/h = 36

# In NOK
COST_info = {
    "salary_loader": 242.15 / 3600,  # salary per second
    "salary_dumper": 233.33 / 3600,  # salary per second
    "fuel_price": 21,  # price per liter fuel
}


#
# D - nearest
# U -> Loading -> shortest dumping
# L -> T/F hvor man skal
# E(R) fra loading agent.
