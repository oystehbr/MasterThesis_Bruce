import random
import numpy as np
import pandas as pd
from nodes_edges import Loading, Dumping, Node, Edge, Parking
from loaders import Loader
from dumpers import Dumper
from IPython import embed
import helper
import json
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
sys.setrecursionlimit(10000)
# TODO: Want this to work


def map1():
    SIX_NODES = [
        Dumping(10000),
        Node(),
        Node(),
        Loading(10000,
                loader=Loader(
                    loading_rate=5,    # s/ tonn,
                    mass_independent_loading_time=30,
                    loader_id='yeah1',
                )
                ),
        Node(),
        Loading(10000,
                loader=Loader(
                    loading_rate=5,    # s/ tonn,
                    mass_independent_loading_time=30,
                    loader_id='yeah2',
                )
                ),
        Node()
    ]

    # TODO: probably need to add all GPS data, and more restrictions
    # if necessary

    edges = []
    edges.append(Edge(SIX_NODES[0], SIX_NODES[1], 100))
    edges.append(Edge(SIX_NODES[1], SIX_NODES[2], 100))
    edges.append(Edge(SIX_NODES[2], SIX_NODES[3], 100))
    edges.append(Edge(SIX_NODES[2], SIX_NODES[4], 100))
    edges.append(Edge(SIX_NODES[4], SIX_NODES[5], 100))
    edges.append(Edge(SIX_NODES[3], SIX_NODES[6], 100))

    edges_all = []
    for edge in edges:
        edges_all.append(edge)
        edges_all.append(edge.get_reversed_edge())

    # TODO: need to make add add_edge() to node

    for node in SIX_NODES:
        print(node.__str__() + ' is connected with: ', end='')
        print(node.get_all_connected_nodes())
        print('----------------------------------------------------------------')

    TWO_DUMPERS = [Dumper(40, 10, 1, 20), Dumper(40, 10, 1, 20)]

    return SIX_NODES, edges_all, TWO_DUMPERS


def map2():
    NODES = [
        Loading(10000, 20),
        Loading(200, 20),
        Dumping(10000, 5),
        Dumping(10000, 5)
    ]

    edges = []
    edges.append(Edge(NODES[0], NODES[1], 50))
    edges.append(Edge(NODES[1], NODES[2], 50))
    edges.append(Edge(NODES[2], NODES[3], 50))
    edges.append(Edge(NODES[3], NODES[0], 50))

    # TODO: need to make add add_edge() to node

    for node in NODES:
        print(node.__str__() + ' is connected with: ', end='')
        print(node.get_all_connected_nodes())
        print('----------------------------------------------------------------')

    return NODES


def map3(print_connections=True):

    with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/mass_transport_problemWithZCoordinates.json') as json_file:
        data = json.load(json_file)

    spatial_graph_container = data['SpatialGraphContainer']
    origin_ = spatial_graph_container['Network']['coordinate_system']['origoInLongitudeLatitude']
    # reordering
    origin = (origin_[1], origin_[0])  # if failing, add float

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}
    for node in spatial_graph_container['Network']['Nodes']:
        coords_df = helper.get_informative_coords_df(
            [node['coordinates']], origin)
        nodes_dict[node['id']] = [Node(0, 0, node['id'], coords_df)]

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################
    loaders = {}
    for loader in data['Loaders']:
        mass_extra_load_time = helper.get_second_from_time_formatted(
            loader['MassIndependentLoadingTime'])
        loaders[loader['Id']] = \
            Loader(loading_rate=10,  # TODO: switch, but kind of wierd values?? loader['LoadingRate'],
                   # TODO: convert to seconds
                   mass_independent_loading_time=mass_extra_load_time,
                   loader_id=loader['Id']
                   )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper in data['Dumpers']:
        mass_extra_dump_time = helper.get_second_from_time_formatted(
            dumper['MassIndependentUnloadingRate'])
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=40,
                speed=dumper['AverageSpeed'],
                unloading_rate=1,  # TODO:['UnloadingRate']
                mass_independent_dumping_time=mass_extra_dump_time,
                dumper_id=dumper['Id']
            )

        )

    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################
    for jobs in data['MassTransportationJobs']:

        # Fix loading location
        load_id = int(jobs['LoadPositionId'])
        _node = nodes_dict[load_id][-1]

        mass_type = jobs['Id']
        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=jobs['TotalVolume'],
                coordinates=_node.get_coordinates(),
                loader=loaders[jobs['LoaderId']],
                mass_type=mass_type
            )
            nodes_dict[load_id].append(loading_node)
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = int(jobs['DumpPositionId'])
        _node = nodes_dict[dump_id][-1]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=jobs['TotalVolume'],
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates(),
                mass_type=mass_type
            )
            nodes_dict[dump_id].append(dumping_node)
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for edge in spatial_graph_container['Network']['Edges']:
        coords_df = helper.get_informative_coords_df(
            edge['coordinates'], origin, is_route=True
        )

        from_node = nodes_dict[edge['Node1']['nodeId']][0]
        to_node = nodes_dict[edge['Node2']['nodeId']][0]
        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=edge['id'],
                         gps=coords_df)

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    ###########################################################
    #### ADDING SMALL INTERSECTIONS BEFORE DUMPING/LOADING ####
    ###########################################################

    for idx, node_id in enumerate(nodes_dict):
        # only for the intersection nodes, before dumping/ loading
        if len(nodes_dict[node_id]) == 2:
            edge_node = Edge(from_node=nodes_dict[node_id][0],
                             to_node=nodes_dict[node_id][1],
                             id=f'intersection_{idx}',
                             distance=10)    # set 10 meters distance
            edges.append(edge_node)
            edges.append(edge_node.get_reversed_edge())

    nodes = Node.nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')

    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping

    return nodes, edges, loaders, dumpers, nodes_dict


def map3_without_intersection(print_connections=True):

    with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/mass_transport_problemWithZCoordinates.json') as json_file:
        data = json.load(json_file)

    spatial_graph_container = data['SpatialGraphContainer']
    origin_ = spatial_graph_container['Network']['coordinate_system']['origoInLongitudeLatitude']
    # reordering
    origin = (origin_[1], origin_[0])  # if failing, add float

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}
    for node in spatial_graph_container['Network']['Nodes']:
        coords_df = helper.get_informative_coords_df(
            [node['coordinates']], origin)
        nodes_dict[node['id']] = Node(0, 0, node['id'], coords_df)

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################
    loaders = {}
    for loader in data['Loaders']:
        mass_extra_load_time = helper.get_second_from_time_formatted(
            loader['MassIndependentLoadingTime'])
        loaders[loader['Id']] = \
            Loader(loading_rate=10,  # TODO: switch, but kind of wierd values?? loader['LoadingRate'],
                   # TODO: convert to seconds
                   mass_independent_loading_time=mass_extra_load_time,
                   loader_id=loader['Id']
                   )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper in data['Dumpers']:
        mass_extra_dump_time = helper.get_second_from_time_formatted(
            dumper['MassIndependentUnloadingRate'])
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=40,
                speed=dumper['AverageSpeed'],
                unloading_rate=1,  # TODO:['UnloadingRate']
                mass_independent_dumping_time=mass_extra_dump_time,
                dumper_id=dumper['Id']
            )

        )

    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################
    for jobs in data['MassTransportationJobs']:

        # Fix loading location
        load_id = int(jobs['LoadPositionId'])
        _node = nodes_dict[load_id]

        mass_type = jobs['Id']
        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=jobs['TotalVolume'],
                coordinates=_node.get_coordinates(),
                loader=loaders[jobs['LoaderId']],
                mass_type=mass_type,
                id=load_id
            )
            nodes_dict[load_id] = loading_node
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = int(jobs['DumpPositionId'])
        _node = nodes_dict[dump_id]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=jobs['TotalVolume'],
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates(),
                mass_type=mass_type,
                id=jobs['DumpPositionId']
            )
            nodes_dict[dump_id] = dumping_node
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for edge in spatial_graph_container['Network']['Edges']:
        coords_df = helper.get_informative_coords_df(
            edge['coordinates'], origin, is_route=True
        )

        from_node = nodes_dict[edge['Node1']['nodeId']]
        to_node = nodes_dict[edge['Node2']['nodeId']]
        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=edge['id'],
                         gps=coords_df)

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    ###########################################################
    #### ADDING SMALL INTERSECTIONS BEFORE DUMPING/LOADING ####
    ###########################################################
    # NOT INCLUDED - HAVE NO FUNCTIONALITY in this moment - map3 - with intersection
    # for idx, node_id in enumerate(nodes_dict):
    #     # only for the intersection nodes, before dumping/ loading
    #     if len(nodes_dict[node_id]) == 2:
    #         edge_node = Edge(from_node=nodes_dict[node_id][0],
    #                          to_node=nodes_dict[node_id][1],
    #                          id=f'intersection_{idx}',
    #                          distance=10)    # set 10 meters distance
    #         edges.append(edge_node)
    #         edges.append(edge_node.get_reversed_edge())

    nodes = list(nodes_dict.values())
    Node.nodes = nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')
    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping

    # embed()
    return nodes, edges, loaders, dumpers, nodes_dict


def map3_without_intersection_small_jobs(print_connections=True):

    with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/mass_transport_problemWithZCoordinates.json') as json_file:
        data = json.load(json_file)

    spatial_graph_container = data['SpatialGraphContainer']
    origin_ = spatial_graph_container['Network']['coordinate_system']['origoInLongitudeLatitude']
    # reordering
    origin = (origin_[1], origin_[0])  # if failing, add float

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}
    for node in spatial_graph_container['Network']['Nodes']:
        coords_df = helper.get_informative_coords_df(
            [node['coordinates']], origin)
        nodes_dict[node['id']] = Node(0, 0, node['id'], coords_df)

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################
    loaders = {}
    for loader in data['Loaders']:
        mass_extra_load_time = helper.get_second_from_time_formatted(
            loader['MassIndependentLoadingTime'])
        loaders[loader['Id']] = \
            Loader(loading_rate=10,  # TODO: switch, but kind of wierd values?? loader['LoadingRate'],
                   # TODO: convert to seconds
                   mass_independent_loading_time=mass_extra_load_time,
                   loader_id=loader['Id']
                   )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper in data['Dumpers']:
        mass_extra_dump_time = helper.get_second_from_time_formatted(
            dumper['MassIndependentUnloadingRate'])
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=40,
                speed=dumper['AverageSpeed'],
                unloading_rate=1,  # TODO:['UnloadingRate']
                mass_independent_dumping_time=mass_extra_dump_time,
                dumper_id=dumper['Id']
            )

        )

    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################
    for jobs in data['MassTransportationJobs']:

        jobs['TotalVolume'] = 400
        # Fix loading location
        load_id = int(jobs['LoadPositionId'])
        _node = nodes_dict[load_id]

        mass_type = jobs['Id']
        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=jobs['TotalVolume'],
                coordinates=_node.get_coordinates(),
                loader=loaders[jobs['LoaderId']],
                mass_type=mass_type,
                id=load_id
            )
            nodes_dict[load_id] = loading_node
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = int(jobs['DumpPositionId'])
        _node = nodes_dict[dump_id]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=jobs['TotalVolume'],
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates(),
                mass_type=mass_type,
                id=jobs['DumpPositionId']
            )
            nodes_dict[dump_id] = dumping_node
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for edge in spatial_graph_container['Network']['Edges']:
        coords_df = helper.get_informative_coords_df(
            edge['coordinates'], origin, is_route=True
        )

        from_node = nodes_dict[edge['Node1']['nodeId']]
        to_node = nodes_dict[edge['Node2']['nodeId']]
        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=edge['id'],
                         gps=coords_df)

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    ###########################################################
    #### ADDING SMALL INTERSECTIONS BEFORE DUMPING/LOADING ####
    ###########################################################
    # NOT INCLUDED - HAVE NO FUNCTIONALITY in this moment - map3 - with intersection
    # for idx, node_id in enumerate(nodes_dict):
    #     # only for the intersection nodes, before dumping/ loading
    #     if len(nodes_dict[node_id]) == 2:
    #         edge_node = Edge(from_node=nodes_dict[node_id][0],
    #                          to_node=nodes_dict[node_id][1],
    #                          id=f'intersection_{idx}',
    #                          distance=10)    # set 10 meters distance
    #         edges.append(edge_node)
    #         edges.append(edge_node.get_reversed_edge())

    nodes = list(nodes_dict.values())
    Node.nodes = nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')
    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping

    # embed()
    return nodes, edges, loaders, dumpers, nodes_dict


def map4(print_connections=True, plot_graph=True):
    """
    Not "infinty with mass"
    """

    with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/mass_transport_problemWithZCoordinates.json') as json_file:
        data = json.load(json_file)

    spatial_graph_container = data['SpatialGraphContainer']
    origin_ = spatial_graph_container['Network']['coordinate_system']['origoInLongitudeLatitude']
    # reordering
    origin = (origin_[1], origin_[0])  # if failing, add float

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}
    for node in spatial_graph_container['Network']['Nodes']:
        coords_df = helper.get_informative_coords_df(
            [node['coordinates']], origin)
        nodes_dict[node['id']] = [Node(0, 0, node['id'], coords_df)]

    ############################################################
    ################### CREATING RANDOM NODES ##################
    ############################################################
    rest_start_id = node['id'] + 1
    random.seed(7)
    np.random.seed(7)

    for start_id in range(rest_start_id, int(3*rest_start_id/2)):
        x = random.randint(2000, 6000)
        y = random.randint(6000, 9000)

        coords_df = pd.DataFrame([[x, y]])
        coords_df.columns = ['x', 'y']

        nodes_dict[start_id] = [Node(0, 0, start_id, coords_df)]

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################

    loaders = {}
    for loader in data['Loaders']:
        mass_extra_load_time = helper.get_second_from_time_formatted(
            loader['MassIndependentLoadingTime'])
        loaders[loader['Id']] = Loader(loading_rate=10,  # TODO: switch, but kind of wierd values?? loader['LoadingRate'],
                                       # TODO: convert to seconds
                                       mass_independent_loading_time=mass_extra_load_time,
                                       loader_id=loader['Id']
                                       )

    ############################################################
    ################### CREATING RANDOM LOADERS ################
    ############################################################

    for random_loader_num in range(10):
        loaders[f'R{random_loader_num}'] = Loader(
            loading_rate=10,
            mass_independent_loading_time=30,
            loader_id=f'R{random_loader_num}'
        )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper in data['Dumpers']:
        mass_extra_dump_time = helper.get_second_from_time_formatted(
            dumper['MassIndependentUnloadingRate'])
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=40,
                speed=dumper['AverageSpeed'],
                unloading_rate=1,  # TODO:['UnloadingRate']
                mass_independent_dumping_time=mass_extra_dump_time,
                dumper_id=dumper['Id']
            )
        )

    ############################################################
    ####################### ADDING EXTRA JOBS ##################
    ############################################################

    points = np.unique(np.random.randint(
        rest_start_id, int(3*rest_start_id/2), 20))
    np.random.shuffle(points)
    loading_points = points[:5]
    dumping_points = points[5:7]

    data['MassTransportationJobs'].append(
        {
            'Id': '100',
            'LoadPositionId': f'{loading_points[0]}',
            'DumpPositionId': f'2',
            'LoaderId': f'R{0}',
            'TotalVolume': 10000
        }
    )

    data['MassTransportationJobs'].append(
        {
            'Id': '100',
            'LoadPositionId': f'{loading_points[1]}',
            'DumpPositionId': f'18',
            'LoaderId': f'R{1}',
            'TotalVolume': 10000
        }
    )

    for extra_job_num in range(2, 5):
        job = {
            'Id': '100',
            'LoadPositionId': f'{loading_points[extra_job_num]}',
            'DumpPositionId': f'{np.random.choice(dumping_points, 1)[0]}',
            'LoaderId': f'R{extra_job_num}',
            'TotalVolume': 10000
        }

        data['MassTransportationJobs'].append(job)

    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################

    for jobs in data['MassTransportationJobs']:

        # Fix loading location
        load_id = int(jobs['LoadPositionId'])
        _node = nodes_dict[load_id][-1]

        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=jobs['TotalVolume'],
                coordinates=_node.get_coordinates(),
                loader=loaders[jobs['LoaderId']]
            )
            nodes_dict[load_id].append(loading_node)
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = int(jobs['DumpPositionId'])
        _node = nodes_dict[dump_id][-1]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=jobs['TotalVolume'],
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates()
            )
            nodes_dict[dump_id].append(dumping_node)
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for edge in spatial_graph_container['Network']['Edges']:
        coords_df = helper.get_informative_coords_df(
            edge['coordinates'], origin, is_route=True
        )

        from_node = nodes_dict[edge['Node1']['nodeId']][0]
        to_node = nodes_dict[edge['Node2']['nodeId']][0]
        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=edge['id'],
                         gps=coords_df)

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    #############################################################
    ####################### ADDING EXTRA EDGES ##################
    #############################################################
    edge_counter = 100
    for i in range(rest_start_id, int(3*rest_start_id/2) - 1):
        from_node = nodes_dict[i][0]
        to_node = nodes_dict[i+1][0]

        df_xy = pd.concat([from_node.get_coordinates(),
                           to_node.get_coordinates()])

        coords_df = helper.get_informative_coords_df_xy(
            df_xy, is_route=True
        )

        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=f'e_{edge_counter}',
                         gps=coords_df)

        edge_counter += 1

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    random_edges = np.sort(np.random.randint(
        rest_start_id, int(3*rest_start_id/2), [10, 2]))
    unique_random_edges = np.unique(random_edges, axis=0)
    unique_random_edges = np.vstack([unique_random_edges, [20, 66]])
    unique_random_edges = np.vstack([unique_random_edges, [34, 67]])

    for edge in unique_random_edges:
        from_node_num = edge[0]
        to_node_num = edge[1]

        if from_node_num == to_node_num:
            continue

        from_node = nodes_dict[from_node_num][0]
        to_node = nodes_dict[to_node_num][0]

        df_xy = pd.concat([from_node.get_coordinates(),
                           to_node.get_coordinates()])

        coords_df = helper.get_informative_coords_df_xy(
            df_xy, is_route=True
        )

        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=f'e_{edge_counter}',
                         gps=coords_df)

        edge_counter += 1

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    ###########################################################
    #### ADDING SMALL INTERSECTIONS BEFORE DUMPING/LOADING ####
    ###########################################################

    for idx, node_id in enumerate(nodes_dict):
        # only for the intersection nodes, before dumping/ loading
        if len(nodes_dict[node_id]) == 2:
            edge_node = Edge(from_node=nodes_dict[node_id][0],
                             to_node=nodes_dict[node_id][1],
                             id=f'intersection_{idx}',
                             distance=10)    # set 10 meters distance
            edges.append(edge_node)
            edges.append(edge_node.get_reversed_edge())

    nodes = Node.nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')

    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping

    for node in nodes:
        x = int(node.get_coordinates()['x'])
        y = int(node.get_coordinates()['y'])

        plt.scatter(x, y)
        plt.annotate(f'{node}', (x, y))

    for edge_num in range(1, len(edges), 2):
        edge = edges[edge_num]
        from_node = edge.get_from_node()
        to_node = edge.get_to_node()

        x1, y1 = from_node.get_coordinates()[['x', 'y']].values[0]
        x2, y2 = to_node.get_coordinates()[['x', 'y']].values[0]

        plt.plot([x1, x2], [y1, y2])

    # plt.savefig(f'BIG_MAP4.png')

    if plot_graph:
        plt.show()    # embed(header='yeah')

    return nodes, edges, loaders, dumpers, nodes_dict


def map5_helga(print_connections=True):

    with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/simple_test.json') as json_file:
        data = json.load(json_file)

    spatial_graph_container = data['SpatialGraphContainer']
    origin_ = spatial_graph_container['Network']['coordinate_system']['origoInLongitudeLatitude']
    # reordering
    origin = (origin_[1], origin_[0])  # if failing, add float

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}
    for node in spatial_graph_container['Network']['Nodes']:
        coords_df = helper.get_informative_coords_df(
            [node['coordinates']], origin)
        nodes_dict[node['id']] = Node(0, 0, node['id'], coords_df)

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################
    loaders = {}
    for loader in data['Loaders']:
        mass_extra_load_time = helper.get_second_from_time_formatted(
            loader['MassIndependentLoadingTime'])
        loaders[loader['Id']] = \
            Loader(loading_rate=10,  # TODO: switch, but kind of wierd values?? loader['LoadingRate'],
                   # TODO: convert to seconds
                   mass_independent_loading_time=mass_extra_load_time,
                   loader_id=loader['Id']
                   )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper in data['Dumpers']:
        mass_extra_dump_time = helper.get_second_from_time_formatted(
            dumper['MassIndependentUnloadingRate'])
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=dumper['Capacity'],
                speed=dumper['AverageSpeed'],
                unloading_rate=1,  # TODO:['UnloadingRate']
                mass_independent_dumping_time=mass_extra_dump_time,
                dumper_id=dumper['Id']
            )

        )

    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################
    for jobs in data['MassTransportationJobs']:

        # jobs['TotalVolume'] = 400
        # Fix loading location
        load_id = int(jobs['LoadPositionId'])
        _node = nodes_dict[load_id]

        mass_type = jobs['Id']
        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=jobs['TotalVolume'],
                coordinates=_node.get_coordinates(),
                loader=loaders[jobs['LoaderId']],
                mass_type=mass_type,
                id=load_id
            )
            nodes_dict[load_id] = loading_node
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = int(jobs['DumpPositionId'])
        _node = nodes_dict[dump_id]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=jobs['TotalVolume'],
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates(),
                mass_type=mass_type,
                id=jobs['DumpPositionId']
            )
            nodes_dict[dump_id] = dumping_node
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + jobs['TotalVolume']
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for edge in spatial_graph_container['Network']['Edges']:
        coords_df = helper.get_informative_coords_df(
            edge['coordinates'], origin, is_route=True
        )

        from_node = nodes_dict[edge['Node1']['nodeId']]
        to_node = nodes_dict[edge['Node2']['nodeId']]
        edge_node = Edge(from_node=from_node,
                         to_node=to_node,
                         id=edge['id'],
                         gps=coords_df)

        # Add edge in both directions
        edges.append(edge_node)
        edges.append(edge_node.get_reversed_edge())

    ###########################################################
    #### ADDING SMALL INTERSECTIONS BEFORE DUMPING/LOADING ####
    ###########################################################
    # NOT INCLUDED - HAVE NO FUNCTIONALITY in this moment - map3 - with intersection
    # for idx, node_id in enumerate(nodes_dict):
    #     # only for the intersection nodes, before dumping/ loading
    #     if len(nodes_dict[node_id]) == 2:
    #         edge_node = Edge(from_node=nodes_dict[node_id][0],
    #                          to_node=nodes_dict[node_id][1],
    #                          id=f'intersection_{idx}',
    #                          distance=10)    # set 10 meters distance
    #         edges.append(edge_node)
    #         edges.append(edge_node.get_reversed_edge())

    nodes = list(nodes_dict.values())
    Node.nodes = nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')
    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping

    # embed()
    return nodes, edges, loaders, dumpers, nodes_dict


def map_generator(num_dumpers=5, rows=10, cols=10, num_loading_nodes=5, loading_rates=[4]*5, num_dumping_nodes=5,
                  pre_dumping_nodes=None, pre_loading_nodes=None, pre_parking_nodes=None, mass_type=['0'],
                  max_capacity=400, edge_changes={}, print_connections=True, seed=7):

    # Resetting all stuff
    Dumper.dumper_counter = 0
    Dumper.dumpers = []
    Node.node_counter = 0
    Node.nodes = []
    Loader.loader_counter = 0
    Loader.loaders = []

    ############################################################
    ####################### ADDING NODES #######################
    ############################################################
    nodes_dict = {}

    for i in range(rows):
        for j in range(cols):
            coords_df = pd.DataFrame(data={'x': [j], 'y': [(rows-1)-i]})
            node_id = cols*i + j
            nodes_dict[node_id] = Node(0, 0, node_id, coords_df)

    ############################################################
    ###################### ADDING LOADERS ######################
    ############################################################
    # loaders = {}
    # for i in range(num_loading_nodes):
    #     loaders[i] = \
    #         Loader(loading_rate=4,  # second per ton
    #                mass_independent_loading_time=30,
    #                loader_id=i
    #                )
    loaders = {}
    for i in range(num_loading_nodes):
        loaders[i] = \
            Loader(loading_rate=loading_rates[i],  # second per ton, normal: 4 sec per ton
                   mass_independent_loading_time=30,
                   loader_id=i
                   )

    ############################################################
    ###################### ADDING DUMPERS ######################
    ############################################################
    dumpers = []
    for dumper_num in range(num_dumpers):
        dumpers.append(
            Dumper(
                # TODO: is too much with 300 ton dumper['Capacity'],
                mass_capacity=40,   # ton
                speed=5,   # m/s
                unloading_rate=0,  # second per ton
                mass_independent_dumping_time=30,
                dumper_id=f'Crazy dumper {dumper_num}'
            )

        )
    random.seed(seed)
    ############################################################
    ####################### ADDING JOBS ########################
    ############################################################
    jobs = random.sample(
        range(rows*cols), num_loading_nodes + num_dumping_nodes)
    loading_nodes = jobs[:num_loading_nodes]
    dumping_nodes = jobs[num_loading_nodes:]

    if not pre_loading_nodes is None:
        loading_nodes = pre_loading_nodes
    if not pre_dumping_nodes is None:
        dumping_nodes = pre_dumping_nodes

    for job_num in range(max(num_loading_nodes, num_dumping_nodes)):
        load_id = loading_nodes[job_num % len(loading_nodes)]
        _node = nodes_dict[load_id]

        # mass_type = ['0']
        if _node.class_id == 'node':
            loading_node = Loading(
                max_capacity=max_capacity,  # ton
                coordinates=_node.get_coordinates(),
                loader=loaders[job_num % num_loading_nodes],
                mass_type=mass_type[job_num % len(mass_type)],
                id=load_id
            )
            nodes_dict[load_id] = loading_node
        elif _node.class_id == 'loading':
            new_max_capacity = _node.get_max_capacity() + max_capacity
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG')

        # Fix dumping location
        dump_id = dumping_nodes[job_num % len(dumping_nodes)]
        _node = nodes_dict[dump_id]

        if _node.class_id == 'node':
            dumping_node = Dumping(
                max_capacity=max_capacity,
                time=0,   # TODO: should be removed
                coordinates=_node.get_coordinates(),
                mass_type=mass_type[job_num % len(mass_type)],
                id=dump_id
            )
            nodes_dict[dump_id] = dumping_node
        elif _node.class_id == 'dumping':
            new_max_capacity = _node.get_max_capacity() + max_capacity
            _node.set_max_capacity(new_max_capacity)
        else:
            embed(header='SOMETHING WENT WRONG2')

    ############################################################
    #################### ADDING PARKING SPOT ###################
    ############################################################
    if pre_parking_nodes is None:
        # Adding one parking slot
        found_parking_spot = False
        if (rows*cols == (len(dumping_nodes) + len(loading_nodes))):
            embed(header='not place for a parking slot')

        while not found_parking_spot:
            parking_node_id = random.sample(range(rows*cols), 1)[0]
            if parking_node_id in loading_nodes:
                continue
            elif parking_node_id in dumping_nodes:
                continue
            else:
                pre_parking_nodes = [parking_node_id]
                found_parking_spot = True

    for parking_node_id in pre_parking_nodes:
        _node = nodes_dict[parking_node_id]

        nodes_dict[parking_node_id] = Parking(
            id=parking_node_id,
            coordinates=_node.get_coordinates()
        )

    ############################################################
    ####################### ADDING EDGES #######################
    ############################################################
    edges = []
    for i in range(rows):
        for j in range(cols):

            # TODO: Create coordinates on the edge
            curr_id = i*cols + j
            right_id = curr_id + 1
            below_id = curr_id + cols
            curr_node = nodes_dict[curr_id]

            # not on the most right column
            if j < cols - 1:
                right_node = nodes_dict[right_id]
                edge_node = Edge(from_node=curr_node,
                                 to_node=right_node,
                                 id=f'edge_{curr_id}_{right_id}',
                                 gps=None)

                # Add edge in both directions
                edges.append(edge_node)
                edges.append(edge_node.get_reversed_edge())

            # not on the most down row
            if i < rows - 1:
                below_node = nodes_dict[below_id]
                try:
                    edge_node = Edge(from_node=curr_node,
                                     to_node=below_node,
                                     id=f'edge_{curr_id}_{below_id}',
                                     gps=None)
                except RecursionError:
                    embed(header='recursionError')

                # Add edge in both directions
                edges.append(edge_node)
                edges.append(edge_node.get_reversed_edge())

    # Setting the distance for each edge - setting the same for forward and backwards of an edge
    # Set a new random seed here, since we want to generate the same map everytime - but switch loading
    # and dumping nodes
    # TODO: make it possible for directed graphs
    random.seed(7)
    for edge_num in range(0, len(edges), 2):
        random_distance = random.randint(100, 1000)

        if f'{edges[edge_num]}' in edge_changes:
            random_distance = edge_changes[f'{edges[edge_num]}']

        edges[edge_num].set_distance(random_distance)
        edges[edge_num + 1].set_distance(random_distance)

    # for edge in edges:
    #     edge.set_distance(random.randint(100, 1000))

    nodes = list(nodes_dict.values())
    Node.nodes = nodes
    if print_connections:
        for node in nodes:
            print(node.__str__() + ' is connected with: ', end='')
            print(node.get_all_connected_nodes())
            print('----------------------------------------------------------------')
    # Is used somewhere, and will be more - since created dummy nodes, before loading/ dumping
    # embed()
    return nodes, edges, loaders, dumpers, nodes_dict
