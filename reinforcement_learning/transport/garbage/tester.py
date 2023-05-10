import random
import json
import os
import road_data_manipulation as rdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
os.getcwd()


with open('/Users/oysteinbruce/Documents/GitHub/SINTEF/other/ReinforcementLearning/transport/data/mass_transport_problemWithZCoordinates.json') as json_file:
    data = json.load(json_file)


start_time = data['StartTime']
stop_time = data['StopTime']
# loader_id, loading_rate, Mass Independent Loading Time
loaders = data['Loaders']
# dumping_id, unloadingRate, AverageSpeed, StartCoordinate- find closest node, Capacity
dumpers = data['Dumpers']
mass_transportation_jobs = data['MassTransportationJobs']
locked_actions = data['LockedActions']
spatial_graph_container = data['SpatialGraphContainer']


nodes = spatial_graph_container['Network']['Nodes']
edges = spatial_graph_container['Network']['Edges']
id = spatial_graph_container['Network']['id']
coordinate_system = spatial_graph_container['Network']['coordinate_system']
edge_coordinates = edges[0]['coordinates']


lol_pd = pd.DataFrame(edge_coordinates)
lol_pd.columns = ['Longitude', 'Latitude', 'Altitude']

origin_ = coordinate_system['origoInLongitudeLatitude']
# Change to (Latitude, Longitude)
origin = (float(origin_[1]), float(origin_[0]))

lol_pd, proj_info = rdm.add_meter_columns(lol_pd, origin)

# i = 1
# while i < len(edges):
#     for j, edge in enumerate(edges):
#         if j < i:
#             edge_coordinates = edge['coordinates']
#             lol_pd = pd.DataFrame(edge_coordinates)
#             lol_pd.columns = ['Longitude', 'Latitude', 'Altitude']
#             lol_pd, proj_info = rdm.add_meter_columns(lol_pd, origin)

#             plt.scatter(lol_pd['x'], lol_pd['y'])

#     i += 1

collect_nodes = {}
for node in nodes:
    lol_pd = pd.DataFrame([node['coordinates']])
    lol_pd.columns = ['Longitude', 'Latitude', 'Altitude']

    lol_pd, proj_info = rdm.add_meter_columns(lol_pd, origin)
    collect_nodes[node["id"]] = [lol_pd['x'], lol_pd['y']]
    plt.scatter(lol_pd['x'], lol_pd['y'])
    plt.annotate(f'{node["id"]}', (lol_pd['x'], lol_pd['y']))

plt.show()

for edge in edges:
    from_node = edge['Node1']['nodeId']
    to_node = edge['Node2']['nodeId']
    x1, y1 = collect_nodes[from_node]
    x2, y2 = collect_nodes[to_node]

    plt.plot([x1, x2], [y1, y2])

plt.show()


# rest_start_id = nodes[-1]['id'] + 1
# print(rest_start_id)


# for start_id in range(nodes[-1]['id'] + 1, int(3*nodes[-1]['id']/2)):
#     x = random.randint(2000, 6000)
#     y = random.randint(6000, 9000)

#     collect_nodes[start_id] = [x, y]

#     plt.scatter(x, y)
#     plt.annotate(f'{start_id}', (x, y))


# random_edges = np.sort(np.random.randint(rest_start_id, start_id, [10, 2]))
# unique_random_edges = np.unique(random_edges, axis=0)


# for i in range(nodes[-1]['id'] + 1, int(3*nodes[-1]['id']/2) - 1):
#     from_node = collect_nodes[i]
#     to_node = collect_nodes[i+1]

#     x1, y1 = from_node
#     x2, y2 = to_node

#     plt.plot([x1, x2], [y1, y2])

# for edge in unique_random_edges:
#     from_node = edge[0]
#     to_node = edge[1]

#     x1, y1 = collect_nodes[from_node]
#     x2, y2 = collect_nodes[to_node]

#     plt.plot([x1, x2], [y1, y2])


# plt.show()
