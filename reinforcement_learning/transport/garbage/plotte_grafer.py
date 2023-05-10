import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# def x_to_long(x, origin_lat, origin_long):
#     return origin_long+360*x/(40075*1000*np.cos(np.pi*origin_lat/180))


# def y_to_lat(y, origin_lat, origin_long):
#     return origin_lat+y/(111.32*1000)


# def plot_map_intersections(graph, origin_lat, origin_long, adjacent_nodes_info, node="all"):
#     # , mapit_dir='full_maps', filename='full_alg',intersection="all"):
#     """
#     Returns a map with plot of the graph
#     """
#     if node == "all":
#         mapit = folium.Map(
#             location=[origin_lat, origin_long], zoom_start=14, control_scale=True)
#     else:
#         long, lat = graph["Network"]["Nodes"][node]["coordinates"]
#         mapit = folium.Map(location=[lat, long],
#                            zoom_start=14, control_scale=True)

#     color = ["red", "green", "blue", "yellow", "black", "pink"]*30
#     d = pd.Series(adjacent_nodes_info.nodes.values,
#                   index=adjacent_nodes_info.edge_id.values).to_dict()
#     counter = 0
#     for t, k in enumerate(graph["Network"]["Edges"]):
#         if node in list(d.get(k["id"])) or (node == "all"):
#             counter += 1
#             for i in k["coordinates"]:
#                 p = [i[1], i[0]]
#                 folium.CircleMarker(
#                     p, fill_color=color[counter], color=color[counter], radius=3, popup=k["id"], weight=2).add_to(mapit)
#     for k in graph["Network"]["Nodes"]:
#         p = [k["coordinates"][1], k["coordinates"][0]]
#         folium.Marker(location=p, icon=folium.DivIcon(
#             html='<div style="font-size: 24pt; color : red">%s</div>' % k["id"],
#         )).add_to(mapit)
#         folium.CircleMarker(location=p, fill_color='red', color='red',
#                             radius=5, popup=k["id"], weight=2).add_to(mapit)
#     #save_obj_html(obj=mapit, directory=mapit_dir, filename=filename)
#     display(mapit)
#     return mapit


# def plot_raw(data, origin_lat, origin_long, sample=False, frac=1):
#     if sample:
#         print(sample)
#         data = data.sample(frac=frac)
#     print(data.shape)
#     mapit = folium.Map(location=[origin_lat, origin_long],
#                        zoom_start=14, control_scale=True)
#     for i, k in data.iterrows():
#         p = [k["Latitude"], k["Longitude"]]
#         folium.CircleMarker(location=p, color="black", fill_color='black',
#                             radius=1, popup=k["TripLogId"]).add_to(mapit)
#     display(mapit)
#     return mapit


# def plot_raw_and_intersections(data, intersections, origin_lat, origin_long, sample=False, frac=1):
#     if sample:
#         data = data.sample(frac=frac)
#     mapit = folium.Map(location=[origin_lat, origin_long],
#                        zoom_start=14, control_scale=True)
#     for i, k in data.iterrows():
#         p = [k["Latitude"], k["Longitude"]]
#         folium.CircleMarker(location=p, color="black", fill_color='black',
#                             radius=1, popup=k["TripLogId"]).add_to(mapit)
#     for i, k in intersections.iterrows():
#         n = k["id"]
#         p = [k["Latitude"], k["Longitude"]]
#         folium.CircleMarker(location=p, popup=n, radius=5,
#                             color="red", fill_color="red").add_to(mapit)
#         folium.Marker(location=p, icon=folium.DivIcon(
#             html='<div style="font-size: 24pt; color : red">%s</div>' % n,
#         )).add_to(mapit)

#     display(mapit)
#     return mapit


# def plot_raw_and_intersections_with_time(data, intersections, origin_lat, origin_long, sample=False, frac=1):
#     if sample:
#         data = data.sample(frac=frac)
#     mapit = folium.Map(location=[origin_lat, origin_long], zoom_start=14)
#     mint = data["timestamp_s"].min()
#     maxt = data["timestamp_s"].max()

#     colormap = cm.LinearColormap(colors=['green', 'blue'], index=[
#                                  mint, maxt], vmin=mint, vmax=maxt)
#     for i, k in data.iterrows():
#         p = [k["Latitude"], k["Longitude"]]
#         folium.CircleMarker(location=p, color=colormap(k["timestamp_s"]), fill_color=colormap(
#             k["timestamp_s"]), radius=1, popup=k["TripLogId"]).add_to(mapit)
#     for i, k in intersections.iterrows():
#         n = k["id"]
#         p = [k["Latitude"], k["Longitude"]]
#         folium.CircleMarker(location=p, popup=n, radius=5,
#                             color="red", fill_color="red").add_to(mapit)
#         folium.Marker(location=p, icon=folium.DivIcon(
#             html='<div style="font-size: 24pt; color : red">%s</div>' % n,
#         )).add_to(mapit)

#     display(mapit)


# def plot_xy_data(data, origin_lat, origin_long):
#     mapit = folium.Map(location=[origin_lat, origin_long],
#                        zoom_start=14, control_scale=True)
#     for i, k in data.iterrows():
#         p = [y_to_lat(k["y"], origin_lat, origin_long),
#              x_to_long(k["x"], origin_lat, origin_long)]
#         folium.CircleMarker(p, radius=3, weight=2).add_to(
#             mapit)
#     display(mapit)


# def plot_roads(trips, cluster_info, title=None, actual_intersections=None, save_name=None):
#     plt.figure(figsize=(15, 15))
#     plt.scatter(trips['y'], trips['x'], c='k', s=1)
#     plt.scatter(cluster_info['y'], cluster_info['x'],
#                 s=150, marker='x', c='red')
#     if actual_intersections is not None:
#         plt.scatter(
#             actual_intersections['y'], actual_intersections['x'], s=50, marker='o', c='dodgerblue')
#     if title is not None:
#         plt.title(title)
#     if save_name is not None:
#         plt.savefig(save_name + '.png')
#     plt.show()


# pl.plot_map_intersections(graph,proj_info["origin"][0],proj_info["origin"][1],adjacent_nodes_info,intersection=16);


def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False, node_dic=None,
                remove_self_loops=False):
    r"""MODIFIED FROM PYTORCH GEOMETRIC FUNCTION
    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

#    G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue
        u_tag, v_tag = node_dic[u], node_dic[v]
        G.add_edge(u_tag, v_tag)

        for key in edge_attrs:
            G[u_tag][v_tag][key] = round(values[key][i])

    # Assign node attributes
    for key in node_attrs:
        node_values_dic = {node_dic[i]: {
            key: data[key][i].item()} for i in range(len(node_dic))}
        nx.set_node_attributes(G, node_values_dic)

    return G


G = to_networkx(snapshot,
                node_attrs=['x'],
                edge_attrs=['edge_attr'],
                to_undirected=False, node_dic=nodes_dic_inv)

G.nodes.data()

coordinates = {'NO2': {'lat': 59.522901, 'lon': 7.842521},
               'NO1': {'lat': 60.538724, 'lon': 11.015993},
               'NO1': {'lat': 60.538724, 'lon': 11.015993},
               'SE3': {'lat': 59.029083, 'lon': 15.136678},
               'NO3': {'lat': 62.776586, 'lon': 9.544526},
               'NO5': {'lat': 60.959681, 'lon': 6.087871},
               'DK1': {'lat': 56.263598, 'lon': 9.362677},
               'NO4': {'lat': 65.581344, 'lon': 13.585454},
               'SE2': {'lat': 63.852089, 'lon': 16.228280},
               'SE1': {'lat': 66.002283, 'lon': 20.357983},
               'FI': {'lat': 62.933040, 'lon': 26.449689},
               'SE4': {'lat': 56.521626, 'lon': 14.789292},
               'DK2': {'lat': 55.535112, 'lon': 11.462506}}


fig, ax = plt.subplots(figsize=(20, 18))
cmap = plt.get_cmap('viridis')
pos = {x[0]: (x[1]['lon'], x[1]['lat']) for x in coordinates.items()}
alpha = 0.8

edge_labels = dict([((u, v,), f'{d["edge_attr"]}\n\n{G.edges[(v,u)]["edge_attr"]}')
                    for u, v, d in G.edges(data=True) if pos[u][0] > pos[v][0]])

node_values = [x[1]['x']*scaling for x in G.nodes.data()]

nx.draw(G, pos,
        node_size=2000, alpha=alpha,  # node size and alpha
        cmap=cmap, node_color=node_values,
        with_labels=True, font_weight="bold",  # Labels on nodes in bold
        arrows=True, connectionstyle='arc3, rad = 0.1')  # bending edges with arrows

nx.draw_networkx_edge_labels(G, pos, edge_labels)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
    vmin=min(node_values), vmax=max(node_values)))
sm._A = []
fig.colorbar(sm, alpha=alpha, label='Regulation')

#nx.draw_networkx_labels(G, pos, font_size=10)
# plt.show()
plt.savefig('graph.png', transparent=True)
