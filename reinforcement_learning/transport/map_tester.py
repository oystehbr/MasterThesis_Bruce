from IPython import embed
import maps
import matplotlib.pyplot as plt
from IPython import display


if __name__ == '__main__':
    nodes, edges, _, _, _ = maps.map5_helga(
        print_connections=True)

    embed(header='yeah')
    for node in nodes:
        x = int(node.get_coordinates()['x'])
        y = int(node.get_coordinates()['y'])

        l = plt.scatter(x, y)
        plt.annotate(f'{node}', (x, y))

    for edge_num in range(1, len(edges), 2):
        edge = edges[edge_num]
        from_node = edge.get_from_node()
        to_node = edge.get_to_node()

        x1, y1 = from_node.get_coordinates()[['x', 'y']].values[0]
        x2, y2 = to_node.get_coordinates()[['x', 'y']].values[0]

        plt.plot([x1, x2], [y1, y2])

    plt.show()
