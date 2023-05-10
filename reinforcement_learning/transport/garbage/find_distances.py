import time
from IPython import embed
import maps


def find_node_route(current_node, to_node, route, solutions):

    for next_node in current_node.get_all_connected_nodes():
        if to_node == next_node:
            finished_route = route.copy()
            finished_route.append(to_node)
            solutions.append(finished_route)
            continue

        if not (next_node in route):
            new_route = route.copy()
            new_route.append(next_node)
            new_current_node = next_node

            find_node_route(new_current_node, to_node,
                            new_route, solutions)


def main(from_node, to_node):
    solutions = []
    solutions_without_load_in_mid = []
    route = [from_node]
    a = time.time()
    find_node_route(from_node,
                    to_node, route, solutions)

    length_of_routes = []
    for sol in solutions:
        distance = 0
        load = 0
        for idx in range(len(sol) - 1):
            temp_edge = sol[idx].get_edge_to_node(sol[idx + 1])
            distance += temp_edge.get_distance()
            # if sol[idx].class_id == 'loading':
            #     load += 1

        # if load == 0:
        #     solutions_without_load_in_mid.append(sol)

        length_of_routes.append(distance)

    min_distance = min(length_of_routes)
    min_idx = length_of_routes.index(min_distance)
    best_route = solutions[min_idx]

    # Without running into another loading node
    # length_of_routes1 = []
    # for sol in solutions_without_load_in_mid:
    #     distance = 0
    #     load = 0
    #     for idx in range(len(sol) - 1):
    #         temp_edge = sol[idx].get_edge_to_node(sol[idx + 1])
    #         distance += temp_edge.get_distance()

    #         if sol[idx].class_id == 'loading':
    #             distance += 1000000

    #     length_of_routes1.append(distance)

    # min_distance1 = 0
    # best_route1 = []
    # if len(length_of_routes1) > 0:
    #     min_distance1 = min(length_of_routes1)
    #     min_idx1 = length_of_routes1.index(min_distance1)
    #     best_route1 = solutions_without_load_in_mid[min_idx1]

    return min_distance, best_route  # , min_distance1, best_route1


if __name__ == '__main__':
    nodes, edges, loaders, dumpers, nodes_dict = maps.map3(True)
    from_node_id = 38
    from_node = nodes_dict[from_node_id][-1]    # dumping
    to_node_ids = [43, 44, 45, 46, 47, 48]
    to_nodes = [nodes_dict[_id][-1] for _id in to_node_ids]  # loading

    embed(header='test')

    results = []
    for to_node in to_nodes:
        min_distance, best_route = main(
            from_node, to_node)

        print(
            f'\n\nFROM {from_node}, TO: {to_node}, DISTANCE: {min_distance: .2f}')  # or {min_distance1: .2f}')
        print('>> ', end=' ')
        for node in best_route:
            print(node, end=' ')

        # print('\n>> ', end=' ')
        # for node in best_route1:
        #     print(node, end=' ')

        # , min_distance1, best_route1])
        results.append([min_distance, best_route])
    print('\n\n')
