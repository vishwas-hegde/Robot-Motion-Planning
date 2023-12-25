# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
import random
from scipy import spatial

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array  # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]  # map size
        self.size_col = map_array.shape[1]  # map size

        self.samples = []  # list of sampled points
        self.graph = nx.Graph()  # constructed graph
        self.path = []  # list of nodes of the found path

    def check_collision(self, p1, p2):
        # Bounding box check
        node1 = p1
        node2 = p2
        x_inc = (node2[0] - node1[0])/100
        y_inc = (node2[1] - node1[1])/100

        for i in range(0, 101):
            x = int(node1[0] + i * x_inc)
            y = int(node1[1] + i * y_inc)

            if (self.map_array[x][y]) == 0:
                return True
            try: # Avoid getting too close to obstacle as x and y are approximated to integer
                if ((self.map_array[x][y+1]) == 0 or (self.map_array[x][y-1]) == 0 or
                        (self.map_array[x+1][y]) == 0 or (self.map_array[x-1][y]) == 0):
                    return True
            except:
                continue
        return False

    def dis(self, point1, point2):
        """Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def uniform_sample(self, n_pts):
        """Use uniform sampling and store valid points
        arguments:
            n_pts - number of points to sample

        Check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        """
        # Initialize graph
        self.graph.clear()

        # Clear previous samples
        self.samples = []

        # Generate random row and column coordinates within the map boundaries
        rnd_row = np.random.uniform(0, self.size_row, n_pts)
        rnd_col = np.random.uniform(0, self.size_col, n_pts)

        for i in range(n_pts):
            row = int(rnd_row[i])
            col = int(rnd_col[i])

            if self.map_array[row][col] == 1:
                # If the randomly sampled point is valid (not in collision), add it to the list of sampled points
                self.samples.append((row, col))

    def gaussian_sample(self, n_pts):
        """Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        """
        # Initialize graph
        self.graph.clear()

        self.samples = []

        # Get the coordinates of obstacle cells
        obstacle_cells = np.argwhere(self.map_array == 0)
        std_dev = 4
        for _ in range(n_pts):
                # Sample a point using a Gaussian distribution around obstacles
            obstacle = obstacle_cells[int(np.random.uniform(0, len(obstacle_cells)))]
            new_point = np.random.normal(loc=obstacle, scale=std_dev, size=2)
            new_point = np.clip(new_point, 0, [self.size_row - 1, self.size_col - 1])
            new_point = (int(new_point[0]), int(new_point[1]))

            # Check if the sampled point is collision-free
            if self.map_array[new_point[0]][new_point[1]] == 1:
                self.samples.append(new_point)

    def bridge_sample(self, n_pts):
        """Use bridge sampling and store valid points
        arguments:
            n_pts - number of points to sample

        Check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        """
        # Clear previous samples
        self.samples = []

        self.graph.clear()
        obstacle_cells = np.argwhere(self.map_array == 0)
        for _ in range(n_pts):
            obstacle1 = obstacle_cells[int(np.random.uniform(0, len(obstacle_cells)))]
            obstacle2 = obstacle_cells[int(np.random.uniform(0, len(obstacle_cells)))]

            if obstacle2[0] != obstacle1[0] and obstacle2[1] != obstacle1[1] and self.dis(obstacle1, obstacle2) < 30:
                new_point = self.generate_bridge(obstacle2, obstacle1)

            # Check if the sampled point is collision-free
                if self.map_array[new_point[0]][new_point[1]] == 1:
                    self.samples.append(new_point)

    def generate_bridge(self, start, goal):
        return (int((start[0] + goal[0]) / 2), int((start[1] + goal[1]) / 2))

    def draw_map(self):
        """Visualization of the result"""
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict(zip(range(len(self.samples)), node_pos))
        pos["start"] = (self.samples[-2][1], self.samples[-2][0])
        pos["goal"] = (self.samples[-1][1], self.samples[-1][0])

        # draw constructed graph
        nx.draw(
            self.graph, pos, node_size=10, node_color="y", edge_color="y", ax=ax
        )

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(
                self.graph,
                pos=pos,
                nodelist=self.path,
                node_size=8,
                node_color="b",
            )
            nx.draw_networkx_edges(
                self.graph,
                pos=pos,
                edgelist=final_path_edge,
                width=2,
                edge_color="b",
            )

        # draw start and goal
        nx.draw_networkx_nodes(
            self.graph,
            pos=pos,
            nodelist=["start"],
            node_size=12,
            node_color="g",
        )
        nx.draw_networkx_nodes(
            self.graph, pos=pos, nodelist=["goal"], node_size=12, node_color="r"
        )

        # show image
        plt.axis("on")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()

    def sample(self, n_pts, sampling_method="uniform"):
        """Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        """
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []
        pairs = []
        max_weights = 30
        q_pairs = 120

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
            max_weights = 20
            self.start_goal_radius = 20
            q_pairs = 50
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
            self.start_goal_radius = 100
            max_weights = 22
            q_pairs = 150
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)
            self.start_goal_radius = 100
            max_weights = 30
            q_pairs = 130

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01),
        #          (p_id0, p_id2, weight_02),
        #          (p_id1, p_id2, weight_12) ...]

        # Iterate through all pairs of sampled points to connect them

        KDtree = spatial.KDTree(np.array(self.samples))
        KDtree_pairs = KDtree.query_pairs(q_pairs)
        for pair in KDtree_pairs:
            point1 = self.samples[pair[0]]
            point2 = self.samples[pair[1]]
            distance = self.dis(point1, point2)
            if distance > max_weights:
                continue
            if self.check_collision(point1, point2):
                continue
            pairs.append((pair[0], pair[1], distance))

        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01),
        #                                     (p_id0, p_id2, weight_02),
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.

        # Extract p_id values from the first two elements of each tuple in pairs
        p_ids = [p[0] for p in pairs] + [p[1] for p in pairs]
        unique_p_ids = list(set(p_ids))

        self.graph.add_nodes_from(unique_p_ids)
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print(
            "The constructed graph has %d nodes and %d edges"
            % (n_nodes, n_edges)
        )

    def search(self, start, goal):
        """Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal nodes, edges of them
        and their nearest neighbors to graph for
        self.graph to search for a path.
        """
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(["start", "goal"])

        start_pairs = []
        goal_pairs = []
        KDtree = spatial.KDTree(np.array(self.samples))
        start_neighbors = spatial.KDTree.query_ball_point(KDtree, start, self.start_goal_radius)
        goal_neighbors = spatial.KDTree.query_ball_point(KDtree, goal, self.start_goal_radius)

        for i in start_neighbors:
            if not self.check_collision(self.samples[i], start):
                distance = self.dis(self.samples[i], start)
                start_pairs.append(("start", i, distance))
        for i in goal_neighbors:
            if not self.check_collision(self.samples[i], goal):
                distance = self.dis(self.samples[i], goal)
                goal_pairs.append(("goal", i, distance))

        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)

        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(
                self.graph, "start", "goal"
            )
            path_length = (
                nx.algorithms.shortest_paths.weighted.dijkstra_path_length(
                    self.graph, "start", "goal"
                )
            )
            print("The path length is %.2f" % path_length)
            print(self.path)
        except nx.exception.NetworkXNoPath:
            print("No path found")

        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(["start", "goal"])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)




