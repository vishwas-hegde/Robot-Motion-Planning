import math

import numpy as np
from scipy import spatial

from sampling_method import SamplingMethod
from utils import angle_diff, wrap_to_pi


class Node:
    """Class for each tree node"""

    def __init__(self, config):
        self.config = config  # configuration
        self.parent = None  # parent node / edge
        self.cost = 0.0  # cost to parent / edge weight

class RRT(SamplingMethod):
    """RRT/RRT* class"""

    def __init__(self, sampling_method, n_configs, kdtree_d=10):
        """Specifying number of configs and sampling method to initialize
        arguments:
            sampling_method - name of the chosen sampling method
            n_configs - number of configs to sample
            kdtree_d - the distance of a node to be considered as a neighbor
        """
        super().__init__()
        self.sampling_method = sampling_method
        self.n_configs = n_configs
        self.kdtree_d = kdtree_d

        # kd tree spatial.KDTree([[]]), euclidean distance only
        self.kdtree = None
        self.samples = []  # list of sampled config nodes
        self.solution = []  # list of nodes of the found solution

        # Additional Class arguments for Informed RRT*
        self.irs = False
        self.c_best = []
        self.c_min = 0
        self.theta = 0

    def init_map(self):
        """Intialize the map before each search"""
        # Add start and goal nodes
        self.start_node = Node(self.start)
        self.goal_node = Node(self.goal)
        # Initialize the containers
        self.samples = []
        self.solution = []

        # Update samples and kdtree
        self.samples.append(self.start_node)
        self.update_kdtree()

    def update_kdtree(self):
        """Update the kd tree after new node is added"""
        self.kdtree = spatial.cKDTree(
            [node.config for node in self.samples]
        )

    def get_new_point(self, goal_bias):
        """Choose the goal or generate a random point in configuration space
        arguments:
            goal_bias - the possibility of choosing the goal
                        instead of a random point

        return:
            point - a new point in the configuration space
        """
        # Select goal
        if np.random.uniform() < goal_bias:
            point = self.goal
        # Or generate a random point
        else:
            point = []
            # sample in each dimension
            for i in range(self.robot.dof):
                point.append(
                    np.random.uniform(
                        self.robot.limits[i][0], 
                        self.robot.limits[i][1]
                    )
                )
        return point

    def get_nearest_node(self, point):
        """Find the nearest node in self.samples from the new point
        arguments:
            point - the new point in configuration space

        return:
            the nearest node in self.samples
        """
        # Use kdtree to find the neighbors within neighbor size
        _, ind = self.kdtree.query(point)
        return self.samples[ind]

    def extend(self, extension_d, goal_bias=0.05):
        """Extend a new node from the current tree
        arguments:
            extension_d - the extension distance
            goal_bias - the possibility of choosing the goal
                        instead of a random sample point

        Create and add a new node if valid.
        """
        # Generate a new point and find the nearest node
        new_point = self.get_new_point(goal_bias)
        nearest_node = self.get_nearest_node(new_point)

        # Calculate new node location by extending the nearest node
        # compute the direction to move
        diff = np.zeros(self.robot.dof)
        for i in range(self.robot.dof):
            # position
            if (self.robot.limits[i][2] != "r"):
                diff[i] = new_point[i] - nearest_node.config[i]
            # rotation
            else:
                # find the shortest angle
                diff[i] = angle_diff(
                    nearest_node.config[i], new_point[i], absolute=False
                )
        # get unit vector 
        if np.linalg.norm(diff) == 0:
            # same configuration
            return None

        new_config = (
            np.array(nearest_node.config) 
            + diff / np.linalg.norm(diff) * extension_d
        ).tolist()
        # wrap the angle if necessary
        for i in range(self.robot.dof):
            if self.robot.limits[i][2] == "r":
                new_config[i] = wrap_to_pi(new_config[i])

        if self.irs:
            a = self.c_best[-1] / 2
            b = (math.sqrt(self.c_best[-1] ** 2 - self.c_min ** 2)) / 2
            configs = self.robot.forward_kinematics(new_config)
            for pos in configs:
                x = pos[0]
                y = pos[1]
                c = (((x - self.u) * math.cos(self.theta) + (y - self.w) * math.sin(self.theta)) ** 2 / a ** 2) + (
                            ((x - self.u) * math.sin(self.theta) - (y - self.w) * math.cos(self.theta)) ** 2 / b ** 2)
                if c > 1:
                    return None

        # Check if the new configuration is valid
        if self.check_collision(new_config, nearest_node.config):
            return None

        # Create a new node
        new_node = Node(new_config)
        cost = self.robot.distance(new_config, nearest_node.config)
        # add the new node to the tree
        new_node.parent = nearest_node
        new_node.cost = cost
        self.samples.append(new_node)
        self.update_kdtree()

        return new_node

    def connect_to_near_goal(self, new_node):
        """Check if the new node is near and has a valid path to goal node
        If yes, connect the new node to the goal node
        """
        # Check if goal is close and there is a valid path
        dis = self.robot.distance(new_node.config, self.goal)
        if (
            dis < self.kdtree_d 
            and not self.check_collision(new_node.config, self.goal)
        ):
            # connect directly to the goal
            self.goal_node.cost = dis
            self.goal_node.parent = new_node
            self.samples.append(self.goal_node)
            self.update_kdtree()

    def get_path_and_cost(self, start_node, end_node):
        """Compute path cost starting from a start node to an end node
        arguments:
            start_node - path start node
            end_node - path end node

        return:
            cost - path cost
        """
        cost = 0
        curr_node = end_node
        path = [curr_node]

        # Keep tracing back
        # until finding the start_node
        # or no path exists
        while curr_node != start_node:
            cost += curr_node.cost
            parent = curr_node.parent
            if parent is None:
                print("There is no path from the given start to goal")
                return [], 0

            curr_node = parent
            path.append(curr_node)

        return path[::-1], cost

    def get_neighbors(self, new_node):
        """Get the neighbors within the neighbor distance from the node
        arguments:
            new_node - a new node
            size - the neighbor size

        return:
            neighbors - list of neighbors within the neighbor distance
        """
        ### YOUR CODE HERE ###
        ind = self.kdtree.query_ball_point(new_node.config, 1.5 * self.kdtree_d)
        neighbors = [self.samples[i] for i in ind]
        neighbors.remove(new_node)
        return neighbors

    def rewire(self, new_node, neighbors):
        """Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors of the new node

        Rewire the new node if connecting to a new neighbor node
        will give least cost.
        Rewire all the other neighbor nodes.
        """
        ### YOUR CODE HERE ###
        c_min = self.get_path_and_cost(self.start_node, new_node)[1]

        # Check for the least cost among the neighbors
        for i in neighbors:
            if self.check_collision(i.config, new_node.config):
                continue
            cost_to_newnode = self.robot.distance(i.config, new_node.config)
            cost = cost_to_newnode + self.get_path_and_cost(self.start_node, i)[1]
            if cost < c_min:
                c_min = cost
                new_node.parent = i
                new_node.cost = cost_to_newnode

        # Check for the least cost for the remaining neighbors with the newnode
        if new_node.parent in neighbors:
            neighbors.remove(new_node.parent)
        for i in neighbors:
            if self.check_collision(i.config, new_node.config):
                continue
            cost_to_newnode = self.robot.distance(i.config, new_node.config)
            c = cost_to_newnode + c_min
            if self.get_path_and_cost(self.start_node, i)[1] > c:
                i.parent = new_node
                i.cost = cost_to_newnode
        
    def RRT(self):
        """RRT search function
        In each step, extend a new node if possible,
        and check if reached the goal
        """
        # Start searching
        while len(self.samples) < self.n_configs:
            # Extend a new node until
            # all the points are sampled
            # or find a path
            new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)

            # If goal is not found, try to connect new node to goal
            if self.goal_node.parent is None:
                if new_node is not None:
                    self.connect_to_near_goal(new_node)
            # goal found
            else:
                break

        # Output
        if self.goal_node.parent is not None:
            num_nodes = len(self.samples)
            path, length = self.get_path_and_cost(
                self.start_node, self.goal_node
            )
            self.solution = [node.config for node in path]
            print("The constructed tree has %d of nodes" % num_nodes)
            print("The path length is %.2f" % length)
        else:
            print("No path found")
            length = 0

        return self.solution, length

    def RRT_star(self):
        """RRT* search function
        In each step, extend a new node if possible,
        and rewire the node and its neighbors.
        """
        # Start searching
        while len(self.samples) < self.n_configs:
            # Extend a new node
            new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)

            # Rewire the new node and its neighbors
            if new_node is not None:
                neighbors = self.get_neighbors(new_node)
                self.rewire(new_node, neighbors)

            # If goal is not found, try to connect new node to goal
            if self.goal_node.parent is None and new_node is not None:
                self.connect_to_near_goal(new_node)

        # Output
        if self.goal_node.parent is not None:
            num_nodes = len(self.samples)
            path, length = self.get_path_and_cost(
                self.start_node, self.goal_node
            )
            self.solution = [node.config for node in path]
            print("The constructed tree has %d of nodes" % num_nodes)
            print("The path length is %.2f" % length)
        else:
            print("No path found")
            length = 0

        return self.solution, length

    def Informed_RRT_star(self):
        '''Informed RRT* search function
        In each step, extend a new node if possible, and rewire the node and its neighbors
        Once a path is found, an ellipsoid will be defined to constrained the sampling area
        '''
        ### YOUR CODE HERE ###
        self.c_min = self.robot.distance(self.start, self.goal)
        x1, y1 = self.start
        x2, y2 = self.goal

        # Center of Ellipse
        self.u = (x1 + x2) / 2
        self.w = (y1 + y2) / 2
        # Tilt of ellipse
        delta_x = x2 - x1
        delta_y = y2 - y1
        self.theta = math.atan2(delta_y, delta_x)

        prev_cost = float('inf')
        i = 0

        def sample_informed():
            while len(self.samples) < self.n_configs:
                # Extend a new node
                new_node = self.extend(extension_d=self.kdtree_d, goal_bias=0.05)

                # Rewire the new node and its neighbors
                if new_node is not None:
                    neighbors = self.get_neighbors(new_node)
                    self.rewire(new_node, neighbors)

                # If goal is not found, try to connect new node to goal
                if self.goal_node.parent is None and new_node is not None:
                    self.connect_to_near_goal(new_node)

            if self.goal_node.parent is not None:
                path, length = self.get_path_and_cost(
                    self.start_node, self.goal_node
                )
                print(f'iteration {i}')
                p = [p.config for p in path]
                self.solution = p
                print(p)
                print(length)
                return length

        while i < 4:
            length = sample_informed()
            if i == 3:
                self.c_best.append(length)
                break
            if length == None:
                self.samples = []
                self.solution = []
                self.kdtree = None
                # Update samples and kdtree
                self.init_map()
                continue
            if length < prev_cost:
                self.irs = True
                i += 1
                self.c_best.append(length)
                prev_cost = length
                self.samples = []
                self.solution = []
                self.kdtree = None
                # Update samples and kdtree
                self.init_map()
                self.a = self.c_best[-1] / 2
                self.b = (math.sqrt(self.c_best[-1] ** 2 - self.c_min ** 2)) / 2
                print(f"Ellipse a = {self.a} and b = {self.b}")

        # print(self.c_best)

        # Output
        # if self.goal_node.parent is not None:
        #     num_nodes = len(self.samples)
        #     path, length = self.get_path_and_cost(
        #         self.start_node, self.goal_node
        #     )
        #     self.solution = [node.config for node in path]
        #
        #     print("The constructed tree has %d of nodes" % num_nodes)
        #     print("The path length is %.2f" % length)
        # else:
        #     print("No path found")

        return [self.solution, (self.u, self.w), 2*self.a, 2*self.b, self.theta*180/np.pi]


    def plan(self, start, goal):
        """Search for a path in graph given start and goal location

        arguments:
            start - start configuration
            goal - goal configuration
        """
        self.start = start
        self.goal = goal
        self.init_map()
        length = 0
        if self.sampling_method == "RRT":
            self.solution = self.RRT()
        elif self.sampling_method == "RRT_star":
            self.solution, length = self.RRT_star()
        elif self.sampling_method == "Informed_RRT_star":
            self.solution = self.Informed_RRT_star()
        else: 
            raise ValueError(f"Sampling method:{self.sampling_method} does not exist!")

        return self.solution, length

    def visualize_sampling_result(self, ax):
        """ Visualization the sampling result."""
        # Draw Trees / sample points
        for node in self.samples:
            # node
            pos1 = self.robot.forward_kinematics(node.config)[-1]
            ax.plot(
                pos1[0], 
                pos1[1], 
                markersize=5, 
                marker=".", 
                color="g", 
                alpha=0.3
            )
            # edge
            if node == self.start_node:
                continue
            pos2 = self.robot.forward_kinematics(node.parent.config)[-1]
            ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color="y",
                alpha=0.3
            )
