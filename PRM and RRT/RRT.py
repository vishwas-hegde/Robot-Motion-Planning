# Standard Algorithm Implementation
# Sampling-based Algorithms RRT
import math
import random
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np


# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row  # coordinate
        self.col = col  # coordinate
        self.parent = None  # parent node
        self.cost = 0.0  # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array  # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]  # map size
        self.size_col = map_array.shape[1]  # map size

        self.start = Node(start[0], start[1])  # start node
        self.goal = Node(goal[0], goal[1])  # goal node
        self.vertices = []  # list of nodes
        self.found = False  # found flag

    def init_map(self):
        """Intialize the map before each search"""
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    def dis(self, node1, node2):
        """Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        """
        # ### YOUR CODE HERE ###
        distance = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        return distance

    def check_collision(self, node1, node2):
        """Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        """
        # ### YOUR CODE HERE ###
        x_inc = (node2[0] - node1[0])/100
        y_inc = (node2[1] - node1[1])/100

        for i in range(0, 101):
            x = int(node1[0] + i * x_inc)
            y = int(node1[1] + i * y_inc)

            if (self.map_array[x][y]) == 0:
                return False
            try:       # Avoid getting too close to obstacle as x and y are approximated to integer
                if ((self.map_array[x][y+1]) == 0 or (self.map_array[x][y-1]) == 0 or
                        (self.map_array[x+1][y]) == 0 or (self.map_array[x-1][y]) == 0):
                    return False
            except:
                continue
        return True

    def get_new_point(self, goal_bias=0.05):
        """Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal
                        instead of a random point

        return:
            point - the new point
        """
        # ### YOUR CODE HERE ###
        x = np.random.random()
        if x <= goal_bias:
            row = self.goal.row
            col = self.goal.col
            return row, col
        else:
            row = int(np.random.uniform(0, self.size_row-1))
            col = int(np.random.uniform(0, self.size_col-1))

            if self.map_array[row][col] == 0:
                return None, None

        return row, col

    def get_new_node(self, rand_point, near_point, g_near=15):
        """
        Get new node based on the random point sampled
        arguments:
        rand_point: Random sampled point
        near_point: nearest node
        g_near: check if new_node is close to goal
        """

        v = np.array([rand_point[0] - near_point[0], rand_point[1] - near_point[1]])
        u_hat = v/np.linalg.norm(v)
        offset = 10 * u_hat

        x = int(near_point[0] + offset[0])
        y = int(near_point[1] + offset[1])
        if 0 <= x < len(self.map_array) and 0 <= y < len(self.map_array[0]):
            if self.map_array[x][y] == 1 and self.check_collision(near_point, (x,y)):
                if (self.goal.row - g_near <= x <= self.goal.row + g_near and
                        self.goal.col - g_near <= y <= self.goal.col + g_near):
                    return ((x,y), True)
                return ((x,y), False)
        return ((None, None), False)


    def get_nearest_node(self, point, neighbours):
        """Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        """
        # ### YOUR CODE HERE ###
        distances = {}
        for node in neighbours:
            d = self.dis(point, (node.row, node.col))
            distances[node] = d
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        first_key, first_value = next(iter(distances.items()))
        return first_key, first_value

    def get_neighbors(self, new_node, neighbor_size):
        """Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - list of neighbors that are within the neighbor distance
        """
        # ### YOUR CODE HERE ###

        samples = [[v.row, v.col] for v in self.vertices]
        kdtree = spatial.cKDTree(samples)
        ind = kdtree.query_ball_point([new_node[0], new_node[1]],
                                      neighbor_size)  # Use Query Ball Point Method to find nodes
        neighbors = [self.vertices[i] for i in ind]

        return neighbors

    def draw_map(self):
        """Visualization of the result"""
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            # print('plot nodes')
            # print(node.col)
            plt.plot(node.col, node.row, markersize= 3, marker="o", color="y")
            plt.plot(
                [node.col, node.parent.col],
                [node.row, node.parent.row],
                color="y",
            )

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot(
                    [cur.col, cur.parent.col],
                    [cur.row, cur.parent.row],
                    color="b",
                )
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker="o", color="b")

        # Draw start and goal
        plt.plot(
            self.start.col, self.start.row, markersize=5, marker="o", color="g"
        )
        plt.plot(
            self.goal.col, self.goal.row, markersize=5, marker="o", color="r"
        )

        # show image
        plt.show()

    def RRT(self, n_pts=1000):
        """RRT main search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        In each step, extend a new node if possible,
        and check if reached the goal
        """
        # Remove previous result
        self.init_map()

        # ### YOUR CODE HERE ###

        # In each step,
        # get a new point,
        # get its nearest node,
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.

        goal_bias = 0.05
        while len(self.vertices) != n_pts:
            row, col = self.get_new_point(goal_bias)    # Get a new point
            if row == None:                             # If None then attempts for a new point
                continue

            # Get the neighbour nodes
            neighbours = self.get_neighbors((row, col), 70)
            if len(neighbours) == 0:               # If no neighbours then get another point
                continue

            # Get parent node with a collision free edge
            parent_node, cost = self.get_nearest_node((row, col), neighbours)
            if parent_node == None:               # If no neighbour has collision free edge, look for a new point
                continue
            if cost <= 10:
                continue

            (row, col), goal_check = self.get_new_node((row, col), (parent_node.row, parent_node.col))
            if row == None:                                          # If None then attempts for a new point
                continue
            cost = self.dis((parent_node.row, parent_node.col), (row, col))
            # Create new node object
            # print(row, col)
            newnode = Node(row, col)
            newnode.parent = parent_node
            newnode.cost = parent_node.cost + cost   # Assign the cost to the new point
            self.vertices.append(newnode)

            if goal_check:  #If the new node is within goalbias then directly connect it to the goal node
                if not self.check_collision((row, col), (self.goal.row, self.goal.col)):
                    # If the edge between new node and goal collides then, continue the loop
                    continue
                self.goal.parent = newnode
                cost = self.dis((row,col), (self.goal.row, self.goal.col))
                self.goal.cost = newnode.cost + cost
                self.found = True
                break

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" % steps)
            print("The path length is %.2f" % length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
