import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from utils import endpoints_to_edges


class Map2D():
    """A 2D map class in the continuous space (x, y).
    The map has a rectangular shape and a list of obstacles.
    """
    def __init__(self, obs):
        map_2d = self.load_map(obs)
        # (x_lim, y_lim)
        self.shape = map_2d[0]
        # four corners of the map
        self.corners = map_2d[1]
        # list of obstacles
        self.obstacles = map_2d[2]
        # list of obstacle edges including map boundary
        self.obstacle_edges = map_2d[3]  

    def load_map(self, obs):
        """Load a map in the continuous space with shape (x_lim, y_lim), and
        each obstacle is defined as a list of points [(x1, y1), (x2, y2), ...]
        """
        # Map container
        shape = [0, 0]
        corners = []
        obstacles = obs
        obstacle_edges = []

        # Load from the csv file
        # the first row is the map size
        # the rest of the rows are obstacle vertices
 
        shape[0] = 1
        shape[1] = 1

   
        
            # obstacle = []
            # # load (x, y) as obstacle vertices
            # for j in range(0, len(row), 2):
            #     if row[j] == '' or row[j + 1] == '':
            #         break
            #     point = (float(row[j]), float(row[j + 1]))
            #     obstacle.append(point)
            # obstacles.append(obstacle)

        # Set the map corners
        corners = [
            [0, 0], 
            [shape[0], 0],
            [shape[0], shape[1]], 
            [0, shape[1]]
        ]

        # Build the edges of obstacles
        # including obstacle edges and map boundary edges

        # a. map boundary edges
        corners = [
            (0, 0), 
            (shape[0], 0), 
            (shape[0], shape[1]), 
            (0, shape[1])
        ]
        boundary_edges = endpoints_to_edges(corners, closed=True)
        obstacle_edges.extend(boundary_edges)

        # b. convert obstacles to obstacle edges
        for obstacle in obstacles:
            each_edges = endpoints_to_edges(obstacle, closed=True)
            obstacle_edges.extend(each_edges)

        return shape, corners, obstacles, obstacle_edges

    def visualize_map(self, return_ax=False):
        """Visualize the map."""
        # Create empty map
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)

        # Set the axis limits based on the map size
        ax.set_xlim(0, self.shape[0])
        ax.set_ylim(0, self.shape[1])
        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Map Visualization')

        # Draw each obstacle as a polygon
        for obstacle in self.obstacles:
            polygon = Polygon(
                obstacle, closed=True, edgecolor='black', facecolor='gray'
            )
            ax.add_patch(polygon)

        if return_ax:
            return ax, fig
        else:
            plt.show()
