import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as manimation

from utils import endpoints_to_edges, angle_diff, interpolate_angle
from utils import is_in_polygon, is_intersecting


class Robot:
    """A parent class for all robots"""

    def __init__(self, limits):
        """Initialize by providing limits of each degree of freedom"""
        # Limits in each dof, each limit is defined as (lower, upper, name)
        self.limits = limits
        self.dof = len(limits)

    def forward_kinematics(self, config):
        """Compute the endpoints of the robot given a configuration
        The last endpoint would be used for visualization of the sampling
        """
        raise NotImplementedError

    def get_edges(self):
        """Return the edges of the robot for collision checking"""
        raise NotImplementedError

    def distance(self, config1, config2):
        """Compute the distance between two configurations"""
        raise NotImplementedError

    def interpolate(self, config1, config2, num):
        """Interpolate between two configurations"""
        raise NotImplementedError

    def check_collision(
        self, config1, config2, map_corners, obstacles, obstacle_edges
    ):
        """Check colliding with obstacles between two configurations
        First perform an interpolation between the two configurations,
        then check if any of the interpolated configurations hit obstacles.
       
        arguments:
            config1 - configuration 1
            config2 - configuration 2
            map_corners - corners of the map
            obstacles - list of obstacles
            obstacle_edges - list of edges of obstacles, including map edges
        
        return:
            True if colliding with obstacles between the two configurations
        """
        # Get intepolated configurations in between config1 and config2
        configs_between = self.interpolate(config1, config2, num=10)

        # check if any of these configurations hit obstacles
        for config in configs_between:
            if self.check_collision_config(
                config, map_corners, obstacles, obstacle_edges
            ):
                return True
        return False

    def check_collision_config(
        self, config, map_corners, obstacles, obstacle_edges
    ):
        """Check if a configuration is colliding with obstacles. Ensure that all  
        cases are checked. Even ones that might not be present in the given map. 
        arguments:
            config - configuration of the robot
            map_corners - corners of the map
            obstacles - list of obstacles
            obstacle_edges - list of edges of obstacles, including map edges
        
        return:
            True if colliding with obstacles
        """
        # Get the edges of the robot for collision checking
        robot_endpoint = self.forward_kinematics(config)[-1]
        robot_edges = self.get_edges(config)

        # Check if the robot endpoint is outside the map
        if not is_in_polygon(robot_endpoint, map_corners):
            return True

        # Check if the robot endpoint is inside any obstacle
        for obstacle in obstacles:
            if is_in_polygon(robot_endpoint, obstacle):
                return True

        ### YOUR CODE HERE ###
        if len(config) == 5: # Checking is the omnidir robot endpoints collide with obstacles
            for c in config[:4]:
                for obstacle in obstacles:
                    if is_in_polygon(c, obstacle):
                        return True

        if is_intersecting(obstacle_edges, robot_edges): # Checking for Edge intersections
            return True

        return False


    def draw_robot(self, ax, config, edgecolor="b", facecolor="g"):
        """Draw the robot given a configuration on a matplotlib axis.
        This is for visualization purpose only.
        """
        raise NotImplementedError


class PointRobot(Robot):
    """2D Point robot class"""

    def __init__(self):
        """Initialize the robot with no limits in x, y (0, 1000))"""
        super().__init__(limits=[
            (0, 1000, "x"),
            (0, 1000, "y")
        ])

    def forward_kinematics(self, config):
        """Simply return the configuration as the endpoint"""
        return [config]

    def get_edges(self, config):
        """Simply return an empty list"""
        return []

    def distance(self, config1, config2):
        """Euclidean distance"""
        x_diff = config1[0] - config2[0]
        y_diff = config1[1] - config2[1]
        return np.sqrt(x_diff**2 + y_diff**2)

    def interpolate(self, config1, config2, num=5):
        """Interpolate between two configurations"""
        configs_between = zip(
            np.linspace(config1[0], config2[0], num),
            np.linspace(config1[1], config2[1], num)
        )
        return configs_between

    def draw_robot(self, ax, config, edgecolor="b", facecolor="g"):
        ax.scatter(config[0], config[1], s=20, c=edgecolor)


class OmnidirectionalRobot(Robot):
    """Omnidirectional navigation robot class
    Its shape is defined as a rectangle with a width and a height.
    The robot could move in any direction with any angle in a 2D plane.
    """

    def __init__(self, width, height):
        """Initialize the robot with a width and height."""
        self.width = width
        self.height = height
        # Limits in each dof: (x, y, theta)
        # x, y have no limits unless bounded by the map (1000 as default)
        # theta range is (-pi, pi)
        super().__init__(limits=[
            (0, 1000, "x"),
            (0, 1000, "y"),
            (-np.pi, np.pi, "r")
        ])

    def forward_kinematics(self, config):
        """Compute the 4 corner coordinates of the robot given a configuration
        Also attach the center of the robot as the last endpoint.
        The last endpoint would be used for visualization of the sampling.
        arguments:
            config: [x, y, theta] of the rectangle

        return:
            endpoints: 4 corner coordinates of the rectangle and its center
                       [corner1, corner2, corner3, corner4, center]
        """
        # Check configuration shape
        assert len(config) == 3, "Configuration should be (x, y, theta)"
        # print(config)
        x, y, theta = config
        endpoints = np.zeros((5, 2))

        ### YOUR CODE HERE ###
        length = self.height # Length of the rectangle
        width = self.width  # Width of the rectangle

        # Calculate the coordinates of the four corners of the rectangle
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Corner 1
        endpoints[0, 0] = x + 0.5 * length * cos_theta + 0.5 * width * sin_theta
        endpoints[0, 1] = y + 0.5 * length * sin_theta - 0.5 * width * cos_theta

        # Corner 2
        endpoints[1, 0] = x + 0.5 * length * cos_theta - 0.5 * width * sin_theta
        endpoints[1, 1] = y + 0.5 * length * sin_theta + 0.5 * width * cos_theta

        # Corner 3
        endpoints[2, 0] = x - 0.5 * length * cos_theta - 0.5 * width * sin_theta
        endpoints[2, 1] = y - 0.5 * length * sin_theta + 0.5 * width * cos_theta

        # Corner 4
        endpoints[3, 0] = x - 0.5 * length * cos_theta + 0.5 * width * sin_theta
        endpoints[3, 1] = y - 0.5 * length * sin_theta - 0.5 * width * cos_theta

        # Center
        endpoints[4, 0] = x
        endpoints[4, 1] = y

        return endpoints

    def get_edges(self, config):
        """Compute the edges of the robot given a configuration"""
        # Get the 4 corner coordinates

        ### YOUR CODE HERE ###
        robot_endpoints = self.forward_kinematics(config)
        robot_edges = [
            (tuple(robot_endpoints[0]), tuple(robot_endpoints[1])),
            (tuple(robot_endpoints[1]), tuple(robot_endpoints[2])),
            (tuple(robot_endpoints[2]), tuple(robot_endpoints[3])),
            (tuple(robot_endpoints[3]), tuple(robot_endpoints[0]))
        ]
        return robot_edges

    def distance(self, config1, config2):
        """Calculate the euclidean distance between two configurations
        arguments:
            p1 - config1, [x, y, theta]
            p2 - config2, [x, y, theta]

        return:
            distance in R^2 x S^1 space
        """

        ### YOUR CODE HERE ###
        x1, y1, theta1 = config1
        x2, y2, theta2 = config2

        # Calculate the difference in S¹ (theta)
        delta_theta = min(abs(theta1 - theta2), 2*np.pi - abs(theta1 - theta2))

        # distances in R² and S¹ space
        distance_r2_s1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + delta_theta ** 2)

        return distance_r2_s1
    
    def interpolate(self, config1, config2, num=5):
        """Interpolate between two configurations
        arguments:
            p1 - config1, [x, y, theta]
            p2 - config2, [x, y, theta]
        return:
            list with num number of configs from linear interploation in R^2 x S^1 space
        """

        ### YOUR CODE HERE ###
        # Extract the individual components of the configurations
        x1, y1, theta1 = config1
        x2, y2, theta2 = config2

        # Perform linear interpolation for each dimension
        x_interp = np.linspace(x1, x2, num)
        y_interp = np.linspace(y1, y2, num)
        theta_interp = interpolate_angle(theta1, theta2, num)

        # Combine the interpolated components to create interpolated configurations
        interpolated_configs = [[x, y, theta] for x, y, theta in zip(x_interp, y_interp, theta_interp)]

        return interpolated_configs

    def draw_robot(self, ax, config, edgecolor="b", facecolor="pink"):
        # compute corners and draw rectangle
        corners = self.forward_kinematics(config)[:4]
        polygon = Polygon(
            corners, closed=True, edgecolor=edgecolor, facecolor=facecolor
        )
        ax.add_patch(polygon)

class KinematicChain(Robot):
    """Kinematic chain robot class
    A planar robot with a fixed base and pure revolute joints.
    Each link is a line segment.
    """

    def __init__(self, link_lengths, base=[0.1, 0.1]):
        """Initialize with a list of link lengths, and a fixed base."""
        self.base = base
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        # Limits in each dof
        # assume all to be (-pi, pi)
        super().__init__(limits=[
            (-np.pi, np.pi, "r") for _ in range(self.num_joints)
        ])

    def forward_kinematics(self, config):
        """Compute the joint coordinates given a configuration of joint angles.
        The last endpoint would be used for visualization of the sampling
        arguments:
            config: A list of joint angles in radians.

        return:
            edges: A list of joint coordinates.
        """
        # Initialize the starting point as the fixed base
        joint_positions = [self.base]
        start_point = np.array(self.base)
        angle = 0

        # Compute the end points of each joint based on the configuration
        ### YOUR CODE HERE ###

        x, y = start_point[0], start_point[1]  # Initial joint position

        for l, conf_angle in zip(self.link_lengths, config):
            angle += conf_angle
            x += l * math.cos(angle)
            y += l * math.sin(angle)

            joint_positions.append([x, y])

        return joint_positions


    def get_edges(self, config):
        """Compute the link line segments of the robot given a configuration.
        arguments:
            config: A list of joint angles in radians.

        return:
            edges: A list of line segments representing the link line segments.
        """
        # Check configuration length
        assert (
            len(config) == self.num_joints
        ), "Configuration should match the number of joints"

        ### YOUR CODE HERE ###
        edges = []
        joints = self.forward_kinematics(config)
        for i in range(1, len(joints)):
            edge = [joints[i - 1], joints[i]]
            edges.append(edge)

        return edges

    def distance(self, config1, config2):
        """Calculate the euclidean distance between two configurations
        arguments:
            p1 - config1, [joint1, joint2, joint3, ..., jointn]
            p2 - config2, [joint1, joint2, joint3, ..., jointn]

        return:
            A Euclidean distance in S^1 x S^1 x ... x S^1 space
        """
        ### YOUR CODE HERE ###
        avg_dist = 0
        for joint1, joint2 in zip(config1, config2):
            dist = abs(joint1 - joint2)
            avg_dist += dist

        return avg_dist

    def interpolate(self, config1, config2, num=10):
        """Interpolate between two configurations
        arguments:
            p1 - config1, [joint1, joint2, joint3, ..., jointn]
            p2 - config2, [joint1, joint2, joint3, ..., jointn]

        return:
            A Euclidean distance in 
            list with num number of configs from linear interploation in S^1 x S^1 x ... x S^1 space.
        """

        ### YOUR CODE HERE ###
        interpolated_configs = [config1]

        # Interpolate between config1 and config2
        for i in range(1, num):
            interpolated_step = [interpolate_angle(j1, j2, num)[i] for j1, j2 in zip(config1, config2)]
            interpolated_configs.append(interpolated_step)

        return interpolated_configs

    def draw_robot(self, ax, config, edgecolor="b", facecolor="black"):
        # compute joint positions and draw lines
        positions = self.forward_kinematics(config)
        # Draw lines between each joint
        for i in range(len(positions) - 1):
            line = np.array([positions[i], positions[i + 1]])
            ax.plot(line[:, 0], line[:, 1], color=edgecolor)
        # Draw joint
        for i in range(len(positions)):
            ax.scatter(positions[i][0], positions[i][1], s=5, c=facecolor)
