Run Point_Robot.ipnb file for CVAE motion planning for Point Robot.

Additional Libraries Needed:
1) NetworkX : pip install networkx


Motion Planning scripts:
1) map_2d.py - reads and initializes the obstacles and map size.
2) planner.py - initializes the sampling method and visualizes the solution.
3) robot.py - initializes the robot and corresponding parameters.
4) PRM.py - used for probabilistic roadmaps to find solution.
5) RRT.py - used for rapidly exploring random trees to find solution.
6) sampling_method.py - parent class for PRM and RRT.
7) utils.py - utilities script.

Usage:
Run the cells in the Point_Robot.ipynb script to get the desired output.