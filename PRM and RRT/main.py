
from PIL import Image
import numpy as np
from RRT import RRT
from PRM import PRM
from PRM_Incremental import PRM_increment
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_map(file_path, resolution_scale):
    """Load map from an image and return a 2D binary numpy array
    where 0 represents obstacles and 1 represents free space
    """
    # Load the image with grayscale
    img = Image.open(file_path).convert("L")
    # Rescale the image
    size_x, size_y = img.size
    new_x, new_y = int(size_x * resolution_scale), int(
        size_y * resolution_scale
    )
    img = img.resize((new_x, new_y), Image.ANTIALIAS)

    map_array = np.asarray(img, dtype="uint8")

    # Get bianry image
    threshold = 127
    map_array = 1 * (map_array > threshold)

    # Result 2D numpy array
    return map_array


if __name__ == "__main__":
    # Load the map
    start = (200, 75)
    goal = (30, 250)
    map_array = load_map("WPI_map.jpg", 0.3)

    # Planning class
    PRM_planner = PRM(map_array)
    RRT_planner = RRT(map_array, start, goal)

    # # Search with PRM
    PRM_planner.sample(n_pts=1000, sampling_method="uniform")
    PRM_planner.search(start, goal)
    PRM_planner.sample(n_pts=2000, sampling_method="gaussian")
    PRM_planner.search(start, goal)
    PRM_planner.sample(n_pts=20000, sampling_method="bridge")
    PRM_planner.search(start, goal)

    # Search with RRT
    RRT_planner.RRT(n_pts=1000)


    # Incremental PRM


    # PRM_bonus = PRM_increment(map_array)
    # node_data = {}
    #
    # def table_create(nodes, path_length, path, case, sampling):
    #     data = {sampling:{
    #         'nodes': nodes,
    #         'path_length': path_length
    #         # 'path' : path
    #     }}
    #     node_data[f'Test case {case+1}'].append(data)
    #
    # for i in range(10):
    #     node_data[f'Test case {i+1}'] = []
    #     nodes, path_length, path = PRM_bonus.search(20000, start, goal, 'uniform')
    #     table_create(nodes, path_length, path, i, 'uniform')
    #     nodes, path_length, path = PRM_bonus.search(20000, start, goal, 'gaussian')
    #     table_create(nodes, path_length, path, i, 'gaussian')
    #     nodes, path_length, path = PRM_bonus.search(20000, start, goal, 'bridge')
    #     table_create(nodes, path_length, path, i, 'bridge')
    #     nodes, path_length, path = PRM_bonus.search(20000, start, goal, 'efficient')
    #     table_create(nodes, path_length, path, i, 'efficient')
    #
    # nodes_data = {}
    # for test_case, methods in node_data.items():
    #     nodes_data[test_case] = {}
    #     for method_data in methods:
    #         method_name, method_info = list(method_data.items())[0]
    #         nodes_data[test_case][method_name] = method_info['nodes']
    #
    # # Create a DataFrame for boxplot
    # df = pd.DataFrame.from_dict(nodes_data)
    # pd.set_option('display.max_rows', 10)  # Set the maximum number of rows to display
    # pd.set_option('display.max_columns', 20)  # Set the maximum number of columns to display
    # print(df)
    #
    # # Create boxplots using seaborn
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(12, 6))
    # sns.boxplot(data=df, palette="Set3")
    # plt.title("Boxplots of Nodes for Each Test Case")
    # plt.xlabel("Test Cases")
    # plt.ylabel("Nodes")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    #
    # # Show the boxplots
    # plt.show()

