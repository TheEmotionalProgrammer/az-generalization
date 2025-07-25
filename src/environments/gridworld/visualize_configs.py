import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("src/")
from experiments.parameters import grid_env_descriptions


def visualize_gridworld(name, desc):
    """
    Visualize a GridWorld environment configuration with borders around each square.
    
    Args:
        desc (list of str): The description of the environment grid.
    """

    # Define a color map for visualization
    color_map = {
        'S': 'green',   # Start
        'F': 'skyblue', # Frozen
        'H': 'blue',   # Hole
        'G': 'gold'     # Goal
    }
    
    # Convert the grid description into a matrix of colors
    grid_size = len(desc)
    colors = np.zeros((grid_size, grid_size), dtype=object)
    for i, row in enumerate(desc):
        for j, cell in enumerate(row):
            colors[i, j] = color_map.get(cell, 'white')  # Default to white for unknown
    
    # Plot the grid
    fig, ax = plt.subplots(figsize=(grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            # Draw the cell with a border
            rect = plt.Rectangle((j, grid_size - i - 1), 1, 1, 
                                  facecolor=colors[i, j], edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

    # Set limits and remove ticks
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.title(f"GridWorld {name} Configuration")
    plt.show()


if __name__ == "__main__":
    map_size = 8
    CONFIG = "NARROW"
    visualize_gridworld(f"{map_size}x{map_size}_{CONFIG}", grid_env_descriptions[f"{map_size}x{map_size}_{CONFIG}"])