import sys

sys.path.append("src/")

import numpy as np
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from mcts_core.node import Node

def invert_grid_coordinate(coords, map_size):
    return coords[0] * map_size + coords[1] if coords is not None else None

def obs_to_cords(state, ncols):
    row, col = divmod(state, ncols)
    return row, col

def calculate_density(tree: Node, ncols, nrows):

    goal_coord = (ncols - 1, nrows - 1)

    visitation_counts = tree.state_visitation_counts()
    density = np.zeros((nrows, ncols))
    for state, count in visitation_counts.items():
        if state == None:
            row, col = goal_coord
        else:
            row, col = obs_to_cords(state, ncols)
        density[row, col] = count
    return density

def plot_density(density, root_state, obst_coords, ncols, nrows, cmap, ax=None):
    goal_coord = (ncols - 1, nrows - 1)

    for (row, col) in obst_coords:
        density[row, col] = np.nan  # remove numbers for obstacles
    density[density == 0.0] = np.nan  # mask zeros

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure  # <-- ensure we have a figure even if ax was passed in

    sns.heatmap(
        density,
        cmap=cmap,
        cbar=False,
        annot=True,
        fmt=".2f",
        mask=np.isnan(density),
        center=0,
        ax=ax,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 10 if ncols == 8 else 6},
    )

    # highlight start/goal/obstacles
    root_row, root_col = obs_to_cords(root_state, ncols)
    ax.add_patch(plt.Rectangle((root_col, root_row), 1, 1, fill=False, color="green", lw=5))
    ax.add_patch(plt.Rectangle(goal_coord, 1, 1, fill=False, color="goldenrod", lw=5))
    for (row, col) in obst_coords:
        ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=True, color="black", lw=0))

    # ticks/aspect
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_aspect("equal")

    # optional outer border around axes area
    pos = ax.get_position()
    fig.add_artist(plt.Rectangle((pos.x0, pos.y0), pos.width, pos.height, edgecolor="black", fill=False, lw=1))

    return ax