import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

map_size = 8

dict_configs = {
        "DEFAULT": f"SPARSE {map_size}x{map_size}",
        "NARROW": f"NARROW {map_size}x{map_size}",
        "SLALOM": f"SLALOM {map_size}x{map_size}",
        "MAZE_LR": "MAZE_LR",
        "MAZE_RL": "MAZE_RL",
        "MAZE_LL": "MAZE_LL",
        "MAZE_RR": "MAZE_RR",
        "RANDOM_HORIZONTAL": "SPARSE",
        "OBS_HORIZONTAL": "HORIZONTAL",
    }

def plot_comparison_subplots(filepaths_dict, labels=None, map_size=16, max_episode_length=100):
    """
    Plots 3 subplots comparing the same metrics across different CONFIG settings.

    Args:
        filepaths_dict (dict): Dictionary where keys are CONFIG labels, and values are lists of file paths.
        labels (list of str, optional): Labels for each file in the filepaths list.
    """

    metric = "Discounted Return"
    

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

    # Define a colormap and generate colors for the labels
    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)  # 6 colors for 6 labels
    colors = [colormap(idx) for idx in color_indices]

    for ax, (config_label, filepaths) in zip(axes, filepaths_dict.items()):

        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))

        dataframes = [pd.read_csv(filepath) for filepath in filepaths]

        for df, label in zip(dataframes, labels):

            # if "c=0.0" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "c=0.1" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "c=0.5" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "c=1.0" in label:
            #     linestyle = "-"
            #     color = colors[3]
            # elif "c=2.0" in label:
            #     linestyle = "-"
            #     color = colors[4]
            # elif "c=100.0" in label:
            #     linestyle = "-"
            #     color = colors[5]
            # else:
            #     linestyle = "-"
            #     color = None

            # if "c=0.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "c=0.1" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "c=0.0" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "c=0.1" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[3]

            # if "Beta=1.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[0]
            # elif "Beta=10.0" in label and "+UCT" in label:
            #     linestyle = "-"
            #     color = colors[1]
            # elif "Beta=1.0" in label and "PUCT" in label:
            #     linestyle = "-"
            #     color = colors[2]
            # elif "Beta=10.0" in label and "PUCT" in label:
            #     color = colors[3]

            if "β=1.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[0]
            elif "β=10.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[2]
            elif "β=100.0" in label and "c=0.0" in label:
                linestyle = "-"
                color = colors[5]
            elif "β=1.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[0]
            elif "β=10.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[2]
            elif "β=100.0" in label and "c=0.1" in label:
                linestyle = "--"
                color = colors[5]


            # Exclude the last budget value and corresponding statistics
            budgets = df["Budget"][:-1]
            means = df[f"{metric} mean"][:-1]
            ses = df[f"{metric} SE"][:-1]

            ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels(budgets, fontsize=12)
        ax.tick_params(axis='y', labelsize=12) 
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(dict_configs[config_label], fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)

        if ax == axes[0]:
            ax.set_ylabel(metric, fontsize=14)


    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(labels), fontsize=12)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.03)
    plt.savefig("comparison_subplots.png")
    plt.show()


def plot_comparison_subplots_2x3(filepaths_dict, labels=None, map_size=16, max_episode_length=100, selpols=("PUCT", "UCT"), train_config=None):
    """
    Plots 2x3 subplots comparing the same metrics across different CONFIG settings and SELPOLs.

    Args:
        filepaths_dict (dict): Nested dict: {SELPOL: {CONFIG: [filepaths]}}
        labels (list of str, optional): Labels for each file in the filepaths list.
        selpols (tuple): Tuple of SELPOL names, e.g., ("PUCT", "UCT")
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)
    colors = [colormap(idx) for idx in color_indices]

    for row, selpol in enumerate(selpols):
        for col, (config_label, filepaths) in enumerate(filepaths_dict[selpol].items()):
            ax = axes[row, col]
            optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))
            if config_label == "MAZE_RL":
                optimal_value = 0.95 ** 20
            dataframes = [pd.read_csv(filepath) for filepath in filepaths]

            for df, label in zip(dataframes, labels):
                if "c=0.0" in label:
                    linestyle = "-"
                    color = colors[0]
                elif "c=0.1" in label:
                    linestyle = "-"
                    color = colors[1]
                elif "c=0.5" in label:
                    linestyle = "-"
                    color = colors[2]
                elif "c=1.0" in label:
                    linestyle = "-"
                    color = colors[3]
                elif "c=2.0" in label:
                    linestyle = "-"
                    color = colors[4]
                elif "c=100.0" in label:
                    linestyle = "-"
                    color = colors[5]
                else:
                    linestyle = "-"
                    color = None
                # if "β=1.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[0]
                # elif "β=10.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[2]
                # elif "β=100.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[5]
                # elif "β=1.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[0]
                # elif "β=10.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[2]
                # elif "β=100.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[5]

                # Exclude the last budget value and corresponding statistics
                budgets = df["Budget"][:-1]
                means = df[f"{metric} mean"][:-1]
                ses = df[f"{metric} SE"][:-1]

                ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
                ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)

            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.set_xticklabels(budgets, fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
            if train_config is None:
                ax.set_title(f"{selpol} - {dict_configs[config_label]}", fontsize=14)
            else:
                ax.set_title(f"{selpol} - {train_config} → {dict_configs[config_label]}", fontsize=14)
                
            ax.grid(True, linestyle="--", linewidth=0.5)
            if col == 0:
                ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_), fontsize=12)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig("comparison_subplots_2x3.png")
    plt.show()

def plot_comparison_subplots_2x3_parking(filepaths_dict, labels=None, max_episode_length=200, selpols=("PUCT", "UCT"), train_config=None):
    """
    Plots 2x3 subplots comparing the same metrics across different CONFIG settings and SELPOLs.

    Args:
        filepaths_dict (dict): Nested dict: {SELPOL: {CONFIG: [filepaths]}}
        labels (list of str, optional): Labels for each file in the filepaths list.
        selpols (tuple): Tuple of SELPOL names, e.g., ("PUCT", "UCT")
    """
    metric = "Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)
    colors = [colormap(idx) for idx in color_indices]

    for row, selpol in enumerate(selpols):
        for col, (config_label, filepaths) in enumerate(filepaths_dict[selpol].items()):
            ax = axes[row, col]
            optimal_value = 1#0.95 ** ((map_size * 2 - 2)) if config_label != "SLALOM" else 0.95 ** ((map_size * 2 - 2 + (4 if map_size == 8 else 6)))
            dataframes = [pd.read_csv(filepath) for filepath in filepaths]

            for df, label in zip(dataframes, labels):
                if "PDDP" in label:
                    color = "purple"
                    linestyle = "-"
                elif "EDP" in label:
                    color = "black"
                    linestyle = "-"
                # elif "c=0.0" in label:
                #     linestyle = "--"
                #     color = colors[0]
                # elif "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[1]
                # elif "c=0.5" in label:
                #     linestyle = "--"
                #     color = colors[2]
                # elif "c=1.0" in label:
                #     linestyle = "--"
                #     color = colors[3]
                # elif "c=2.0" in label:
                #     linestyle = "--"
                #     color = colors[4]
                # elif "c=100.0" in label:
                #     linestyle = "--"
                #     color = colors[5]
                # else:
                #     linestyle = "-"
                #     color = None
                elif "β=1.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[0]
                elif "β=10.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[2]
                elif "β=100.0" in label and "c=0.0" in label:
                    linestyle = "-"
                    color = colors[5]
                elif "β=1.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[0]
                elif "β=10.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[2]
                elif "β=100.0" in label and "c=0.1" in label:
                    linestyle = "--"
                    color = colors[5]

                # Exclude the last budget value and corresponding statistics
                budgets = df["Budget"][:-1]
                means = df[f"{metric} mean"][:-1]
                ses = df[f"{metric} SE"][:-1]

                ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
                ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)

            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.set_xticklabels(budgets, fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
            if train_config is None:
                ax.set_title(f"{selpol} - {dict_configs[config_label]}", fontsize=14)
            else:
                ax.set_title(f"{selpol} - {train_config} → {dict_configs[config_label]}", fontsize=14)
                
            ax.grid(True, linestyle="--", linewidth=0.5)
            if col == 0:
                ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_), fontsize=12)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig("comparison_subplots_2x3.png")
    plt.show()

def plot_comparison_subplots_2x2_parking(filepaths_dict, labels=None, max_episode_length=200, selpols=("PUCT", "UCT"), train_config=None):
    """
    Plots 2x2 subplots comparing the same metrics across different CONFIG settings and SELPOLs.

    Args:
        filepaths_dict (dict): Nested dict: {SELPOL: {CONFIG: [filepaths]}}
        labels (list of str, optional): Labels for each file in the filepaths list.
        selpols (tuple): Tuple of SELPOL names, e.g., ("PUCT", "UCT")
    """
    metric = "Return"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)  # Wider figure

    colormap = cm.rainbow
    color_indices = np.linspace(0, 1, 6)
    colors = [colormap(idx) for idx in color_indices]

    for row, selpol in enumerate(selpols):
        for col, (config_label, filepaths) in enumerate(filepaths_dict[selpol].items()):
            ax = axes[row, col]
            optimal_value = 1  # Set as needed
            dataframes = [pd.read_csv(filepath) for filepath in filepaths]

            for df, label in zip(dataframes, labels):
                if "PDDP" in label:
                    color = "purple"
                    linestyle = "-"
                elif "EDP" in label:
                    color = "black"
                    linestyle = "-"
                # elif "β=1.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[0]
                # elif "β=10.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[2]
                # elif "β=100.0" in label and "c=0.0" in label:
                #     linestyle = "-"
                #     color = colors[5]
                # elif "β=1.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[0]
                # elif "β=10.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[2]
                # elif "β=100.0" in label and "c=0.1" in label:
                #     linestyle = "--"
                #     color = colors[5]
                # else:
                #     linestyle = "-"
                #     color = None

                elif "c=0.0" in label:
                    linestyle = "-"
                    color = colors[0]
                elif "c=0.1" in label:
                    linestyle = "-"
                    color = colors[1]
                elif "c=0.5" in label:
                    linestyle = "-"
                    color = colors[2]
                elif "c=1.0" in label:
                    linestyle = "-"
                    color = colors[3]
                elif "c=2.0" in label:
                    linestyle = "-"
                    color = colors[4]
                elif "c=100.0" in label:
                    linestyle = "-"
                    color = colors[5]
                else:
                    linestyle = "-"
                    color = None

                # Exclude the last budget value and corresponding statistics
                budgets = df["Budget"][:-1]
                means = df[f"{metric} mean"][:-1]
                ses = df[f"{metric} SE"][:-1]

                ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
                ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)

            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.set_xticklabels(budgets, fontsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
            if train_config is None:
                ax.set_title(f"{selpol} - {dict_configs[config_label]}", fontsize=14)
            else:
                ax.set_title(f"{selpol} - {train_config} → {dict_configs[config_label]}", fontsize=14)

            ax.grid(True, linestyle="--", linewidth=0.5)
            if col == 0:
                ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_), fontsize=12)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.15, hspace=0.3)
    plt.savefig("comparison_subplots_2x2.png")
    plt.show()

def plot_comparison_subplots_1x2(files_list, labels, colors, linestyles, metric="Return"):
    """
    Plots a 1x2 subplot: first is SPARSE, second is HORIZONTAL.
    Args:
        files_list (list of list of str): [[files_for_sparse], [files_for_horizontal]]
        labels (list of str): Labels for each line (must match files in each subplot).
        colors (list of str or tuple): Colors for each line (must match files in each subplot).
        linestyles (list of str): Linestyles for each line (must match files in each subplot).
        metric (str): Metric to plot (default: "Return").
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    config_names = ["SPARSE", "HORIZONTAL"]

    for i, (ax, files) in enumerate(zip(axes, files_list)):
        for filepath, label, color, linestyle in zip(files, labels, colors, linestyles):
            df = pd.read_csv(filepath)
            budgets = df["Budget"][:-1]
            means = df[f"{metric} mean"][:-1]
            ses = df[f"{metric} SE"][:-1]
            ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels(budgets, fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.axhline(0, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(config_names[i], fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if i == 0:
            if metric == "Collisions":
                ax.set_ylabel("Fraction of Seeds with Collision", fontsize=14)
            else:
                ax.set_ylabel(metric, fontsize=14)

    fig.supxlabel("Planning Budget (log scale)", fontsize=14)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(labels_), fontsize=12)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.13, wspace=0.12)
    plt.savefig("comparison_subplots_1x2.png")
    plt.show()

if __name__ == "__main__":

    TRAIN_CONFIG = "MAZE_RL"
    map_size = 8
    #CONFIGS = ["DEFAULT", "NARROW", "SLALOM"]
    #CONFIGS = ["MAZE_LL", "MAZE_RR", "MAZE_LR"]
    selpols = ("PUCT", "UCT")

    # labels = [
    #     f"c=0.0, β=1.0",
    #     f"c=0.0, β=10.0",
    #     f"c=0.0, β=100.0",
    #     f"c=0.1, β=1.0",
    #     f"c=0.1, β=10.0",
    #     f"c=0.1, β=100.0",
    #     # "PDDP",
    #     # "EDP"
    # ]
    labels = [
        f"c=0.0",
        f"c=0.1",
        f"c=0.5",
        f"c=1.0",
        f"c=2.0",
        f"c=100.0",
        # "PDDP",
        # "EDP"
        ]

    # filepaths_dict = {
    #     selpol: {
    #         cfg: [
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.1)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.5)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(2.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         ] for cfg in CONFIGS
    #     } for selpol in selpols
    # }

    # filepaths_dict = {
    #     selpol: {
    #         cfg: [
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(10.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #             f"thesis_exp/{map_size}x{map_size}/Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_{map_size}x{map_size}_{TRAIN_CONFIG}_{cfg}.csv",
    #         ] for cfg in CONFIGS
    #     } for selpol in selpols
    # }

    # thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(2.0)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv

    CONFIGS = ["RANDOM_HORIZONTAL", "OBS_HORIZONTAL"]

    filepaths_dict = {
        selpol: {
            cfg: [
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.1)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.5)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(2.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(100.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
                # f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.025)_subthresh_(0.05)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_0.5_treuse_True_{cfg}_bump_True_randstart_False.csv",
                # f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_treuse_True_bloops_True_loopstr_0.03_ttemp_(None)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv"

            ] for cfg in CONFIGS
        } for selpol in selpols
    }

    # filepaths_dict = {
    #     selpol: {
    #         cfg: [
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(1.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(10.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.0)_Beta_(100.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(1.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_({"PolicyUCT" if selpol=="UCT" else "PolicyPUCT"})_c_(0.1)_Beta_(100.0)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv",
    #             f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.025)_subthresh_(0.05)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_0.2_treuse_True_{cfg}_bump_True_randstart_False.csv",
    #             #f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.025)_subthresh_(0.05)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_0.1_treuse_True_{cfg}_bump_True_randstart_False.csv",
    #             #f"thesis_exp/parking_csvs/PARKING-ACC_Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_treuse_True_bloops_True_loopstr_0.03_ttemp_(None)_ValueEst_(nn)_{cfg}_bump_True_randstart_False.csv"
    #         ] for cfg in CONFIGS
    #     } for selpol in selpols
    # }

    


    # Format labels for each SELPOL
    labels_puct = [lbl.format(SELPOL="PUCT") for lbl in labels]
    labels_uct = [lbl.format(SELPOL="UCT") for lbl in labels]

    # Call the new plotting function
    #plot_comparison_subplots_2x2_parking(filepaths_dict, labels_puct, selpols=selpols, train_config=None)
    labels = [
        "AZ+PUCT",
        "AZ+UCT",
        "MVC+PUCT",
        "MVC+UCT",
        "PDDP",
        "EDP"
    ]

    files_sparse = [
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_RANDOM_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_RANDOM_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_RANDOM_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_RANDOM_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.025)_subthresh_(0.05)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_0.2_treuse_True_RANDOM_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_treuse_True_bloops_True_loopstr_0.03_ttemp_(None)_ValueEst_(nn)_RANDOM_HORIZONTAL_bump_True_randstart_False.csv"
    ]

    files_horizontal = [
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts)_EvalPol_(mvc)_SelPol_(PolicyUCT)_c_(0.1)_Beta_(10.0)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(pddp)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(10.0)_Predictor_(current_value)_eps_(0.025)_subthresh_(0.05)_ValueEst_(nn)_ttemp_(None)_Value_Penalty_0.2_treuse_True_OBS_HORIZONTAL_bump_True_randstart_False.csv",
        f"thesis_exp/parking_csvs/COLLISIONS_PARKING-ACC_Algorithm_(azmcts_no_loops)_EvalPol_(mvc)_SelPol_(PolicyPUCT)_c_(0.1)_Beta_(1.0)_treuse_True_bloops_True_loopstr_0.03_ttemp_(None)_ValueEst_(nn)_OBS_HORIZONTAL_bump_True_randstart_False.csv"
    ]

    colors = [
        "blue",  # PUCT
        "blue",  # UCT
        "green",  # MVC PUCT
        "green",  # MVC UCT
        "purple",  # PDDP
        "black"  # EDP
    ]

    linestyles = [
        "-",  # Solid
        "--",  # Dashed
        "-",  # Solid
        "--",  # Dotted
        "-",  # Solid
        "-"   # Solid
    ]

    # plot_comparison_subplots_1x2([files_sparse, files_horizontal], labels, colors, linestyles, metric="Return")
    # plot_comparison_subplots_2x3_parking(filepaths_dict, labels, max_episode_length=200, selpols=selpols, train_config=None)
    #plot_comparison_subplots_2x3(filepaths_dict, labels, map_size=8, max_episode_length=100, selpols=selpols, train_config=TRAIN_CONFIG)

    plot_comparison_subplots_1x2([files_sparse, files_horizontal], labels, colors, linestyles, metric="Collisions")

