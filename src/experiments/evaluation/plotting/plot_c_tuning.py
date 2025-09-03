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
    }

def plot_c_tuning(filepaths_dict, labels=None, map_size=16, max_episode_length=100, selpols=("PUCT", "UCT"), train_config=None):

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

                # Exclude the last budget value and corresponding statistics
                budgets = df["Budget"][:-1]
                means = df[f"{metric} mean"][:-1]
                ses = df[f"{metric} SE"][:-1]

                ax.plot(budgets, means, marker="o", linestyle=linestyle, color=color, label=label)
                ax.fill_between(budgets, means - ses, means + ses, alpha=0.1, color=color)

            ax.set_xscale("log", base=2)
            ax.set_xticks(budgets)
            ax.set_xticklabels(budgets, fontsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
            if train_config is None:
                ax.set_title(f"{selpol} - {dict_configs[config_label]}", fontsize=18)
            else:
                ax.set_title(f"{selpol} - {train_config} → {dict_configs[config_label]}", fontsize=18)
                
            ax.grid(True, linestyle="--", linewidth=0.5)
            if col == 0:
                ax.set_ylabel("Mean Discounted Return", fontsize=16)

    fig.supxlabel("Planning Budget (log scale)", fontsize=18)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_), fontsize=16)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig("comparison_subplots_2x3.png")
    plt.show()

if __name__ == "__main__":

    TRAIN_CONFIG = "MAZE_RL"
    CONFIGS = ["MAZE_LL", "MAZE_RR", "MAZE_LR"]
    selpols = ("PUCT", "UCT")

    labels = [
        f"c=0.0",
        f"c=0.1",
        f"c=0.5",
        f"c=1.0",
        f"c=2.0",
        f"c=100.0",
        ]

    filepaths_dict = {
        selpol: {
            cfg: [
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.1)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(0.5)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(2.0)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
                f"hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_({selpol})_c_(100.0)_ValueEst_(nn)_8x8_{TRAIN_CONFIG}_{cfg}.csv",
            ] for cfg in CONFIGS
        } for selpol in selpols
    }

    plot_c_tuning(
        filepaths_dict=filepaths_dict,
        labels=labels,
        map_size=map_size,
        selpols=selpols,
        train_config=TRAIN_CONFIG
    )



