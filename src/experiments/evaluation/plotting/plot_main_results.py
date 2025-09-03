import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_mazes(filepaths_dict_maze_lr, filepaths_dict_maze_rl, labels_styles_colors, map_size=8, output_filename="combined_comparison.png"):

    metric = "Discounted Return"
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)

    # Plot MAZE_LR configurations (top row)
    for ax, (config_label, filepaths) in zip(axes[0], filepaths_dict_maze_lr.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "MAZE_RL" else 0.95 ** 20
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"MAZE_LR → {config_label}", fontsize=18)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[0, 0]:
            ax.set_ylabel("Mean Discounted Return", fontsize=16)

    # Plot MAZE_RL configurations (bottom row)
    for ax, (config_label, filepaths) in zip(axes[1], filepaths_dict_maze_rl.items()):
        optimal_value = 0.95 ** ((map_size * 2 - 2)) if config_label != "MAZE_RL" else 0.95 ** 20
        for filepath, (label, color, linestyle) in zip(filepaths, labels_styles_colors):
            df = pd.read_csv(filepath)
            ax.plot(df["Budget"], df[f"{metric} mean"], marker="o", linestyle=linestyle, color=color, label=label)
            ax.fill_between(df["Budget"],
                            df[f"{metric} mean"] - df[f"{metric} SE"],
                            df[f"{metric} mean"] + df[f"{metric} SE"],
                            alpha=0.1, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["Budget"])
        ax.set_xticklabels(df["Budget"], fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.axhline(optimal_value, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
        ax.set_title(f"MAZE_RL → {config_label}", fontsize=18)
        ax.grid(True, linestyle="--", linewidth=0.5)
        if ax == axes[1, 0]:
            ax.set_ylabel("Mean Discounted Return", fontsize=16)

    # Add shared x-axis label
    fig.supxlabel("Planning Budget (log scale)", fontsize=18)

    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=16)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1, wspace=0.1, hspace=0.3)
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":

    # Define labels, colors, and line styles (shared across all subplots)
    labels_styles_colors = [
        ("EDP (ours)", "red", "-"),
        ("AZ+PUCT", "blue", "-"),
        ("AZ+UCT", "blue", "--"),
        # ("EDP+PUCT", "red", "-"),
        # ("EDP+UCT", "red", "--"),
    ]

    filepaths_maze_lr = {
        "MAZE_LL": [
             "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_LL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_LL.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_LL.csv",
           
        ],
        "MAZE_RR": [
             "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RR.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RR.csv",
           
        ],
        "MAZE_RL": [
            "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_LR_MAZE_RL.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
        ],
    }

    filepaths_maze_rl = {
        "MAZE_LL": [
            "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LL.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LL.csv",
        ],
        "MAZE_RR": [
            "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_RR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_RR.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_RR.csv",
        ],
        "MAZE_LR": [
            "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(UCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
            "hpc/Algorithm_(azmcts)_EvalPol_(visit)_SelPol_(UCT)_c_(0.1)_ValueEst_(nn)_8x8_MAZE_RL_MAZE_LR.csv",
            # "hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_(PUCT)_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
            
        ],
    }

    plot_combined_mazes(filepaths_maze_lr, filepaths_maze_rl, labels_styles_colors, map_size=8)
