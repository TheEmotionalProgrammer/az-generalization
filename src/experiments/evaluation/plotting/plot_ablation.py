import pandas as pd
import matplotlib.pyplot as plt

metric_dict = {
    "Discounted Return": "Mean Discounted Return",
}

def plot_ablation_subplots(filepaths_lr_rl, filepaths_rl_lr, labels_styles_colors, output_filename):
    """
    Plots ablation studies for 8x8 and 16x16 narrow configurations as subplots.

    Args:
        filepaths_lr_rl (list): List of file paths for the LR-RL ablation study.
        filepaths_rl_lr (list): List of file paths for the RL-LR ablation study.
        labels_styles_colors (list of tuples): List of tuples (label, color, linestyle, marker) for the plot.
        output_filename (str): Name of the output file to save the plot.
    """
    metric = "Discounted Return"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot for LR-RL
    optimal_value_lr_rl = 0.95 ** 20
    for filepath, (label, color, linestyle, marker) in zip(filepaths_lr_rl, labels_styles_colors):
        df = pd.read_csv(filepath)
        axes[0].plot(df["Budget"], df[f"{metric} mean"], marker=marker, linestyle=linestyle, color=color, label=label)
        axes[0].fill_between(df["Budget"],
                             df[f"{metric} mean"] - df[f"{metric} SE"],
                             df[f"{metric} mean"] + df[f"{metric} SE"],
                             alpha=0.1, color=color)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(df["Budget"])
    axes[0].set_xticklabels(df["Budget"], fontsize=16)
    axes[0].tick_params(axis='y', labelsize=16)
    axes[0].axhline(optimal_value_lr_rl, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
    axes[0].set_title("MAZE_LR → MAZE_RL", fontsize=18)
    axes[0].set_xlabel("Planning Budget (log scale)", fontsize=18)
    axes[0].set_ylabel(metric_dict[metric], fontsize=18)
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    axes[0].set_ylim(0, optimal_value_lr_rl * 1.1)  # Set y-axis limit for 8x8

    # Plot for 16x16
    optimal_value_rl_lr =  0.95 ** ((8 * 2 - 2))
    for filepath, (label, color, linestyle, marker) in zip(filepaths_rl_lr, labels_styles_colors):
        df = pd.read_csv(filepath)
        axes[1].plot(df["Budget"], df[f"{metric} mean"], marker=marker, linestyle=linestyle, color=color, label=label)
        axes[1].fill_between(df["Budget"],
                             df[f"{metric} mean"] - df[f"{metric} SE"],
                             df[f"{metric} mean"] + df[f"{metric} SE"],
                             alpha=0.1, color=color)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(df["Budget"])
    axes[1].set_xticklabels(df["Budget"], fontsize=16)
    axes[1].tick_params(axis='y', labelsize=16)
    axes[1].axhline(optimal_value_rl_lr, color='red', linestyle='dotted', linewidth=1.5, label="Optimal")
    axes[1].set_title("MAZE_RL → MAZE_LR", fontsize=18)
    axes[1].set_xlabel("Planning Budget (log scale)", fontsize=18)
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    axes[1].set_ylim(0, optimal_value_rl_lr * 1.1)  # Set y-axis limit for 16x16

    # Collect unique legend entries from both axes
    handles_0, labels_0 = axes[0].get_legend_handles_labels()
    handles_1, labels_1 = axes[1].get_legend_handles_labels()

    # Combine and remove duplicates while preserving order
    from collections import OrderedDict
    combined = list(zip(handles_0 + handles_1, labels_0 + labels_1))
    unique = list(OrderedDict((label, handle) for handle, label in combined).items())

    # Unzip
    labels_unique, handles_unique = zip(*unique)

    fig.legend(handles_unique, labels_unique, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels_unique), fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_filename)
    plt.show()


if __name__ == "__main__":

    labels_styles_colors = [
        ("EDP (standard)", "red", "-", "o"),
        ("C=1", "green", "-", "s"),
        ("NO TREE REUSE", "#56B4E9", "-", "^"),
        ("NO BLOCK LOOPS", "#E69F00", "-", "D"),
    ]

    selpol = "UCT"  # or "UCT"

    filepaths_lr_rl = [
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_False_bloops_True_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_True_bloops_False_ttemp_(None)_8x8_MAZE_LR_MAZE_RL.csv",
    ]

    filepaths_rl_lr = [
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(1.0)_treuse_True_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_False_bloops_True_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
        f"hpc/Algorithm_(edp)_EvalPol_(visit)_SelPol_({selpol})_c_(0.0)_treuse_True_bloops_False_ttemp_(None)_8x8_MAZE_RL_MAZE_LR.csv",
    ]

    # Generate the subplot
    plot_ablation_subplots(filepaths_lr_rl, filepaths_rl_lr, labels_styles_colors, "ablation_subplots.png")