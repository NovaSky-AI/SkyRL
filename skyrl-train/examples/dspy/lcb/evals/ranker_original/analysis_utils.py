import json
import os
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

def load_result_files(result_file_enum, base_path):
    """
    Load JSON result files from a directory based on an Enum class.
    
    Args:
        result_file_enum: An Enum class where each value is a file name
        base_path: Base directory path where the files are located
    
    Returns:
        dict: Dictionary mapping file names to their loaded JSON data (or None if missing/invalid)
    """
    results = {}
    
    for result_file in result_file_enum:
        file_name = result_file.value
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    results[file_name] = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON file: {file_name}")
                    results[file_name] = None
        else:
            results[file_name] = None
            print(f"File {file_name} does not exist in {base_path}")
    
    return results

def extract_scores_from_results(results_dict, result_file_enum, validate=True):
    """
    Extract scores from results dictionary organized by runs.
    
    Args:
        results_dict: Dictionary mapping file names to their loaded JSON data
        result_file_enum: An Enum class where each value is a file name
        validate: If True, perform assertions to validate the extracted scores
    
    Returns:
        dict: Dictionary mapping enum names to lists of scores per run
              Format: {enum_name: [scores_run_0, scores_run_1, ...]}
    """
    scores = {}
    
    for file_name in result_file_enum:
        results = results_dict[file_name.value]['results']
        scores_by_model = []
        for run_idx in range(len(results)):
            scores_by_run = [result['score'] for result in results[str(run_idx)]]
            scores_by_model.append(scores_by_run)
        
        scores[file_name.name] = scores_by_model
    
    if validate:
        assert len(scores) == len(result_file_enum)
        # Use the first file's results for length validation
        first_file = list(result_file_enum)[0]
        first_results = results_dict[first_file.value]['results']
        assert len(scores[first_file.name]) == len(first_results)
    
    return scores

def compute_mean_scores_per_run(scores_dict, result_file_enum, save_path=None):
    """
    Compute mean scores per run for each file in the scores dictionary.
    
    Args:
        scores_dict: Dictionary mapping enum names to lists of scores per run
                     Format: {enum_name: [scores_run_0, scores_run_1, ...]}
        result_file_enum: An Enum class where each value is a file name
        save_path: Optional path to save the results as a pickle file
    
    Returns:
        dict: Dictionary mapping enum names to lists of mean scores per run
              Format: {enum_name: [mean_run_0, mean_run_1, ...]}
    """
    mean_scores_per_run = {}
    
    for file_name in result_file_enum:
        mean_scores_per_run[file_name.name] = [
            np.mean(scores_dict[file_name.name][i]) 
            for i in range(len(scores_dict[file_name.name]))
        ]
    
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(mean_scores_per_run, f)
    
    return mean_scores_per_run

def compute_mean_scores_across_runs(mean_scores_per_run_dict, result_file_enum, save_path=None):
    """
    Compute mean scores across all runs for each file.
    
    Args:
        mean_scores_per_run_dict: Dictionary mapping enum names to lists of mean scores per run
                                  Format: {enum_name: [mean_run_0, mean_run_1, ...]}
        result_file_enum: An Enum class where each value is a file name
        save_path: Optional path to save the results as a pickle file
    
    Returns:
        dict: Dictionary mapping enum names to single mean score across all runs
              Format: {enum_name: mean_score}
    """
    mean_scores_across_runs = {}
    
    for file_name in result_file_enum:
        mean_scores_across_runs[file_name.name] = np.mean(mean_scores_per_run_dict[file_name.name])
    
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(mean_scores_across_runs, f)
    
    return mean_scores_across_runs

def parse_model_and_split(key, comparison = False):
    """
    Parse model name and dataset split from a key string.
    
    Expected key formats:
    - PROG_<MODEL>_<SPLIT> (e.g., "PROG_7B_TEST")
    - PROG_<MODEL>_<STEPS>_<SPLIT> (e.g., "PROG_1POINT5B_100STEPS_TEST")
    
    Args:
        key: String key to parse (e.g., "PROG_1POINT5B_100STEPS_TEST")
    
    Returns:
        tuple: (model, split) where model is the normalized model name and split is "Train" or "Test"
               Returns (None, None) if key cannot be parsed
    """
    parts = key.split("_")
    if len(parts) < 3:
        return None, None
    
    # Determine split first (it's usually the last part or contains TRAIN/TEST)
    split = None
    if "TRAIN" in key.upper():
        split = "Train"
    elif "TEST" in key.upper():
        split = "Test"
    else:
        # Fallback: use the last part
        split = parts[-1]
    
    # Determine model - handle special multi-part model names first
    if "1POINT5B_100STEPS" in key:
        model = "1.5B 100 Steps"
        if comparison:
            model = "1.5B"
    elif "3B_100STEPS" in key:
        model = "3B 100 Steps"
        if comparison:
            model = "3B"
    elif "3B_140STEPS" in key:  
        model = "3B 140 Steps"
        if comparison:
            model = "3B"
    else:
        # Standard case: model is parts[1]
        model = parts[1]
        if model == "15B" or model == "1POINT5B":
            model = "1.5B"
    print(model, split, key, comparison, split)
    return model, split

def plot_mean_scores_barplot(
    mean_scores_dict,
    title="Model Performance",
    xlabel="Model Size",
    ylabel="Mean Accuracy",
    model_order=None,
    split_order=None,
    figsize=(8, 5),
    palette_name="Blues",
    font_scale=1.4,
    y_padding_factor=1.25,
    show_annotations=True,
    annotation_format="{:.3f}"
):
    """
    Create a barplot of mean scores across runs, grouped by model and split.
    
    Args:
        mean_scores_dict: Dictionary mapping enum names to mean scores
                          Format: {enum_name: mean_score}
                          Expected enum_name format: "PROG_<MODEL>_<SPLIT>"
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        model_order: List of model names in desired order (default: ["1.5B", "3B", "7B"])
        split_order: List of split names in desired order (default: ["TRAIN", "TEST"])
        figsize: Figure size tuple
        palette_name: Seaborn palette name
        font_scale: Font scale for seaborn theme
        y_padding_factor: Factor to multiply max score for y-axis limit
        show_annotations: Whether to annotate bars with values
        annotation_format: Format string for annotations
    
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Parse scores into DataFrame
    rows = []
    for key, val in mean_scores_dict.items():
        model, split = parse_model_and_split(key)
        if model is not None and split is not None:
            rows.append({"Model": model, "Split": split, "Score": float(val)})
    
    df = pd.DataFrame(rows)
    
    # Set default model order if not provided
    if model_order is None:
        model_order = ["1.5B", "3B", "7B"]
    if split_order is None:
        split_order = ["TRAIN", "TEST"]
    
    # Order models numerically
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df["Split"] = pd.Categorical(df["Split"], categories=split_order, ordered=True)
    
    # Journal-ready plot configuration
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )
    
    plt.figure(figsize=figsize)
    
    # Determine number of colors needed (based on unique splits)
    n_colors = len(df["Split"].unique())
    
    ax = sns.barplot(
        data=df,
        x="Model",
        y="Score",
        hue="Split",
        palette=sns.color_palette(palette_name, n_colors=n_colors),
        edgecolor="black",
        linewidth=1.2
    )
    
    # Annotate bars with values
    if show_annotations:
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    annotation_format.format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom",
                    fontsize=10,
                    xytext=(0, 3),
                    textcoords="offset points"
                )
    
    # Titles + labels
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Set y-limit with padding
    ax.set_ylim(0, df["Score"].max() * y_padding_factor)
    
    # Cleaner legend placed outside
    ax.legend(
        title="Dataset Split",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()
    
    return ax

def plot_mean_scores_barplot_with_ci(
    mean_scores_per_run_dict,
    mean_scores_across_runs_dict,
    title="Model Performance",
    xlabel="Model Size",
    ylabel="Mean Accuracy",
    model_order=None,
    split_order=None,
    figsize=(8, 5),
    palette_name="Blues",
    font_scale=1.2,
    y_padding_factor=1.3,
    ci_cap_size=0.06,
    confidence_level=0.95
):
    """
    Create a barplot of mean scores with 95% confidence intervals, grouped by model and split.
    
    Args:
        mean_scores_per_run_dict: Dictionary mapping enum names to lists of mean scores per run
                                  Format: {enum_name: [mean_run_0, mean_run_1, ...]}
        mean_scores_across_runs_dict: Dictionary mapping enum names to single mean score across all runs
                                      Format: {enum_name: mean_score}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        model_order: List of model names in desired order (default: ["1.5B", "3B", "7B"])
        split_order: List of split names in desired order (default: ["TRAIN", "TEST"])
        figsize: Figure size tuple
        palette_name: Seaborn palette name
        font_scale: Font scale for seaborn theme
        y_padding_factor: Factor to multiply max score for y-axis limit
        ci_cap_size: Size of the confidence interval cap (horizontal line width)
        confidence_level: Confidence level for intervals (default: 0.95 for 95% CI)
    
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Set default orders if not provided
    if model_order is None:
        model_order = ["1.5B", "3B", "7B"]
    if split_order is None:
        split_order = ["TRAIN", "TEST"]
    
    # ------------------------------------------------------------
    # Construct per-run dataframe
    # ------------------------------------------------------------
    rows_run = []
    for key, vals in mean_scores_per_run_dict.items():
        model, split = parse_model_and_split(key)
        if model is not None and split is not None:
            for v in vals:
                rows_run.append({"Model": model, "Split": split, "Score": float(v)})
    df_runs = pd.DataFrame(rows_run)
    
    # ------------------------------------------------------------
    # Construct mean dataframe
    # ------------------------------------------------------------
    rows_mean = []
    for key, val in mean_scores_across_runs_dict.items():
        model, split = parse_model_and_split(key)
        if model is not None and split is not None:
            rows_mean.append({"Model": model, "Split": split, "Score": float(val)})
    df_mean = pd.DataFrame(rows_mean)
    
    # Consistent ordering
    df_mean["Model"] = pd.Categorical(df_mean["Model"], categories=model_order, ordered=True)
    df_runs["Model"] = pd.Categorical(df_runs["Model"], categories=model_order, ordered=True)
    df_mean["Split"] = pd.Categorical(df_mean["Split"], categories=split_order, ordered=True)
    df_runs["Split"] = pd.Categorical(df_runs["Split"], categories=split_order, ordered=True)
    
    # ------------------------------------------------------------
    # Compute confidence intervals per (Model, Split)
    # ------------------------------------------------------------
    ci_rows = []
    for (model, split), group in df_runs.groupby(["Model", "Split"]):
        scores = group["Score"].to_numpy()
        n = len(scores)
        mean = scores.mean()
        std = scores.std(ddof=1)
        
        # CI using Student-t
        alpha = 1 - confidence_level
        ci = t.ppf(1 - alpha/2, df=n - 1) * std / np.sqrt(n)
        
        ci_rows.append({
            "Model": model,
            "Split": split,
            "mean": mean,
            "ci_low": mean - ci,
            "ci_high": mean + ci,
        })
    
    df_ci = pd.DataFrame(ci_rows)
    # Make sure df_ci is sorted in the SAME order as the bars:
    df_ci["Model"] = pd.Categorical(df_ci["Model"], categories=model_order, ordered=True)
    df_ci["Split"] = pd.Categorical(df_ci["Split"], categories=split_order, ordered=True)
    df_ci = df_ci.sort_values(["Model", "Split"]).reset_index(drop=True)
    
    # ------------------------------------------------------------
    # Journal-style plot
    # ------------------------------------------------------------
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )
    
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(
        data=df_mean,
        x="Model",
        y="Score",
        hue="Split",
        hue_order=split_order,
        palette=sns.color_palette(palette_name, n_colors=len(split_order)),
        edgecolor="black",
        linewidth=1.2
    )
    
    # ------------------------------------------------------------
    # Compute actual x-positions for each (Model, Split)
    # ------------------------------------------------------------
    x_ticks = ax.get_xticks()
    group_width = 0.8
    n_hue = len(split_order)
    bar_width = group_width / n_hue
    
    # Map Split → hue index
    hue_index_map = {split: i for i, split in enumerate(split_order)}
    
    # Bar center positions
    bar_positions = {}
    for model, x_base in zip(model_order, x_ticks):
        for split in split_order:
            hue_i = hue_index_map[split]
            x_center = (
                x_base - group_width/2
                + (hue_i + 0.5) * bar_width
            )
            bar_positions[(model, split)] = x_center
    
    # ------------------------------------------------------------
    # Draw confidence interval whiskers at correct bar x positions
    # ------------------------------------------------------------
    for i, row in df_ci.iterrows():
        model = row["Model"]
        split = row["Split"]
        
        x = bar_positions[(model, split)]
        y_low = row["ci_low"]
        y_high = row["ci_high"]
        
        # vertical line
        ax.plot([x, x], [y_low, y_high], color="black", linewidth=1.2)
        
        # caps
        ax.plot([x - ci_cap_size, x + ci_cap_size], [y_low, y_low], color="black", linewidth=1.2)
        ax.plot([x - ci_cap_size, x + ci_cap_size], [y_high, y_high], color="black", linewidth=1.2)
    
    # Titles & labels
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.set_ylim(0, df_mean["Score"].max() * y_padding_factor)
    
    # Legend
    ax.legend(
        title="Dataset Split",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()
    
    return ax

def plot_comparison_barplot(
    mean_scores_across_runs_dict_1,
    mean_scores_across_runs_dict_2,
    dataset_1_name="Dataset 1",
    dataset_2_name="Dataset 2",
    title="Model Performance Comparison",
    xlabel="Dataset Split",
    ylabel="Mean Accuracy",
    model_order=None,
    split_order=None,
    figsize=(12, 6),
    dataset_1_color="#B3CEDE",
    dataset_2_color="#4884AF",
    font_scale=1.2,
    y_padding_factor=1.25,
    show_annotations=True,
    annotation_format="{:.3f}"
):
    """
    Create a comparison barplot of two datasets, grouped by split (TRAIN/TEST) with model sizes within each group.
    Bars are defined by x-axis labels (Split-Model), with two colors for the two datasets.
    
    Args:
        mean_scores_across_runs_dict_1: First dataset - Dictionary mapping enum names to mean scores
                                       Format: {enum_name: mean_score}
        mean_scores_across_runs_dict_2: Second dataset - Dictionary mapping enum names to mean scores
                                       Format: {enum_name: mean_score}
        dataset_1_name: Name for the first dataset (e.g., "Baseline")
        dataset_2_name: Name for the second dataset (e.g., "Trained")
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        model_order: List of model names in desired order for both datasets (default: ["1.5B", "3B", "7B"])
        split_order: List of split names in desired order (default: ["TRAIN", "TEST"])
        figsize: Figure size tuple
        dataset_1_color: Color for dataset 1 bars
        dataset_2_color: Color for dataset 2 bars
        font_scale: Font scale for seaborn theme
        y_padding_factor: Factor to multiply max score for y-axis limit
        show_annotations: Whether to annotate bars with values
        annotation_format: Format string for annotations
    
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Set default orders if not provided
    if model_order is None:
        model_order = ["1.5B", "3B", "7B"]
    if split_order is None:
        split_order = ["TRAIN", "TEST"]
    
    # Helper function to parse scores
    def parse_scores(mean_scores_dict, dataset_name):
        rows = []
        for key, val in mean_scores_dict.items():
            model, split = parse_model_and_split(key, comparison = True)
            if model is None:
                raise ValueError(f"Failed to parse model and split from key: {key}")
            
            rows.append({
                "Dataset": dataset_name,
                "Model": model,
                "Split": split,
                "Score": float(val)
            })
        return rows
    
    # Parse both datasets
    rows_1 = parse_scores(mean_scores_across_runs_dict_1, dataset_1_name)
    rows_2 = parse_scores(mean_scores_across_runs_dict_2, dataset_2_name)
    
    # Combine into single DataFrame
    df = pd.DataFrame(rows_1 + rows_2)
    
    # Set categorical ordering
    # df["Split"] = pd.Categorical(df["Split"], categories=split_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    
    # Sort by Split, then Model, then Dataset
    df = df.sort_values(["Split", "Model", "Dataset"]).reset_index(drop=True)
    
    # Journal-style plot configuration
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis labels: Split\nModel for each combination
    x_labels = []
    x_positions = []
    bar_positions = {}  # (split, model, dataset) -> x position
    bar_width = 0.35  # Width of each bar
    gap_between_models = 0.2  # Gap between model groups
    gap_between_splits = 0.5  # Gap between split groups
    
    x = 0
    for split in split_order:
        for model in model_order:
            # Create label: Split\nModel
            label = f"{split}\n{model}"
            x_labels.append(label)
            
            # Position for dataset 1 (left bar)
            x1 = x - bar_width / 2
            bar_positions[(split, model, dataset_1_name)] = x1
            
            # Position for dataset 2 (right bar)
            x2 = x + bar_width / 2
            bar_positions[(split, model, dataset_2_name)] = x2
            
            x_positions.append(x)
            
            # Move to next model group (with gap)
            x += bar_width * 2 + gap_between_models
        
        # Add gap between splits
        x += gap_between_splits
    
    # Draw bars manually
    for (split, model, dataset), x_pos in bar_positions.items():
        score = df[(df["Split"] == split) & (df["Model"] == model) & (df["Dataset"] == dataset)]["Score"].values
        if len(score) > 0:
            color = dataset_1_color if dataset == dataset_1_name else dataset_2_color
            ax.bar(x_pos, score[0], width=bar_width, color=color, edgecolor="black", linewidth=1.2)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(-bar_width - gap_between_models/2, x_positions[-1] + bar_width + gap_between_models/2)
    
    # Annotate bars with values
    if show_annotations:
        for (split, model, dataset), x_pos in bar_positions.items():
            score = df[(df["Split"] == split) & (df["Model"] == model) & (df["Dataset"] == dataset)]["Score"].values
            if len(score) > 0 and score[0] > 0:
                ax.annotate(
                    annotation_format.format(score[0]),
                    (x_pos, score[0]),
                    ha="center", va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points"
                )
    
    # Titles & labels
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.set_ylim(0, df["Score"].max() * y_padding_factor)
    
    # Create custom legend for datasets
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dataset_1_color, edgecolor="black", label=dataset_1_name),
        Patch(facecolor=dataset_2_color, edgecolor="black", label=dataset_2_name)
    ]
    ax.legend(
        handles=legend_elements,
        title="Dataset",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()
    
    return ax

def plot_comparison_barplot_with_ci(
    mean_scores_per_run_dict_1,
    mean_scores_across_runs_dict_1,
    mean_scores_per_run_dict_2,
    mean_scores_across_runs_dict_2,
    dataset_1_name="Dataset 1",
    dataset_2_name="Dataset 2",
    title="Model Performance Comparison",
    xlabel="Dataset Split",
    ylabel="Mean Accuracy",
    model_order=None,
    split_order=None,
    figsize=(12, 6),
    dataset_1_color="#B3CEDE",
    dataset_2_color="#4884AF",
    font_scale=1.2,
    y_padding_factor=1.3,
    ci_cap_size=0.06,
    confidence_level=0.95,
    show_annotations=False
):
    """
    Create a comparison barplot of two datasets with confidence intervals, grouped by split (TRAIN/TEST) 
    with model sizes within each group. Bars are defined by x-axis labels (Split-Model), with two colors for the two datasets.
    
    Args:
        mean_scores_per_run_dict_1: First dataset - Dictionary mapping enum names to lists of mean scores per run
                                    Format: {enum_name: [mean_run_0, mean_run_1, ...]}
        mean_scores_across_runs_dict_1: First dataset - Dictionary mapping enum names to single mean score
                                       Format: {enum_name: mean_score}
        mean_scores_per_run_dict_2: Second dataset - Dictionary mapping enum names to lists of mean scores per run
                                   Format: {enum_name: [mean_run_0, mean_run_1, ...]}
        mean_scores_across_runs_dict_2: Second dataset - Dictionary mapping enum names to single mean score
                                       Format: {enum_name: mean_score}
        dataset_1_name: Name for the first dataset (e.g., "Baseline")
        dataset_2_name: Name for the second dataset (e.g., "Trained")
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        model_order: List of model names in desired order (default: ["1.5B", "3B", "7B"])
        split_order: List of split names in desired order (default: ["TRAIN", "TEST"])
        figsize: Figure size tuple
        dataset_1_color: Color for dataset 1 bars
        dataset_2_color: Color for dataset 2 bars
        font_scale: Font scale for seaborn theme
        y_padding_factor: Factor to multiply max score for y-axis limit
        ci_cap_size: Size of the confidence interval cap (horizontal line width)
        confidence_level: Confidence level for intervals (default: 0.95 for 95% CI)
        show_annotations: Whether to annotate bars with values
    
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Set default orders if not provided
    if model_order is None:
        model_order = ["1.5B", "3B", "7B"]
    if split_order is None:
        split_order = ["TRAIN", "TEST"]
    
    # Helper function to parse per-run scores
    def parse_per_run_scores(mean_scores_per_run_dict, dataset_name):
        rows = []
        for key, vals in mean_scores_per_run_dict.items():
            model, split = parse_model_and_split(key, comparison = True)
            if model is None:
                raise ValueError(f"Failed to parse model and split from key: {key}")
            
            for v in vals:
                rows.append({
                    "Dataset": dataset_name,
                    "Model": model,
                    "Split": split,
                    "Score": float(v)
                })
        return rows
    
    # Helper function to parse mean scores
    def parse_mean_scores(mean_scores_dict, dataset_name):
        rows = []
        for key, val in mean_scores_dict.items():
            model, split = parse_model_and_split(key, comparison = True)
            if model is None:
                raise ValueError(f"Failed to parse model and split from key: {key}")
            
            rows.append({
                "Dataset": dataset_name,
                "Model": model,
                "Split": split,
                "Score": float(val)
            })
        return rows
    
    # Parse both datasets
    rows_run_1 = parse_per_run_scores(mean_scores_per_run_dict_1, dataset_1_name)
    rows_run_2 = parse_per_run_scores(mean_scores_per_run_dict_2, dataset_2_name)
    rows_mean_1 = parse_mean_scores(mean_scores_across_runs_dict_1, dataset_1_name)
    rows_mean_2 = parse_mean_scores(mean_scores_across_runs_dict_2, dataset_2_name)
    
    # Combine into DataFrames
    df_runs = pd.DataFrame(rows_run_1 + rows_run_2)
    df_mean = pd.DataFrame(rows_mean_1 + rows_mean_2)
    
    # Set categorical ordering
    df_mean["Split"] = pd.Categorical(df_mean["Split"], categories=split_order, ordered=True)
    df_runs["Split"] = pd.Categorical(df_runs["Split"], categories=split_order, ordered=True)
    df_mean["Model"] = pd.Categorical(df_mean["Model"], categories=model_order, ordered=True)
    df_runs["Model"] = pd.Categorical(df_runs["Model"], categories=model_order, ordered=True)
    
    # Sort
    df_mean = df_mean.sort_values(["Split", "Model", "Dataset"]).reset_index(drop=True)
    df_runs = df_runs.sort_values(["Split", "Model", "Dataset"]).reset_index(drop=True)
    
    # Compute confidence intervals per (Split, Model, Dataset)
    ci_rows = []
    for (split, model, dataset), group in df_runs.groupby(["Split", "Model", "Dataset"]):
        scores = group["Score"].to_numpy()
        n = len(scores)
        mean = scores.mean()
        std = scores.std(ddof=1)
        
        # CI using Student-t
        alpha = 1 - confidence_level
        ci = t.ppf(1 - alpha/2, df=n - 1) * std / np.sqrt(n)
        
        ci_rows.append({
            "Split": split,
            "Model": model,
            "Dataset": dataset,
            "mean": mean,
            "ci_low": mean - ci,
            "ci_high": mean + ci,
        })
    
    df_ci = pd.DataFrame(ci_rows)
    
    # Set categorical ordering for CI dataframe
    df_ci["Split"] = pd.Categorical(df_ci["Split"], categories=split_order, ordered=True)
    df_ci["Model"] = pd.Categorical(df_ci["Model"], categories=model_order, ordered=True)
    df_ci = df_ci.sort_values(["Split", "Model", "Dataset"]).reset_index(drop=True)
    
    # Journal-style plot configuration
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create x-axis labels: Split\nModel for each combination
    x_labels = []
    x_positions = []
    bar_positions = {}  # (split, model, dataset) -> x position
    bar_width = 0.35  # Width of each bar
    gap_between_models = 0.2  # Gap between model groups
    gap_between_splits = 0.5  # Gap between split groups
    
    x = 0
    for split in split_order:
        for model in model_order:
            # Create label: Split\nModel
            label = f"{split}\n{model}"
            x_labels.append(label)
            
            # Position for dataset 1 (left bar)
            x1 = x - bar_width / 2
            bar_positions[(split, model, dataset_1_name)] = x1
            
            # Position for dataset 2 (right bar)
            x2 = x + bar_width / 2
            bar_positions[(split, model, dataset_2_name)] = x2
            
            x_positions.append(x)
            
            # Move to next model group (with gap)
            x += bar_width * 2 + gap_between_models
        
        # Add gap between splits
        x += gap_between_splits
    
    # Draw bars manually
    for (split, model, dataset), x_pos in bar_positions.items():
        score = df_mean[(df_mean["Split"] == split) & (df_mean["Model"] == model) & (df_mean["Dataset"] == dataset)]["Score"].values
        if len(score) > 0:
            color = dataset_1_color if dataset == dataset_1_name else dataset_2_color
            ax.bar(x_pos, score[0], width=bar_width, color=color, edgecolor="black", linewidth=1.2)
    
    # Draw confidence interval whiskers
    for i, row in df_ci.iterrows():
        split = row["Split"]
        model = row["Model"]
        dataset = row["Dataset"]
        
        x_pos = bar_positions[(split, model, dataset)]
        y_low = row["ci_low"]
        y_high = row["ci_high"]
        
        # vertical line
        ax.plot([x_pos, x_pos], [y_low, y_high], color="black", linewidth=1.2)
        
        # caps
        ax.plot([x_pos - ci_cap_size, x_pos + ci_cap_size], [y_low, y_low], color="black", linewidth=1.2)
        ax.plot([x_pos - ci_cap_size, x_pos + ci_cap_size], [y_high, y_high], color="black", linewidth=1.2)
    
    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlim(-bar_width - gap_between_models/2, x_positions[-1] + bar_width + gap_between_models/2)
    
    # Annotate bars with values if requested
    if show_annotations:
        for (split, model, dataset), x_pos in bar_positions.items():
            score = df_mean[(df_mean["Split"] == split) & (df_mean["Model"] == model) & (df_mean["Dataset"] == dataset)]["Score"].values
            if len(score) > 0 and score[0] > 0:
                ax.annotate(
                    f"{score[0]:.3f}",
                    (x_pos, score[0]),
                    ha="center", va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points"
                )
    
    # Titles & labels
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    ax.set_ylim(0, df_mean["Score"].max() * y_padding_factor)
    
    # Create custom legend for datasets
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dataset_1_color, edgecolor="black", label=dataset_1_name),
        Patch(facecolor=dataset_2_color, edgecolor="black", label=dataset_2_name)
    ]
    ax.legend(
        handles=legend_elements,
        title="Dataset",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()
    
    return ax

def plot_horizontal_stacked_barplot(
    mean_scores_across_runs_dict_baseline,
    mean_scores_across_runs_dict_improved,
    baseline_name="Baseline",
    improved_name="Improvement",
    title="Model Performance Comparison",
    xlabel="Performance",
    ylabel="Model",
    model_order=None,
    split_order=None,
    figsize=(10, 8),
    baseline_color="#1f497d",
    improvement_color="#f79646",
    font_scale=1.2,
    x_padding_factor=1.15,
    show_annotations=True,
    annotation_format="{:.2f}"
):
    """
    Create a horizontal stacked barplot comparing baseline and improved performance.
    Shows baseline as one segment and improvement as another stacked on top.
    
    Args:
        mean_scores_across_runs_dict_baseline: Baseline dataset - Dictionary mapping enum names to mean scores
                                              Format: {enum_name: mean_score}
        mean_scores_across_runs_dict_improved: Improved dataset - Dictionary mapping enum names to mean scores
                                              Format: {enum_name: mean_score}
        baseline_name: Name for the baseline segment
        improved_name: Name for the improvement segment
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        model_order: List of model names in desired order (default: ["1.5B", "3B", "7B"])
        split_order: List of split names in desired order (default: ["TRAIN", "TEST"])
                     If provided, will create separate bars for each split
        figsize: Figure size tuple
        baseline_color: Color for baseline segment
        improvement_color: Color for improvement segment
        font_scale: Font scale for seaborn theme
        x_padding_factor: Factor to multiply max score for x-axis limit
        show_annotations: Whether to annotate segments with values
        annotation_format: Format string for annotations
    
    Returns:
        matplotlib.axes.Axes: The axes object
    """
    # Set default orders if not provided
    if model_order is None:
        model_order = ["1.5B", "3B", "7B"]
    if split_order is None:
        split_order = ["TRAIN", "TEST"]
    
    # Helper function to parse scores
    def parse_scores(mean_scores_dict, dataset_name):
        rows = []
        for key, val in mean_scores_dict.items():
            parts = key.split("_")
            if len(parts) >= 3:
                model = parts[1]
                split = parts[2]
                if model == "15B" or model == "1POINT5B":
                    model = "1.5B"
                # Handle both single values and lists (compute mean if list)
                if isinstance(val, list):
                    score = float(np.mean(val))
                else:
                    score = float(val)
                rows.append({
                    "Dataset": dataset_name,
                    "Model": model,
                    "Split": split,
                    "Score": score
                })
        return rows
    
    # Parse both datasets
    rows_baseline = parse_scores(mean_scores_across_runs_dict_baseline, baseline_name)
    rows_improved = parse_scores(mean_scores_across_runs_dict_improved, improved_name)
    
    # Combine into DataFrames
    df_baseline = pd.DataFrame(rows_baseline)
    df_improved = pd.DataFrame(rows_improved)
    
    # Set categorical ordering
    df_baseline["Split"] = pd.Categorical(df_baseline["Split"], categories=split_order, ordered=True)
    df_improved["Split"] = pd.Categorical(df_improved["Split"], categories=split_order, ordered=True)
    df_baseline["Model"] = pd.Categorical(df_baseline["Model"], categories=model_order, ordered=True)
    df_improved["Model"] = pd.Categorical(df_improved["Model"], categories=model_order, ordered=True)
    
    # Sort
    df_baseline = df_baseline.sort_values(["Split", "Model"]).reset_index(drop=True)
    df_improved = df_improved.sort_values(["Split", "Model"]).reset_index(drop=True)
    
    # Journal-style plot configuration
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "font.family": "serif",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "figure.dpi": 300,
        }
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for stacking
    # Create labels: Split-Model or just Model
    y_labels = []
    baseline_values = []
    improvement_values = []
    y_positions = []
    
    y_pos = 0
    for split in split_order:
        for model in model_order:
            # Get baseline and improved scores
            baseline_score = df_baseline[
                (df_baseline["Split"] == split) & (df_baseline["Model"] == model)
            ]["Score"].values
            
            improved_score = df_improved[
                (df_improved["Split"] == split) & (df_improved["Model"] == model)
            ]["Score"].values
            
            if len(baseline_score) > 0 and len(improved_score) > 0:
                baseline_val = baseline_score[0]
                improved_val = improved_score[0]
                improvement_val = improved_val - baseline_val
                
                # Create label
                if len(split_order) > 1:
                    label = f"{split} {model}"
                else:
                    label = model
                
                y_labels.append(label)
                baseline_values.append(baseline_val)
                improvement_values.append(improvement_val)
                y_positions.append(y_pos)
                y_pos += 1
    
    # Reverse order so top model is at top (like in the image)
    y_labels = y_labels[::-1]
    baseline_values = baseline_values[::-1]
    improvement_values = improvement_values[::-1]
    y_positions = [len(y_positions) - 1 - pos for pos in y_positions]
    
    # Create horizontal stacked bars
    bars_baseline = ax.barh(
        y_positions,
        baseline_values,
        height=0.9,
        color=baseline_color,
        edgecolor="none",
        linewidth=0,
        label=baseline_name,
    )
    
    bars_improvement = ax.barh(
        y_positions,
        improvement_values,
        left=baseline_values,
        height=0.9,
        color=improvement_color,
        edgecolor="none",
        linewidth=0,
        label=improved_name,
    )
    
    # Annotate segments
    if show_annotations:
        for i, (y_pos, baseline_val, improvement_val) in enumerate(zip(y_positions, baseline_values, improvement_values)):
            # Annotate baseline segment
            if baseline_val > 0:
                ax.text(baseline_val / 2, y_pos, 
                       annotation_format.format(baseline_val),
                       ha="center", va="center",
                       fontsize=9, fontweight="bold", color="white")
            
            # Annotate improvement segment
            if improvement_val > 0:
                improvement_x = baseline_val + improvement_val / 2
                ax.text(improvement_x, y_pos,
                       f"+{annotation_format.format(improvement_val)}",
                       ha="center", va="center",
                       fontsize=9, fontweight="bold", color="white")
    
    # Set y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()  # Top model at top
    # Remove extra padding between bars
    ax.set_ylim(-0.5, len(y_positions) - 0.5)
    
    # Set x-axis
    max_total = max([b + i for b, i in zip(baseline_values, improvement_values)])
    ax.set_xlim(0, max_total * x_padding_factor)
    ax.set_xlabel(xlabel, fontsize=14)
    
    # Titles & labels
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Legend
    ax.legend(
        title="",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    
    plt.tight_layout()
    plt.show()
    
    return ax