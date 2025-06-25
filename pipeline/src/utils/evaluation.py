"""
Evaluation module for contact prediction model.
Contains functions for computing metrics, visualizations, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from utils.metrics import ContactPredictionMetrics
from datamodules.dataset import ProteinSequenceDataset


class PredictionDataset:
    def __init__(self, data: dict[dict]):
        self.data = data

    @classmethod
    def from_batches(cls, batches: list[dict]):
        data = {}
        for batch in batches:
            for contact_logits, mask, metadata in zip(
                batch["contact_logits"], batch["mask_2d"], batch["metadata"]
            ):
                contact_logits = contact_logits[
                    : metadata["length"], : metadata["length"]
                ]
                mask = mask[: metadata["length"], : metadata["length"]]
                data[metadata["id"]] = {
                    "contact_logits": contact_logits,
                    "mask_2d": mask,
                    "metadata": metadata,
                }
        return PredictionDataset(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(
    dataset_path: str, predictions_path: str, clusters_path: str
) -> Tuple[ProteinSequenceDataset, PredictionDataset, pd.DataFrame]:
    """
    Load all necessary data for evaluation.

    Args:
        dataset_path: Path to validation dataset
        predictions_path: Path to predictions pickle file
        clusters_path: Path to clusters TSV file

    Returns:
        Tuple of (dataset, predictions, clusters)
    """
    # Load validation dataset
    dataset = ProteinSequenceDataset.load_from_disk(dataset_path)

    # Load predictions
    with open(predictions_path, "rb") as f:
        predictions = pickle.load(f)
    prediction_dataset = PredictionDataset.from_batches(predictions)

    # Load clusters information
    clusters = pd.read_csv(clusters_path, sep="\t", header=None)
    clusters.columns = ["cluster", "id"]

    return dataset, prediction_dataset, clusters


def compute_metrics_dataframe(
    dataset: ProteinSequenceDataset, prediction_dataset: PredictionDataset
) -> pd.DataFrame:
    """
    Compute all metrics for each sequence and return as DataFrame.

    Args:
        dataset: Validation dataset with ground truth
        prediction_dataset: Dataset with predictions

    Returns:
        DataFrame with computed metrics for each sequence
    """
    scorer = ContactPredictionMetrics()
    metrics = []

    for item in tqdm(dataset, desc="Computing metrics"):
        id_ = item["metadata"]["id"]

        # Create contact map from distance map (< 8 Angstroms)
        contact_map = torch.tensor(
            item["distance_map"] < 8, dtype=torch.long
        ).unsqueeze(0)

        # Get prediction
        pred_item = prediction_dataset[id_]
        contact_logits = pred_item["contact_logits"]

        # Compute metrics
        metric = scorer.compute_all_metrics(
            contact_logits=contact_logits.unsqueeze(0),
            contact_maps=contact_map,
            seq_lengths=torch.tensor([item["seq_length"]], dtype=torch.long),
            base_mask=pred_item["mask_2d"].unsqueeze(0),
            add_other_metrics=True,
        )

        # Add ID to metrics
        metric.update({"id": id_})
        metrics.append(metric)

    # Convert to DataFrame and set index
    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index("id", inplace=True)

    return metrics_df


def compute_average_metrics(
    metrics_df: pd.DataFrame, dataset: ProteinSequenceDataset, method: str = "simple"
) -> Dict[str, float]:
    """
    Compute average metrics across all sequences.

    Args:
        metrics_df: DataFrame with per-sequence metrics
        dataset: Dataset for extracting PDB information
        method: 'simple' for direct averaging, 'pdb_first' for averaging within PDB first

    Returns:
        Dictionary with averaged metrics
    """
    if method == "simple":
        return metrics_df.mean().to_dict()

    elif method == "pdb_first":
        metrics_with_pdb = metrics_df.copy()
        _, metrics_with_pdb["pdb_id"], _ = zip(*metrics_df.index.str.split("_"))
        pdb_averages = metrics_with_pdb.groupby("pdb_id").mean()
        return pdb_averages.mean().to_dict()

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_contacts_and_predictions(
    contact_map: np.ndarray,
    probas: np.ndarray,
    distances: np.ndarray,
    metrics: Dict[str, float],
    ax: plt.Axes,
    title: Optional[str] = None,
    mode: str = 'normal',
) -> None:
    """
    Plot single contact map with predictions overlay and metrics.
    Adapted from: https://github.com/rmrao/evo/blob/main/evo/visualize.py

    Args:
        contact_map: True contact map
        probas: Predicted contact probabilities
        distances: Distance map (not used currently)
        metrics: Dictionary with computed metrics for this sequence
        ax: Axes to plot on
        title: Optional title for the plot
    """
    seqlen = contact_map.shape[0]

    # Create relative distance matrix and masks (fixed diagonal like in original)
    relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
    bottom_mask = relative_distance < 0  # This masks the bottom triangle (original way)

    # Invalid mask for short-range contacts
    invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
    probas_work = probas.copy()
    probas_work[invalid_mask] = float("-inf")

    # Get top-L predictions
    topl_val = np.sort(probas_work.reshape(-1))[-seqlen]
    pred_contacts = probas_work >= topl_val

    # Calculate contact categories
    true_positives = contact_map & pred_contacts & ~bottom_mask
    false_positives = ~contact_map & pred_contacts & ~bottom_mask
    other_contacts = contact_map & ~pred_contacts & ~bottom_mask

    ms = max(1, int(40 / seqlen))
    if mode == 'colored':
        probas = probas_work*(contact_map*2 -1)
        cmap=plt.cm.get_cmap('bwr').reversed()
    else:
        cmap='Blues'
    # Plot contact map with overlay
    masked_predictions = np.ma.masked_where(bottom_mask, probas)
    ax.imshow(masked_predictions, cmap=cmap, norm=None)
    ax.plot(*np.where(other_contacts), "o", c="grey", ms=ms)
    ax.plot(*np.where(false_positives), "o", c="r", ms=ms)
    ax.plot(*np.where(true_positives), "o", c="b", ms=ms)

    # Add metrics to title (below ID)
    metrics_text = "\n".join(
        [f"{k}: {v:.2f}" for k, v in list(metrics.items()) if isinstance(v, float) or isinstance(v, int)]
    )  # Limit to first 6 metrics
    full_title = f"{title}\n{metrics_text}" if title else metrics_text
    ax.set_title(full_title, fontsize=9)
    ax.axis("square")
    ax.set_xlim([0, seqlen])
    ax.set_ylim([0, seqlen])


def create_contact_visualizations(
    dataset: ProteinSequenceDataset,
    prediction_dataset: PredictionDataset,
    metrics_df: pd.DataFrame,
    fixed_ids: Optional[List[str]] = None,
    n_samples: int = 3,
    mode: str = 'normal',
) -> plt.Figure:
    """
    Create contact visualization plots for selected sequences in a grid (3 per row).

    Args:
        dataset: Validation dataset
        prediction_dataset: Predictions dataset
        metrics_df: DataFrame with computed metrics
        fixed_ids: Optional list of specific IDs to visualize
        n_samples: Number of random samples if fixed_ids not provided

    Returns:
        Single figure with grid of contact plots
    """
    if fixed_ids is None:
        # Select random samples
        sample_indices = np.random.choice(
            len(dataset), size=min(n_samples, len(dataset)), replace=False
        )
        selected_ids = [dataset[i]["metadata"]["id"] for i in sample_indices]
    else:
        selected_ids = fixed_ids

    # Filter out IDs not found in dataset
    valid_ids = []
    valid_data = []

    for seq_id in selected_ids:
        # Find sequence in dataset
        data_item = None
        for item in dataset:
            if item["metadata"]["id"] == seq_id:
                data_item = item
                break

        if data_item is None:
            print(f"Warning: ID {seq_id} not found in dataset")
            continue

        if seq_id not in metrics_df.index:
            print(f"Warning: ID {seq_id} not found in metrics")
            continue

        valid_ids.append(seq_id)
        valid_data.append(data_item)

    n_plots = len(valid_ids)
    if n_plots == 0:
        print("No valid IDs found for visualization")
        return plt.figure()

    # Calculate grid dimensions (3 per row)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    # Create figure with subplots and reduced spacing
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 5 * nrows),
        dpi=600,
        gridspec_kw={
            "hspace": 1.1,
            "wspace": 0.3,
        },  # Increase vertical spacing for titles
    )

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes) if ncols > 1 else [axes]
    else:
        axes = axes.flatten()

    # Plot each sequence
    for i, (seq_id, data_item) in enumerate(zip(valid_ids, valid_data)):
        # Get prediction and metrics
        pred_item = prediction_dataset[seq_id]
        seq_metrics = metrics_df.loc[seq_id].to_dict()

        # Prepare data
        distance_map = data_item["distance_map"]
        if isinstance(distance_map, torch.Tensor):
            distance_map = distance_map.detach().cpu().numpy()
        contact_logits = pred_item["contact_logits"]
        if isinstance(contact_logits, torch.Tensor):
            contact_logits = contact_logits.detach().cpu().numpy()
        contact_map = (distance_map < 8).astype(int)
        probas = 1 / (1 + np.exp(-contact_logits))
        distances = distance_map

        # Plot on the i-th subplot
        plot_contacts_and_predictions(
            contact_map=contact_map,
            probas=probas,
            distances=distances,
            metrics=seq_metrics,
            ax=axes[i],
            title=seq_id,
            mode=mode,
        )

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    # plt.tight_layout()

    return fig


def add_cluster_information(
    metrics_df: pd.DataFrame, clusters: pd.DataFrame, dataset: ProteinSequenceDataset
) -> pd.DataFrame:
    """
    Add cluster size information to metrics DataFrame.

    Args:
        metrics_df: DataFrame with metrics
        clusters: DataFrame with cluster information
        dataset: Dataset (not used in current implementation)

    Returns:
        DataFrame with added cluster_size column
    """
    # Calculate cluster sizes
    clusters = clusters.copy()
    clusters["split"], clusters["pdb_id"], clusters["chain"] = zip(
        *clusters["id"].str.split("_")
    )
    cluster_sizes = (
        clusters[~clusters["id"].isin(metrics_df.index) & (clusters["split"] == 'train')]
        .groupby("cluster")['pdb_id'].nunique()
    )
    # Create ID to cluster mapping using the clusters dataframe directly
    id_cluster_map = clusters.set_index("id")
    id_cluster_map["size"] = cluster_sizes.reindex(id_cluster_map["cluster"]).fillna(1).values

    # Add cluster size to metrics
    metrics_with_clusters = metrics_df.copy()
    metrics_with_clusters["cluster_size"] = id_cluster_map.loc[metrics_df.index]["size"]

    return metrics_with_clusters


def create_metrics_lineplots(
    avg_metrics_simple: Dict[str, float], avg_metrics_pdb_first: Dict[str, float]
) -> plt.Figure:
    """
    Create line plots showing how averaged metrics change across distance ranges.

    Args:
        avg_metrics_simple: Simple averaged metrics
        avg_metrics_pdb_first: PDB-first averaged metrics

    Returns:
        matplotlib Figure object
    """
    # Define distance ranges and metrics
    distance_ranges = ["short", "medium", "long"]
    roc_metrics = ["roc_auc_short", "roc_auc_medium", "roc_auc_long"]
    precision_l1_metrics = [
        "precision_short@L/1",
        "precision_medium@L/1",
        "precision_long@L/1",
    ]
    precision_l2_metrics = [
        "precision_short@L/2",
        "precision_medium@L/2",
        "precision_long@L/2",
    ]
    precision_l5_metrics = [
        "precision_short@L/5",
        "precision_medium@L/5",
        "precision_long@L/5",
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=600)

    # Colors for different metrics
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot 1: Simple averaging
    ax1.plot(
        distance_ranges,
        [avg_metrics_simple[m] for m in roc_metrics],
        "o-",
        color=colors[0],
        label="ROC AUC",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        distance_ranges,
        [avg_metrics_simple[m] for m in precision_l1_metrics],
        "o-",
        color=colors[1],
        label="Precision@L/1",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        distance_ranges,
        [avg_metrics_simple[m] for m in precision_l2_metrics],
        "o-",
        color=colors[2],
        label="Precision@L/2",
        linewidth=2,
        markersize=6,
    )
    ax1.plot(
        distance_ranges,
        [avg_metrics_simple[m] for m in precision_l5_metrics],
        "o-",
        color=colors[3],
        label="Precision@L/5",
        linewidth=2,
        markersize=6,
    )

    ax1.set_xlabel("Distance Range")
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Simple Averaging")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: PDB-first averaging (same colors for consistency)
    ax2.plot(
        distance_ranges,
        [avg_metrics_pdb_first[m] for m in roc_metrics],
        "o-",
        color=colors[0],
        label="ROC AUC",
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        distance_ranges,
        [avg_metrics_pdb_first[m] for m in precision_l1_metrics],
        "o-",
        color=colors[1],
        label="Precision@L/1",
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        distance_ranges,
        [avg_metrics_pdb_first[m] for m in precision_l2_metrics],
        "o-",
        color=colors[2],
        label="Precision@L/2",
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        distance_ranges,
        [avg_metrics_pdb_first[m] for m in precision_l5_metrics],
        "o-",
        color=colors[3],
        label="Precision@L/5",
        linewidth=2,
        markersize=6,
    )

    ax2.set_xlabel("Distance Range")
    ax2.set_ylabel("Metric Value")
    ax2.set_title("PDB-first Averaging")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()

    return fig


def create_metrics_boxplots(metrics_df: pd.DataFrame, bins=None, labels=None, split_column=None, plt_type='boxplot', plot_additional_metrics=False) -> plt.Figure:
    """
    Create boxplot visualization of metrics by cluster size bins.
    Creates 4 columns x 4 rows = 16 subplots for different metrics and distance ranges.

    Args:
        metrics_df: DataFrame with metrics and cluster_size column

    Returns:
        Figure with boxplots - does not display
    """
    assert plt_type in ['boxplot', 'violinplot']
    # Define cluster size bins (matching notebook)
    if bins is None:
        bins = [0, 1, 2, 3, 4, 5, 7, 10, 20, 40, np.inf]
    if labels is None:
        labels = ["1", "2", "3", "4", "5", "5-7", "7-10", "10-20", "20-40", "40+"]
    if split_column is None:
        split_column = "cluster_size"

    # Add binned cluster size
    metrics_binned = metrics_df.copy()
    metrics_binned["split_binned"] = pd.cut(
        metrics_binned[split_column], bins=bins, labels=labels
    )

    # Define the metrics to plot (4 columns)
    metric_columns = ["roc_auc", "precision@L/5", "precision@L/2", "precision@L/1"]
    additional_metrics = ['precision', 'recall', 'f1']
    metric_columns = additional_metrics if plot_additional_metrics else metric_columns

    # Define distance ranges (4 rows)
    distance_ranges = ["short", "medium", "long", "full"]

    # Create subplots
    fig, axes = plt.subplots(len(distance_ranges), len(metric_columns), figsize=(16, 12), dpi=600)
    fig.suptitle(f"Metrics by {split_column}", fontsize=16)

    for row, distance_range in enumerate(distance_ranges):
        for col, base_metric in enumerate(metric_columns):
            ax = axes[row, col]
            if '@' in base_metric:
                base_metric, L_fraction = base_metric.split('@')
                metric_name = f"precision_{distance_range}@{L_fraction}"
            else:
                metric_name = f"{base_metric}_{distance_range}"
            
            # Create boxplot
            if metric_name in metrics_binned.columns:
                if plt_type == 'boxplot':
                    sns.boxplot(
                        y=metrics_binned[metric_name],
                        x=metrics_binned["split_binned"],
                        order=labels,
                        ax=ax,
                    )
                elif plt_type == 'violinplot':
                    sns.violinplot(
                        y=metrics_binned[metric_name],
                        x=metrics_binned["split_binned"],
                        order=labels,
                        ax=ax,
                    )
                ax.set_ylim(-0.1, 1.1)
                ax.set_title(f"{distance_range.capitalize()} - {metric_name}")
                ax.set_xlabel(split_column)
                ax.set_ylabel(metric_name)

                # Rotate x-axis labels for better readability
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Metric\n{metric_name}\nnot found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{distance_range.capitalize()} - {base_metric}")

    plt.tight_layout()
    return fig


def run_full_evaluation(
    dataset_path: str,
    predictions_path: str,
    clusters_path: str,
    fixed_visualization_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.

    Args:
        dataset_path: Path to validation dataset
        predictions_path: Path to predictions pickle file
        clusters_path: Path to clusters TSV file
        fixed_visualization_ids: Optional list of IDs for visualization

    Returns:
        Dictionary with all evaluation results
    """
    print("Loading data...")
    dataset, prediction_dataset, clusters = load_data(
        dataset_path, predictions_path, clusters_path
    )

    print("Computing metrics...")
    metrics_df = compute_metrics_dataframe(dataset, prediction_dataset)

    print("Computing average metrics...")
    avg_metrics_simple = compute_average_metrics(metrics_df, dataset, method="simple")
    avg_metrics_pdb_first = compute_average_metrics(
        metrics_df, dataset, method="pdb_first"
    )

    print("Adding cluster information...")
    metrics_with_clusters = add_cluster_information(metrics_df, clusters, dataset)

    print("Creating visualizations...")
    contact_figure = create_contact_visualizations(
        dataset, prediction_dataset, metrics_df, fixed_ids=fixed_visualization_ids,
        mode='colored'
    )

    print("Creating line plots...")
    lineplot_figure = create_metrics_lineplots(
        avg_metrics_simple, avg_metrics_pdb_first
    )

    print("Creating boxplots...")
    boxplot_figure = create_metrics_boxplots(metrics_with_clusters)
    boxplot_figure_additional_metrics = create_metrics_boxplots(metrics_with_clusters, plot_additional_metrics=True)
    return {
        "metrics_df": metrics_df,
        "metrics_with_clusters": metrics_with_clusters,
        "avg_metrics_simple": avg_metrics_simple,
        "avg_metrics_pdb_first": avg_metrics_pdb_first,
        "contact_figure": contact_figure,
        "lineplot_figure": lineplot_figure,
        "boxplot_figure": boxplot_figure,
        "boxplot_figure_additional_metrics": boxplot_figure_additional_metrics,
        "dataset": dataset,
        "prediction_dataset": prediction_dataset,
        "clusters": clusters,
    }
