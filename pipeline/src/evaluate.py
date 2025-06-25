"""
Evaluation script for contact prediction model.
"""

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import wandb

# import seed to make all sampling deterministic
import utils.seed

from simple_parsing import ArgumentParser, ArgumentGenerationMode, NestedMode
from utils.evaluation import run_full_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluateConfig:
    # Data paths
    dataset_path: str
    predictions_path: str
    clusters_path: str

    # Output directory
    output_dir: str

    # Fixed IDs for visualization (optional)
    fixed_visualization_ids: Optional[List[str]] = None

    # WandB settings
    wo_wandb: bool = False
    wandb_run_dir: Optional[str] = (
        None  # If None, will look for ${output_dir}.parent/train_checkpoints/latest-run
    )


def get_wandb_run_id(wandb_run_dir: Optional[str], output_dir: str) -> Optional[str]:
    """Get WandB run ID from latest-run file."""
    if wandb_run_dir:
        latest_run_path = Path(wandb_run_dir) / "latest-run"
    else:
        latest_run_path = (
            Path(output_dir).parent / "train_checkpoints" / "wandb" / "latest-run"
        )

    if latest_run_path.exists():
        if latest_run_path.is_symlink():
            target = latest_run_path.readlink()
            run_id = target.stem.split("-")[-1]
        else:
            with open(latest_run_path, "r") as f:
                run_id = f.read().strip().split("-")[-1]
        logger.info(f"Found WandB run ID: {run_id}")
        return run_id
    else:
        logger.warning(f"WandB latest-run file not found at {latest_run_path}")
        return None


def main(cfg: EvaluateConfig):
    # Create output directory
    if not cfg.wo_wandb:
        run_id = get_wandb_run_id(cfg.wandb_run_dir, cfg.output_dir)

        if run_id:
            # Resume existing run
            wandb.init(id=run_id, resume="must", project='deep-origin-task')
            logger.info(f"Resumed WandB run: {run_id}")
        else:
            raise ValueError("WandB run ID not found")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation...")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Predictions: {cfg.predictions_path}")
    logger.info(f"Clusters: {cfg.clusters_path}")
    logger.info(f"Output directory: {cfg.output_dir}")

    # Run full evaluation
    results = run_full_evaluation(
        dataset_path=cfg.dataset_path,
        predictions_path=cfg.predictions_path,
        clusters_path=cfg.clusters_path,
        fixed_visualization_ids=cfg.fixed_visualization_ids,
    )

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "avg_metrics_simple": results["avg_metrics_simple"],
                "avg_metrics_pdb_first": results["avg_metrics_pdb_first"],
            },
            f,
            indent=2,
        )
    logger.info(f"Saved metrics to {metrics_path}")

    # Save metrics DataFrame
    metrics_df_path = output_dir / "metrics_dataframe.csv"
    results["metrics_df"].to_csv(metrics_df_path)
    logger.info(f"Saved metrics DataFrame to {metrics_df_path}")

    # Save metrics with clusters
    metrics_clusters_path = output_dir / "metrics_with_clusters.csv"
    results["metrics_with_clusters"].to_csv(metrics_clusters_path)
    logger.info(f"Saved metrics with clusters to {metrics_clusters_path}")

    # Save visualizations
    contact_plot_path = output_dir / "contact_visualizations.png"
    results["contact_figure"].savefig(contact_plot_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved contact visualizations to {contact_plot_path}")

    lineplot_path = output_dir / "metrics_lineplots.png"
    results["lineplot_figure"].savefig(lineplot_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved line plots to {lineplot_path}")

    boxplot_path = output_dir / "metrics_boxplots.png"
    results["boxplot_figure"].savefig(boxplot_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved box plots to {boxplot_path}")

    boxplot_path_additional_metrics = output_dir / "metrics_boxplots_additional.png"
    results["boxplot_figure_additional_metrics"].savefig(boxplot_path_additional_metrics, dpi=600, bbox_inches="tight")
    logger.info(f"Saved box plots to {boxplot_path_additional_metrics}")

    # Initialize WandB and log results
    if not cfg.wo_wandb:

        # Log metrics
        wandb.log(
            {
                f"val-avg_metrics_simple/{k}": v
                for k, v in results["avg_metrics_simple"].items()
            }
        )
        wandb.log(
            {
                f"val-avg_metrics_pdb_first/{k}": v
                for k, v in results["avg_metrics_pdb_first"].items()
            }
        )

        # Log images
        wandb.log(
            {
                "val-contact_visualizations": wandb.Image(str(contact_plot_path)),
                "val-metrics_lineplots": wandb.Image(str(lineplot_path)),
                "val-metrics_boxplots": wandb.Image(str(boxplot_path)),
                "val-metrics_boxplots_additional": wandb.Image(str(boxplot_path_additional_metrics)),
            }
        )

        # Log files
        # wandb.save(str(metrics_path))
        # wandb.save(str(metrics_df_path))
        # wandb.save(str(metrics_clusters_path))

        logger.info("Logged results to WandB")
        wandb.finish()

    logger.info("Evaluation completed successfully!")

    return results


if __name__ == "__main__":
    parser = ArgumentParser(
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    parser.add_arguments(EvaluateConfig, dest="config")
    args = parser.parse_args()
    main(args.config)
