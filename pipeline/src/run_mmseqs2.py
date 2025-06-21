#!/usr/bin/env python3

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MMseqs2Config:
    """Configuration for MMseqs2 pipeline."""

    # Input/Output paths
    train_fasta: str
    test_fasta: str
    combined_fasta: str
    db_path: str

    # Clustering parameters
    cluster_coverage: float
    cluster_min_seq_id: float
    cluster_tmp_dir: str
    cluster_output_tsv: str

    # Search parameters
    search_coverage: float
    search_min_seq_id: float
    search_tmp_dir: str
    search_output_tsv: str


def empty_dir(path: str) -> None:
    """Empty directory."""
    if Path(path).exists():
        shutil.rmtree(path)
    os.makedirs(path)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], description: str) -> None:
    """Run shell command and handle errors."""
    try:
        logger.info(f"Running {description}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stderr:
            logger.warning(f"Command produced warnings: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {description}: {e.stderr}")
        raise


def combine_sequences(train_fasta: str, test_fasta: str, output_fasta: str) -> None:
    """Combine train and test sequences into one file."""
    ensure_dir(output_fasta)
    with open(output_fasta, "wb") as outfile:
        for infile in [train_fasta, test_fasta]:
            with open(infile, "rb") as f:
                shutil.copyfileobj(f, outfile)


def run_mmseqs2(cfg: MMseqs2Config) -> None:
    """Run MMseqs2 pipeline: clustering and similarity search."""
    # Combine sequences
    empty_dir(Path(cfg.combined_fasta).parent)
    logger.info(
        f"Combining sequences from {cfg.train_fasta} and {cfg.test_fasta} into {cfg.combined_fasta}"
    )
    combine_sequences(cfg.train_fasta, cfg.test_fasta, cfg.combined_fasta)
    logger.info(f"Combined sequences into {cfg.combined_fasta}")

    # Clean previous results
    if os.path.exists(cfg.db_path):
        shutil.rmtree(os.path.dirname(cfg.db_path))
    ensure_dir(cfg.db_path)

    # Create database
    logger.info(f"Creating database at {cfg.db_path}")
    run_command(
        ["mmseqs", "createdb", cfg.combined_fasta, cfg.db_path],
        "database creation",
    )

    # Run clustering
    logger.info(
        f"Running clustering with coverage {cfg.cluster_coverage} and min-seq-id {cfg.cluster_min_seq_id}"
    )
    run_command(
        [
            "mmseqs",
            "cluster",
            cfg.db_path,
            f"{cfg.db_path}_clust",
            cfg.cluster_tmp_dir,
            "-c",
            str(cfg.cluster_coverage),
            "--min-seq-id",
            str(cfg.cluster_min_seq_id),
        ],
        "clustering",
    )

    # Create clustering TSV
    ensure_dir(cfg.cluster_output_tsv)
    logger.info(f"Creating clustering TSV at {cfg.cluster_output_tsv}")
    run_command(
        [
            "mmseqs",
            "createtsv",
            cfg.db_path,
            cfg.db_path,
            f"{cfg.db_path}_clust",
            cfg.cluster_output_tsv,
        ],
        "clustering TSV creation",
    )

    # Run similarity search
    logger.info(
        f"Running similarity search with coverage {cfg.search_coverage} and min-seq-id {cfg.search_min_seq_id}"
    )
    run_command(
        [
            "mmseqs",
            "search",
            cfg.db_path,
            cfg.db_path,
            f"{cfg.db_path}_similarity",
            cfg.search_tmp_dir,
            "--min-seq-id",
            str(cfg.search_min_seq_id),
            "-c",
            str(cfg.search_coverage),
        ],
        "similarity search",
    )

    # Create similarity search TSV
    ensure_dir(cfg.search_output_tsv)
    logger.info(f"Creating similarity search TSV at {cfg.search_output_tsv}")
    run_command(
        [
            "mmseqs",
            "createtsv",
            cfg.db_path,
            cfg.db_path,
            f"{cfg.db_path}_similarity",
            cfg.search_output_tsv,
        ],
        "similarity search TSV creation",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(MMseqs2Config, dest="config")
    args = parser.parse_args()

    logger.info("Starting MMseqs2 pipeline...")
    logger.info(f"Configuration: {args.config}")

    try:
        run_mmseqs2(args.config)
        logger.info("MMseqs2 pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
