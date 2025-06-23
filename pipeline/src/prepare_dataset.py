import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datamodules.dataset import ProteinSequenceDataset
import logging
from simple_parsing import ArgumentParser
import utils.seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_validation_pdbs(json_path: str) -> list:
    """Load validation PDB IDs from JSON file."""
    with open(json_path, "r") as f:
        validation_pdbs = json.load(f)
    return validation_pdbs


@dataclass
class PrepareDatasetConfig:
    train_seq_path: str
    validation_pdbs_path: str
    test_seq_path: str
    output_train_path: str
    output_val_path: str
    output_test_path: str
    output_stats_path: str
    max_seq_length: int

def load_save_dataset(df: pd.DataFrame, max_seq_length: int, output_path: str) -> int:
    dataset = ProteinSequenceDataset.from_df(df, max_seq_length=max_seq_length)
    dataset.save_to_disk(output_path)
    logger.info(f"Loaded {len(dataset)} sequences and saved dataset to {output_path}")
    return len(dataset)

def main(cfg: PrepareDatasetConfig) -> None:
    """Main function."""

    print("=== Preparing Protein Datasets ===")

    # Load validation PDB IDs
    print("Loading validation PDB IDs...")
    validation_pdbs = load_validation_pdbs(cfg.validation_pdbs_path)
    print(f"Found {len(validation_pdbs)} validation PDB IDs")

    train_df = pd.read_parquet(cfg.train_seq_path)
    val_df = train_df.loc[train_df["pdb_id"].isin(validation_pdbs)]
    train_df = train_df.loc[~train_df["pdb_id"].isin(validation_pdbs)]
    test_df = pd.read_parquet(cfg.test_seq_path)

    # Load train dataset from parquet
    print("Loading train dataset from parquet...")
    train_size = load_save_dataset(train_df, cfg.max_seq_length, cfg.output_train_path)

    # Create validation dataset
    val_size = load_save_dataset(val_df, cfg.max_seq_length, cfg.output_val_path)

    # Load test dataset from parquet
    print("Loading test dataset from parquet...")
    test_size = load_save_dataset(test_df, cfg.max_seq_length, cfg.output_test_path)


    # Save stats
    stats = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "max_seq_length": cfg.max_seq_length,
        "validation_pdbs_count": len(validation_pdbs),
    }

    with open(cfg.output_stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Dataset preparation completed!")
    print(f"- Train: {train_size} sequences")
    print(f"- Val: {val_size} sequences")
    print(f"- Test: {test_size} sequences")
    print(f"Saved to: {cfg.output_train_path}, {cfg.output_val_path}, {cfg.output_test_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(PrepareDatasetConfig, dest="config")
    args = parser.parse_args()
    main(args.config)
