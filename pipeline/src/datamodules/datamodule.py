import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from datamodules.dataset import ProteinSequenceDataset
from registry import BaseDataModuleConfig, register_datamodule
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import logging

# import seed to make all sampling deterministic
import utils.seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SamplingConfig:
    cluster_path: str
    train_n_pdb: float = 1.0  # if <=1 - percentage, else number of pdbs. percentage after apply intersection with val and test
    train_intersect_val_clusters: bool = False
    train_intersect_test_clusters: bool = False
    val_n_pdb: float = 1.0  # if <=1 - percentage, else number of pdbs
    test_n_pdb: float = 1.0  # if <=1 - percentage, else number of pdbs
    train_max_chains_per_pdb: Optional[int] = None  # num of chains from each pdb
    val_max_chains_per_pdb: Optional[int] = None
    test_max_chains_per_pdb: Optional[int] = None


@dataclass
class ProteinDataModuleConfig(BaseDataModuleConfig):
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    tokenizer_name: str
    batch_size: int
    num_workers: int
    max_seq_length: int
    contact_threshold: float
    sampler: SamplingConfig
    include_b_factors: bool = False


class Sampler:
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.clusters = pd.read_csv(config.cluster_path, sep="\t", header=None)
        self.clusters.columns = ["cluster_id", "seq_id"]
        self.clusters[["split", "pdb_id", "chain"]] = self.clusters["seq_id"].str.split(
            "_", expand=True
        )

    def select_num_chains_from_pdb(self, df: pd.DataFrame, num_chains: int):
        df_pdb = df.groupby("pdb_id").apply(
            lambda x: x.sample(min(len(x), num_chains), replace=False)
        )
        return df_pdb["id"].unique()

    def select_num_pdbs_from_df(self, df: pd.DataFrame, num_pdbs: float):
        pdbs = df.pdb_id.unique()
        if num_pdbs <= 1:
            return np.random.choice(pdbs, size=int(len(pdbs) * num_pdbs), replace=False)
        else:
            return np.random.choice(pdbs, size=int(num_pdbs), replace=False)

    def sample_train(
        self, train_dataset: ProteinSequenceDataset, val_dataset: ProteinSequenceDataset
    ):
        df_train = train_dataset.metadata_to_df()
        print(df_train.head())
        df_train[["split", "pdb_id", "chain"]] = df_train["id"].str.split(
            "_", expand=True
        )
        df_val = val_dataset.metadata_to_df()
        print(df_val.head())
        df_val[["split", "pdb_id", "chain"]] = df_val["id"].str.split("_", expand=True)
        if not self.config.train_intersect_val_clusters:
            val_clusters = self.clusters.loc[self.clusters.seq_id.isin(df_val.id)][
                "cluster_id"
            ].unique()
            selected_seq_id = self.clusters.loc[
                ~self.clusters.cluster_id.isin(val_clusters)
            ]["seq_id"].unique()
            df_train = df_train[df_train.id.isin(selected_seq_id)]
            logger.info(
                f"Sample Train: After removing intersection with val clusters, train dataset size: {len(df_train)}"
            )

        if not self.config.train_intersect_test_clusters:
            test_clusters = self.clusters.loc[self.clusters["split"] == "test"][
                "cluster_id"
            ].unique()
            selected_seq_id = self.clusters.loc[
                ~self.clusters.cluster_id.isin(test_clusters)
            ]["seq_id"].unique()
            df_train = df_train[df_train.id.isin(selected_seq_id)]
            logger.info(
                f"Sample Train: After removing intersection with test clusters, train dataset size: {len(df_train)}"
            )

        if self.config.train_n_pdb:
            selected_pdb_ids = self.select_num_pdbs_from_df(
                df_train, self.config.train_n_pdb
            )
            df_train = df_train[df_train.pdb_id.isin(selected_pdb_ids)]
            logger.info(
                f"Sample Train: After selecting {self.config.train_n_pdb} n/frac of pdbs, train dataset size: {len(df_train)}"
            )

        if self.config.train_max_chains_per_pdb:
            selected_seq_ids = self.select_num_chains_from_pdb(
                df_train, self.config.train_max_chains_per_pdb
            )
            df_train = df_train[df_train.id.isin(selected_seq_ids)]
            logger.info(
                f"Sample Train: After selecting {self.config.train_max_chains_per_pdb} chains per pdb, train dataset size: {len(df_train)}"
            )
        selected_sequences = df_train.id.unique()
        logger.info(
            f"Sample Train: After sampling, train dataset size: {len(selected_sequences)}"
        )
        return train_dataset.filter_seq_ids(selected_sequences)

    def sample_val(self, val_dataset: ProteinSequenceDataset):
        df_val = val_dataset.metadata_to_df()
        df_val[["split", "pdb_id", "chain"]] = df_val["id"].str.split("_", expand=True)
        if self.config.val_n_pdb:
            selected_pdb_ids = self.select_num_pdbs_from_df(
                df_val, self.config.val_n_pdb
            )
            df_val = df_val[df_val.pdb_id.isin(selected_pdb_ids)]
            logger.info(
                f"Sample Val: After selecting {self.config.val_n_pdb} n/frac of pdbs, val dataset size: {len(df_val)}"
            )
        if self.config.val_max_chains_per_pdb:
            selected_seq_ids = self.select_num_chains_from_pdb(
                df_val, self.config.val_max_chains_per_pdb
            )
            df_val = df_val[df_val.id.isin(selected_seq_ids)]
        selected_sequences = df_val.id.unique()
        logger.info(
            f"Sample Val: After sampling, val dataset size: {len(selected_sequences)}"
        )
        return val_dataset.filter_seq_ids(selected_sequences)

    def sample_test(self, test_dataset: ProteinSequenceDataset):
        df_test = test_dataset.metadata_to_df()
        df_test[["split", "pdb_id", "chain"]] = df_test["id"].str.split(
            "_", expand=True
        )
        if self.config.test_n_pdb:
            selected_pdb_ids = self.select_num_pdbs_from_df(
                df_test, self.config.test_n_pdb
            )
            df_test = df_test[df_test.pdb_id.isin(selected_pdb_ids)]
            logger.info(
                f"Sample Test: After selecting {self.config.test_n_pdb} n/frac of pdbs, test dataset size: {len(df_test)}"
            )
        if self.config.test_max_chains_per_pdb:
            selected_seq_ids = self.select_num_chains_from_pdb(
                df_test, self.config.test_max_chains_per_pdb
            )
            df_test = df_test[df_test.id.isin(selected_seq_ids)]
        selected_sequences = df_test.id.unique()
        logger.info(
            f"Sample Test: After sampling, test dataset size: {len(selected_sequences)}"
        )
        return test_dataset.filter_seq_ids(selected_sequences)


@register_datamodule("protein_datamodule", ProteinDataModuleConfig)
class ProteinDataModule(L.LightningDataModule):
    """
    LightningDataModule for protein sequences with ESM-2 support.
    """

    def __init__(self, config: ProteinDataModuleConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.sampler = Sampler(config.sampler)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation and testing."""

        if stage == "fit" or stage is None:
            self.train_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.train_dataset_path
            )
            self.val_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.val_dataset_path
            )
            self.train_dataset = self.sampler.sample_train(
                self.train_dataset, self.val_dataset
            )
            self.val_dataset = self.sampler.sample_val(self.val_dataset)

        if stage == "test" or stage is None:
            self.test_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.test_dataset_path
            )
            self.test_dataset = self.sampler.sample_test(self.test_dataset)

    def train_dataloader(self) -> TorchDataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized. Call setup() first.")
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True,
        )

    def val_dataloader(self) -> TorchDataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise ValueError(
                "Validation dataset is not initialized. Call setup() first."
            )
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True,
        )

    def test_dataloader(self) -> TorchDataLoader:
        """Return test dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup() first.")
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True,
        )

    def predict_dataloader(self) -> TorchDataLoader:
        """Return prediction dataloader."""
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized. Call setup() first.")
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.get_collate_fn(),
            pin_memory=True,
        )

    def distance_to_contact_map(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """Transform distance matrix to contact map using threshold."""
        return (distance_matrix < self.config.contact_threshold).float()

    def get_collate_fn(self):
        def collate_fn(batch: list) -> Dict[str, Any]:
            """
            Custom collate function for batching protein sequences.
            Handles tokenization, padding, and optional fields.
            """
            sequences = [item["sequence"] for item in batch]
            seq_lengths = torch.tensor(
                [item["seq_length"] for item in batch], dtype=torch.long
            )
            metadata = [item["metadata"] for item in batch]

            # Tokenize sequences
            tokenized = self.tokenizer(
                sequences,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )

            batch_dict = {
                "sequences": sequences,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "seq_lengths": seq_lengths,
                "metadata": metadata,
            }

            # Handle b-factors if present and configured
            if self.config.include_b_factors:
                max_len = tokenized["input_ids"].shape[1]  # Use tokenized length
                b_factors_batch = torch.zeros(len(batch), max_len)
                for i, item in enumerate(batch):
                    if "b_factors" in item:
                        # Account for special tokens (<cls> and <eos>)
                        # Shift b-factors by 1 to account for <cls> token
                        length = min(
                            item["b_factors"].shape[0], max_len - 2
                        )  # -2 for special tokens
                        b_factors_batch[i, 1 : length + 1] = item["b_factors"][
                            :length
                        ]  # Shift by 1
                batch_dict["b_factors"] = b_factors_batch

            # Handle distance maps -> contact maps if present
            max_len = int(seq_lengths.max().item())  # Use original length
            assert max_len + 2 == tokenized["input_ids"].shape[1]
            contact_maps_batch = torch.zeros(len(batch), max_len, max_len)
            for i, item in enumerate(batch):
                if "distance_map" in item:
                    dist_map = item["distance_map"]
                    h, w = (
                        min(dist_map.shape[0], max_len),
                        min(dist_map.shape[1], max_len),
                    )
                    # Transform distance to contact map
                    contact_map = self.distance_to_contact_map(dist_map[:h, :w])
                    contact_maps_batch[i, :h, :w] = contact_map
            batch_dict["contact_maps"] = contact_maps_batch

            return batch_dict

        return collate_fn
