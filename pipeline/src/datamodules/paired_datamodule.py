import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch

# import seed to make all sampling deterministic
import utils.seed
from datamodules.datamodule import (
    ProteinDataModule,
    ProteinDataModuleConfig,
    ProteinSequenceDataset,
    Sampler,
)
from datamodules.paired_dataset import PairedDataset
from registry import register_datamodule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PairedProteinDataModuleConfig(ProteinDataModuleConfig):
    similarity_file_path: str = ""
    min_similarity_threshold: float = 0.3
    max_similarity_threshold: int = 0.95


@register_datamodule("paired_protein_datamodule", PairedProteinDataModuleConfig)
class PairedProteinDataModule(ProteinDataModule):
    def __init__(self, config: PairedProteinDataModuleConfig):
        super().__init__(config)
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_similarity_df(self, similarity_file_path: str) -> pd.DataFrame:
        df = pd.read_csv(similarity_file_path, sep="\t", header=None)
        df.columns = [
            "query_id",
            "target_id",
            "score",
            "identity",
            "evalue",
            "q_start",
            "q_end",
            "q_len",
            "t_start",
            "t_end",
            "t_len",
        ]
        df[["query_split", "query_pdb_id", "query_chain"]] = df["query_id"].str.split(
            "_", expand=True
        )
        df[["target_split", "target_pdb_id", "target_chain"]] = df[
            "target_id"
        ].str.split("_", expand=True)
        return df

    def similarity_df_to_pairs(
        self,
        similarity_df: pd.DataFrame,
        min_similarity_threshold: float,
    ) -> pd.DataFrame:
        similarity_df = similarity_df[
            similarity_df["query_pdb_id"] != similarity_df["target_pdb_id"]
        ]
        similarity_df = similarity_df[similarity_df["target_split"] != "test"]
        similarity_df = similarity_df[
            similarity_df["identity"] >= min_similarity_threshold
        ]
        similarity_df = similarity_df[
            similarity_df["identity"] <= self.config.max_similarity_threshold
        ]
        similarity_df = (
            similarity_df.sort_values(by="identity", ascending=False)
            .drop_duplicates(subset=["query_id"], keep="first")
            .set_index("query_id")
        )
        logger.info(
            f"Num train in queries: {len(similarity_df[similarity_df['query_split'] == 'train'])}"
        )
        logger.info(
            f"Num test in queries: {len(similarity_df[similarity_df['query_split'] == 'test'])}"
        )
        logger.info(f"Loaded {len(similarity_df)} similarity pairs: ")
        similarity_df = similarity_df[["target_id", "identity"]]

        logger.info(f"{similarity_df.sample(10)}")
        pairs = similarity_df[["identity", "target_id"]].to_dict("index")
        return pairs

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            similarity_df = self.load_similarity_df(self.config.similarity_file_path)
            pairs = self.similarity_df_to_pairs(
                similarity_df, self.config.min_similarity_threshold
            )
            single_train_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.train_dataset_path
            )
            single_val_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.val_dataset_path
            )
            filtered_train_dataset = self.sampler.sample_train(
                single_train_dataset, single_val_dataset
            )
            filtered_val_dataset = self.sampler.sample_val(single_val_dataset)
            paired_train_dataset = PairedDataset(
                filtered_train_dataset,
                single_train_dataset,
                pairs,
            )
            paired_val_dataset = PairedDataset(
                filtered_val_dataset,
                single_train_dataset,
                pairs,
            )
            self.train_dataset = paired_train_dataset
            self.val_dataset = paired_val_dataset
        if stage == "test" or stage is None:
            similarity_df = self.load_similarity_df(self.config.similarity_file_path)
            pairs = self.similarity_df_to_pairs(
                similarity_df, self.config.min_similarity_threshold
            )
            single_test_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.test_dataset_path
            )
            single_train_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.train_dataset_path
            )
            filtered_test_dataset = self.sampler.sample_test(single_test_dataset)
            paired_test_dataset = PairedDataset(
                filtered_test_dataset,
                single_train_dataset,
                pairs,
            )
            self.test_dataset = paired_test_dataset

    def get_collate_fn(self):
        super_collate_fn = super().get_collate_fn()

        def collate_fn(batch: list) -> Dict[str, Any]:
            """
            Custom collate function for paired protein sequences.
            Handles tokenization for both primary and similar sequences.
            """
            # Since batch_size=1, we only have one item
            assert len(batch) == 1, f"Expected batch_size=1, got {len(batch)}"
            item = batch[0]
            print(batch)

            # Extract sequences
            primary_sequence = item["primary_sequence"]
            similar_sequence = item["similar_sequence"]
            similarity_score = item["similarity_score"]

            # Initialize result dictionary
            result = {
                "primary_sequence": super_collate_fn(
                    [
                        primary_sequence,
                    ]
                ),
                "similar_sequence": super_collate_fn(
                    [
                        similar_sequence,
                    ]
                ) if similar_sequence is not None else None,
                "similarity_score": similarity_score,
            }

            return result

        return collate_fn
