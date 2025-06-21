import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from datamodules.dataset import ProteinSequenceDataset

@dataclass
class DataLoaderConfig:
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    batch_size: int = 32
    num_workers: int = 4
    max_seq_length: int = 1024
    tokenizer_name: str = "facebook/esm2_t33_650M_UR50D"
    include_b_factors: bool = True
    contact_threshold: float = 8.0


class ProteinDataModule(L.LightningDataModule):
    """
    LightningDataModule for protein sequences with ESM-2 support.
    """

    def __init__(self, config: DataLoaderConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation and testing."""

        if stage == "fit" or stage is None:
            self.train_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.train_dataset_path
            )
            self.val_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.val_dataset_path
            )

        if stage == "test" or stage is None:
            self.test_dataset = ProteinSequenceDataset.load_from_disk(
                self.config.test_dataset_path
            )

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
            seq_lengths = torch.tensor([item["seq_length"] for item in batch])
            metadata = [item["metadata"] for item in batch]

            # Tokenize sequences
            tokenized = self.tokenizer(
                sequences, 
                padding=True, 
                truncation=True, 
                max_length=self.config.max_seq_length,
                return_tensors="pt"
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
                        length = min(item["b_factors"].shape[0], max_len)
                        b_factors_batch[i, :length] = item["b_factors"][:length]
                batch_dict["b_factors"] = b_factors_batch

            # Handle distance maps -> contact maps if present  
            max_len = tokenized["input_ids"].shape[1]  # Use tokenized length
            contact_maps_batch = torch.zeros(len(batch), max_len, max_len)
            for i, item in enumerate(batch):
                if "distance_map" in item:
                    dist_map = item["distance_map"]
                    h, w = min(dist_map.shape[0], max_len), min(dist_map.shape[1], max_len)
                    # Transform distance to contact map
                    contact_map = self.distance_to_contact_map(dist_map[:h, :w])
                    contact_maps_batch[i, :h, :w] = contact_map
            batch_dict["contact_maps"] = contact_maps_batch

            return batch_dict
        return collate_fn

    def predict_dataloader(self) -> TorchDataLoader:
        """Return prediction dataloader (same as test)."""
        return self.test_dataloader()
