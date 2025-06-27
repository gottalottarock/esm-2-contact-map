import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datamodules.dataset import compute_distance_matrix, ProteinSequenceDataset

import logging

logger = logging.getLogger(__name__)


class PairedDataset(Dataset):
    def __init__(
        self,
        dataset: ProteinSequenceDataset,
        similar_dataset: ProteinSequenceDataset,
        pairs_map: dict[str, dict[str, Any]],
    ):
        logger.info(
            f"Initializing PairedDataset with {len(dataset)} sequences and {len(similar_dataset)} similar sequences"
        )
        self.dataset = dataset
        self.similar_dataset = similar_dataset
        dataset_seq_ids = set(self.dataset.get_ids())
        similar_seq_ids = set(self.similar_dataset.get_ids())
        pairs_map = {k: v for k, v in pairs_map.items() if k in dataset_seq_ids}
        pairs_map = {k: v for k, v in pairs_map.items() if v['target_id'] in similar_seq_ids}
        self.pairs_map = pairs_map
        logger.info(f"Filtered similar pairs map to {len(self.pairs_map)} pairs")
        required_similar_ids = set([v['target_id'] for v in pairs_map.values()])
        self.similar_dataset = self.similar_dataset.filter_seq_ids(
            list(required_similar_ids)
        )
        logger.info(
            f"Filtered similar dataset to {len(self.similar_dataset)} sequences"
        )
        self._self_similar_map = {
            item["metadata"]["id"]: idx for idx, item in enumerate(self.similar_dataset)
        }
        logger.info(
            f"Dataset has {len(self.dataset)} sequences and {len(self.pairs_map)} of them have similar sequence."
        )

    @classmethod
    def from_df(
        cls,
        df_dataset: pd.DataFrame,
        df_similar_dataset: pd.DataFrame,
        pairs_map: dict[str, dict[str, Any]],
    ):
        dataset = ProteinSequenceDataset.from_df(df_dataset)
        similar_dataset = ProteinSequenceDataset.from_df(df_similar_dataset)
        return cls(dataset, similar_dataset, pairs_map)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        similar_id = self.pairs_map.get(item["metadata"]["id"], None)
        if similar_id is None:
            return {
                "primary_sequence": item,
                "similar_sequence": None,
                "similarity_score": None,
            }
        similar_idx = self._self_similar_map[similar_id['target_id']]
        similar_item = self.similar_dataset[similar_idx]
        return {
            "primary_sequence": item,
            "similar_sequence": similar_item,
            "similarity_score": similar_id["identity"],
        }

    def filter_seq_ids(self, ids: List[str]):
        self.dataset = self.dataset.filter_seq_ids(ids)
        if len(self.dataset) == 0:
            raise ValueError("No sequences left in the dataset")
        return PairedDataset(self.dataset, self.similar_dataset, self.pairs_map)

    def filter_pdb_ids(self, pdb_ids: List[str]):
        self.dataset = self.dataset.filter_pdb_ids(pdb_ids)
        if len(self.dataset) == 0:
            raise ValueError("No sequences left in the dataset")
        return PairedDataset(self.dataset, self.similar_dataset, self.pairs_map)

    def metadata_to_df(self) -> pd.DataFrame:
        """
        Convert metadata to DataFrame.
        """
        return pd.DataFrame.from_records(self.metadata)
