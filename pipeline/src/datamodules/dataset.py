import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def compute_distance_matrix(coords_ca: np.ndarray) -> np.ndarray:
    """
    Compute distance matrix from CA coordinates.

    Args:
        coords_ca: Array of shape (L, 3) where L is sequence length

    Returns:
        Distance matrix of shape (L, L)
    """
    coords_ca = np.array(coords_ca)
    if coords_ca.ndim == 1:
        # Reshape flat coordinates to (L, 3)
        coords_ca = coords_ca.reshape(-1, 3)

    # Compute pairwise distances
    diff = coords_ca[:, np.newaxis, :] - coords_ca[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    return distances


class ProteinSequenceDataset(Dataset):
    """
    Dataset for protein sequences with b-factors, distance maps and metadata.
    Can be constructed from seq.parquet and saved/loaded from disk.
    """

    def __init__(
        self,
        sequences: list,
        b_factors: list,
        distance_maps: list,
        metadata: list,
        max_seq_length: int = 1024,
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of protein sequences (strings)
            b_factors: List of b-factor arrays for each sequence
            distance_maps: List of distance map matrices for each sequence
            metadata: List of metadata dictionaries for each sequence
            max_seq_length: Maximum sequence length for padding/truncation
            cache_dir: Directory to cache processed data
        """
        self.sequences = sequences
        self.b_factors = b_factors
        self.distance_maps = distance_maps
        self.metadata = metadata
        self.max_seq_length = max_seq_length
        # Validate data consistency
        assert (
            len(sequences) == len(b_factors) == len(distance_maps) == len(metadata)
        ), "All data lists must have the same length"

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        max_seq_length: int = 1024,
    ) -> "ProteinSequenceDataset":
        """
        Create dataset from parquet file.

        Args:
            df: DataFrame containing sequences, b-factors, and distance maps
            max_seq_length: Maximum sequence length

        Returns:
            ProteinSequenceDataset instance
        """

        # Extract required columns
        sequences = df["sequence"].tolist()
        b_factors = (
            df["b_factors"].tolist()
            if "b_factors" in df.columns
            else [None] * len(sequences)
        )

        # Compute distance maps from CA coordinates if available
        distance_maps = []

        for coords in tqdm(df["coords_ca"].tolist(), desc="Computing distance maps"):
            distance_maps.append(compute_distance_matrix(coords))

        # Create metadata from remaining columns
        metadata_columns = [
            col
            for col in df.columns
            if col not in ["sequence", "b_factors", "coords_ca"]
        ]

        # Convert to records format properly
        if metadata_columns:
            metadata = df[metadata_columns].to_dict("records")
        else:
            metadata = [{} for _ in range(len(sequences))]

        return cls(
            sequences=sequences,
            b_factors=b_factors,
            distance_maps=distance_maps,
            metadata=metadata,
            max_seq_length=max_seq_length,
        )

    def save_to_disk(self, save_path: str):
        """Save dataset to disk using pickle."""
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "sequences": self.sequences,
            "b_factors": self.b_factors,
            "distance_maps": self.distance_maps,
            "metadata": self.metadata,
            "max_seq_length": self.max_seq_length,
        }

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_disk(cls, load_path: str) -> "ProteinSequenceDataset":
        """Load dataset from disk."""
        with open(load_path, "rb") as f:
            data = pickle.load(f)

        return cls(**data)

    def filter_seq_ids(self, seq_ids: List[str]) -> "ProteinSequenceDataset":
        """
        Filter dataset by sequence IDs.

        Args:
            seq_ids: List of sequence IDs to keep

        Returns:
            New ProteinSequenceDataset with filtered data
        """
        # Find indices of sequences to keep
        indices_to_keep = []
        for i, metadata in enumerate(self.metadata):
            if "id" in metadata and metadata["id"] in seq_ids:
                indices_to_keep.append(i)

        if len(indices_to_keep) == 0:
            raise ValueError("No sequences found in the dataset.")

        # Filter all data arrays
        filtered_sequences = [self.sequences[i] for i in indices_to_keep]
        filtered_b_factors = [self.b_factors[i] for i in indices_to_keep]
        filtered_distance_maps = [self.distance_maps[i] for i in indices_to_keep]
        filtered_metadata = [self.metadata[i] for i in indices_to_keep]

        # Create new dataset with filtered data
        return ProteinSequenceDataset(
            sequences=filtered_sequences,
            b_factors=filtered_b_factors,
            distance_maps=filtered_distance_maps,
            metadata=filtered_metadata,
            max_seq_length=self.max_seq_length,
        )

    def filter_pdb_ids(self, pd_ids: List[str]) -> "ProteinSequenceDataset":
        """
        Filter dataset by PDB IDs.
        """
        indices_to_keep = []
        for i, metadata in enumerate(self.metadata):
            if "pdb_id" in metadata and metadata["pdb_id"] in pd_ids:
                indices_to_keep.append(i)

        if len(indices_to_keep) == 0:
            raise ValueError("No sequences found in the dataset.")

        filtered_sequences = [self.sequences[i] for i in indices_to_keep]
        filtered_b_factors = [self.b_factors[i] for i in indices_to_keep]
        filtered_distance_maps = [self.distance_maps[i] for i in indices_to_keep]
        filtered_metadata = [self.metadata[i] for i in indices_to_keep]

        return ProteinSequenceDataset(
            sequences=filtered_sequences,
            b_factors=filtered_b_factors,
            distance_maps=filtered_distance_maps,
            metadata=filtered_metadata,
            max_seq_length=self.max_seq_length,
        )

    def get_ids(self) -> List[str]:
        """
        Get list of all sequence IDs in the dataset.

        Returns:
            List of sequence IDs
        """
        ids = []
        for metadata in self.metadata:
            if "id" in metadata:
                ids.append(metadata["id"])
            else:
                ids.append(None)
        return ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.

        Returns:
            Dictionary containing:
                - sequence: protein sequence string
                - b_factors: b-factor values (if available)
                - distance_map: distance map matrix (if available)
                - metadata: metadata dictionary
                - seq_length: original sequence length
        """
        sequence = self.sequences[idx]
        seq_length = len(sequence)

        # Truncate or pad sequence if needed
        if len(sequence) > self.max_seq_length:
            sequence = sequence[: self.max_seq_length]

        item = {
            "sequence": sequence,
            "seq_length": min(seq_length, self.max_seq_length),
            "metadata": self.metadata[idx],
        }

        # Add b-factors if available
        if self.b_factors[idx] is not None:
            b_factors = np.array(self.b_factors[idx])
            if len(b_factors) > self.max_seq_length:
                b_factors = b_factors[: self.max_seq_length]
            item["b_factors"] = torch.tensor(b_factors, dtype=torch.float32)

        # Add distance map if available
        if self.distance_maps[idx] is not None:
            distance_map = np.array(self.distance_maps[idx])
            max_len = min(distance_map.shape[0], self.max_seq_length)
            distance_map = distance_map[:max_len, :max_len]
            item["distance_map"] = torch.tensor(distance_map, dtype=torch.float32)

        return item
