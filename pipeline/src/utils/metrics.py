"""
Contact prediction metrics module.
Implements ROC-AUC and Precision@L metrics for different contact distance ranges.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torcheval.metrics.functional import binary_auroc


class ContactPredictionMetrics:
    """Contact prediction metrics calculator for different distance ranges."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def create_distance_range_masks(
        self, mask_2d: torch.Tensor, from_index: int, to_index: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create masks for different contact distance ranges.

        Args:
            mask_2d: Mask for valid positions, shape (batch_size, max_len, max_len)
            from_index: Minimum distance
            to_index: Maximum distance for short range(optional)

        Returns:
            Mask for valid positions, shape (batch_size, max_len, max_len)
        """
        device = mask_2d.device
        _, max_len, _ = mask_2d.shape
        separation = torch.arange(max_len, device=device).unsqueeze(1) - torch.arange(
            max_len, device=device
        ).unsqueeze(0)
        mask_dist = separation >= from_index
        if to_index is not None:
            mask_dist = mask_dist & (separation < to_index)
        return mask_2d & mask_dist

    def precision_topk(
        self, predictions: torch.Tensor, true_labels: torch.Tensor, k: int
    ) -> float:
        """
        Calculate precision@k for given predictions and true labels.

        Args:
            predictions: Predicted probabilities, shape (max_len, )
             true_labels: True labels, shape (max_len, )
            k: Number of top predictions to consider

        Returns:
            Precision@k score
        """
        sorted_indices = torch.argsort(predictions, stable=True, descending=True)
        sorted_indices = sorted_indices[:k]
        sorted_true_labels = true_labels[sorted_indices]
        return float(sorted_true_labels.sum()) / k

    def precision_recall_f1(
        self, predictions: torch.Tensor, true_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for given predictions and true labels.
        """
        TP = (predictions * true_labels).sum().item()
        pred_sum = predictions.sum().item()
        true_sum = true_labels.sum().item()
        precision = TP / (pred_sum + 1e-10)
        recall = TP / (true_sum + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return {"precision": precision, "recall": recall, "f1": f1}

    def masked_other_metrics(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        return self.precision_recall_f1(predictions[mask], true_labels[mask])

    def masked_average_other_metrics(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        mask: torch.Tensor,
        average: bool = True,
        threshold: float = 0.5,
    ) -> Dict[str, float] | Dict[str, np.ndarray]:
        metrics = []
        contact_preds = torch.sigmoid(contact_logits) > threshold
        for contact_pred_seq, contact_map_seq, mask_seq in zip(
            contact_preds, contact_maps, mask
        ):
            metric = self.masked_other_metrics(
                contact_pred_seq, contact_map_seq, mask_seq
            )
            metrics.append(metric)
        metrics_dict = {
            k: np.array([item[k] for item in metrics]) for k in metrics[0].keys()
        }
        if average:
            return {k: np.mean(metrics_dict[k]).item() for k in metrics_dict.keys()}
        else:
            return metrics_dict

    def masked_precision_topk(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        mask: torch.Tensor,
        k: int,
    ) -> float:
        """
        Calculate precision@k for given predictions and true labels.

        Args:
            contact_logits: Predicted contact logits, shape (batch_size, max_len, max_len)
            contact_maps: True contact maps, shape (batch_size, max_len, max_len)
            mask: Valid positions mask, shape (batch_size, max_len, max_len)
            k: Number of top predictions to consider

        Returns:
            Precision@k score
        """
        return self.precision_topk(contact_logits[mask], contact_maps[mask], k)

    def masked_roc_auc(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Calculate ROC-AUC for given mask.

        Args:
            contact_logits: Predicted contact probabilities
            contact_maps: True contact maps
            mask: Valid positions mask

        Returns:
            ROC-AUC score
        """
        return binary_auroc(contact_logits[mask], contact_maps[mask]).item()

    def masked_average_precision_l_fraction(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        L_fraction: float,
        average: bool = True,
    ) -> float | np.ndarray:
        """
        Calculate average precision for given predictions and true labels.

        Args:
            true_labels: True labels, shape (batch_size, max_len, max_len)
            predictions: Predicted probabilities, shape (batch_size, max_len, max_len)
            mask: Valid positions mask, shape (batch_size, max_len, max_len)
            length: Sequence length
            L_fraction: Fraction of sequence length to consider

        Returns:
            Average precision score
        """
        precisions = []
        for contact_logit_seq, contact_map_seq, mask_seq, length in zip(
            contact_logits, contact_maps, mask, lengths
        ):
            precision = self.masked_precision_topk(
                contact_logit_seq, contact_map_seq, mask_seq, k=int(length * L_fraction)
            )
            precisions.append(precision)
        if average:
            return np.mean(precisions).item()
        else:
            return np.array(precisions)

    def masked_average_roc_auc(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        mask: torch.Tensor,
        average: bool = True,
    ) -> float | np.ndarray:
        """
        Calculate average ROC-AUC for given predictions and true labels.
        """
        roc_aucs = []
        for contact_logit_seq, contact_map_seq, mask_seq in zip(
            contact_logits, contact_maps, mask
        ):
            roc_auc = self.masked_roc_auc(contact_logit_seq, contact_map_seq, mask_seq)
            roc_aucs.append(roc_auc)
        if average:
            return np.mean(roc_aucs).item()
        else:
            return np.array(roc_aucs)

    def compute_all_metrics(
        self,
        contact_logits: torch.Tensor,
        contact_maps: torch.Tensor,
        seq_lengths: torch.Tensor,
        base_mask: torch.Tensor,
        average: bool = True,
        add_other_metrics: bool = False,
    ) -> Dict[str, float] | Dict[str, np.ndarray]:
        """
        Compute all metrics for all distance ranges.

        Args:
            contact_probs: Predicted contact probabilities
            contact_maps: True contact maps
            seq_lengths: Sequence lengths
            base_mask: Base mask from model (should exclude diagonal and padding)
            L_fractions: List of L fractions to evaluate

        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}

        # Metrics for all contacts (with base mask)
        ranges = {
            "short": (6, 12),
            "medium": (12, 24),
            "long": (24, None),
            "full": (6, None),
        }
        L_fractions = {
            "L/1": 1.0,
            "L/2": 0.5,
            "L/5": 0.2,
        }
        for range_name, (from_index, to_index) in ranges.items():
            mask = self.create_distance_range_masks(base_mask, from_index, to_index)
            all_metrics[f"roc_auc_{range_name}"] = self.masked_average_roc_auc(
                contact_logits, contact_maps, mask, average=average
            )
            if add_other_metrics:
                other_metrics = self.masked_average_other_metrics(
                    contact_logits, contact_maps, mask, average=average
                )
                for k in other_metrics.keys():
                    all_metrics[f"{k}_{range_name}"] = other_metrics[k]
            for L_fraction_name, L_fraction in L_fractions.items():
                all_metrics[f"precision_{range_name}@{L_fraction_name}"] = (
                    self.masked_average_precision_l_fraction(
                        contact_logits,
                        contact_maps,
                        mask,
                        seq_lengths,
                        L_fraction,
                        average=average,
                    )
                )
        return all_metrics
