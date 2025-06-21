import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from simple_parsing import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    input_parquet: str
    input_fasta: str
    output_parquet: str
    output_fasta: str
    min_length: int = 30
    max_length: int = 1000
    max_x_fraction: float = 0.1
    max_b_factor: float = 100.0


def analyze_sequence_quality(df):
    """Analyze sequence quality and report statistics."""
    logger.info("Quality Analysis:")
    logger.info("=" * 50)

    too_short = df[df["length"] < 30]
    too_long = df[df["length"] > 1000]

    invalid_chars = df[
        df["sequence"].str.contains(r"[^ACDEFGHIKLMNPQRSTVWY]", na=False)
    ]

    x_rich = df[df["sequence"].str.count("X") / df["length"] > 0.1]

    high_b_factor = df[df["mean_b_factor"] > 100]
    missing_b_factor = df[df["mean_b_factor"].isna()]

    logger.info(
        f"Sequences too short (<30 aa): {len(too_short)} ({len(too_short) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"Sequences too long (>1000 aa): {len(too_long)} ({len(too_long) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"Sequences with invalid characters: {len(invalid_chars)} ({len(invalid_chars) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"Sequences with >10% X residues: {len(x_rich)} ({len(x_rich) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"High B-factor structures (>100): {len(high_b_factor)} ({len(high_b_factor) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"Missing B-factor data: {len(missing_b_factor)} ({len(missing_b_factor) / len(df) * 100:.2f}%)"
    )

    if len(invalid_chars) > 0:
        logger.warning("Invalid characters found:")
        for idx in invalid_chars.index[:5]:
            seq = invalid_chars.loc[idx, "sequence"]
            invalid = "".join(
                set(char for char in seq if char not in "ACDEFGHIKLMNPQRSTVWY")
            )
            logger.warning(f"  Index {idx}: {invalid}")

    if len(x_rich) > 0:
        logger.info("Examples of X-rich sequences:")
        for idx in x_rich.index[:3]:
            seq = x_rich.loc[idx, "sequence"]
            x_count = seq.count("X")
            logger.info(
                f"  Index {idx}: {x_count}/{len(seq)} X's ({x_count / len(seq) * 100:.1f}%)"
            )


def apply_quality_filters(df, args):
    """Apply quality filters to the dataset."""
    logger.info("Applying Quality Filters:")
    logger.info("=" * 50)

    filters = {
        "length": (df["length"] >= args.min_length) & (df["length"] <= args.max_length),
        "valid_chars": ~df["sequence"].str.contains(
            r"[^ACDEFGHIKLMNPQRSTVWY]", na=False
        ),
        "x_content": df["sequence"].str.count("X") / df["length"]
        <= args.max_x_fraction,
        "b_factor": (df["mean_b_factor"] <= args.max_b_factor)
        | df["mean_b_factor"].isna(),
    }

    combined_filter = (
        filters["length"]
        & filters["valid_chars"]
        & filters["x_content"]
        & filters["b_factor"]
    )

    filtered_df = df[combined_filter].copy()

    logger.info(f"Original sequences: {len(df)}")
    for filter_name, filter_mask in filters.items():
        passed = filter_mask.sum()
        logger.info(
            f"  {filter_name}: {passed} sequences passed ({passed / len(df) * 100:.1f}%)"
        )

    logger.info(
        f"Final filtered sequences: {len(filtered_df)} ({len(filtered_df) / len(df) * 100:.2f}%)"
    )
    logger.info(
        f"Removed: {len(df) - len(filtered_df)} sequences ({(len(df) - len(filtered_df)) / len(df) * 100:.2f}%)"
    )

    return filtered_df


def filter_fasta(input_fasta_path, output_fasta_path, valid_ids):
    """Filter FASTA file to keep only sequences with valid IDs."""
    records = []
    valid_ids_set = set(valid_ids)

    for record in SeqIO.parse(input_fasta_path, "fasta"):
        if record.id in valid_ids_set:
            records.append(record)

    SeqIO.write(records, output_fasta_path, "fasta")
    logger.info(f"Filtered FASTA written to: {output_fasta_path}")
    logger.info(f"FASTA sequences: {len(records)}")


def filter_sequences(args: Args):
    """Main function to filter sequences based on quality criteria."""
    logger.info(f"Loading data from: {args.input_parquet}")

    # Load the data
    df = pd.read_parquet(args.input_parquet)
    # sort df for reproducibility
    df = df.sort_values(by="id")
    assert df["id"].is_unique, "IDs are not unique"
    logger.info(f"Loaded {len(df)} sequences")

    # Analyze quality before filtering
    analyze_sequence_quality(df)

    # Apply filters
    filtered_df = apply_quality_filters(df, args)

    # Save filtered parquet
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(args.output_parquet, index=False)
    logger.info(f"Filtered parquet written to: {args.output_parquet}")

    # Filter and save FASTA file
    logger.info(f"Filtering FASTA file: {args.input_fasta}")
    Path(args.output_fasta).parent.mkdir(parents=True, exist_ok=True)
    filter_fasta(args.input_fasta, args.output_fasta, filtered_df["id"].tolist())

    logger.info("Filtering completed successfully!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()
    filter_sequences(args.config)
