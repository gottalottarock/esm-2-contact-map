import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB import PDBParser, PPBuilder
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from simple_parsing import ArgumentParser
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    pdb_dir: str
    split: str
    output_fasta: str
    output_parquet: str
    max_sequences: int = None


def extract_ca_info(chain):
    ca_coords = []
    b_factors = []
    residue_indices = []

    for residue in chain:
        if "CA" in residue:
            ca_atom = residue["CA"]
            ca_coords.append(ca_atom.get_coord())
            b_factors.append(ca_atom.get_bfactor())
            residue_indices.append(residue.get_id()[1])

        else:
            logger.critical(f"CA atom not found for {chain.id}")
            raise ValueError(f"CA atom not found for {chain.id}")

    return np.array(ca_coords), np.array(b_factors), residue_indices


def extract_sequences_from_pdb_dir(args: Args):
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()
    records = []
    protein_info = []
    sequence_count = 0

    for fname in tqdm(Path(args.pdb_dir).glob("*.pdb")):
        if args.max_sequences and sequence_count >= args.max_sequences:
            break
        structure = parser.get_structure(fname.stem, fname)
        for model in structure:
            if args.max_sequences and sequence_count >= args.max_sequences:
                break

            for chain in model:
                if args.max_sequences and sequence_count >= args.max_sequences:
                    break

                seq = ""
                for pp in ppb.build_peptides(chain):
                    seq += pp.get_sequence()

                if seq:
                    ca_coords, b_factors, residue_indices = extract_ca_info(chain)

                    if len(ca_coords) != len(seq):
                        logger.warning(
                            f"Length mismatch for {fname.stem}_{chain.id}: seq={len(seq)}, CA={len(ca_coords)}"
                        )
                        continue

                    protein_id = f"{args.split}_{fname.stem}_{chain.id}"

                    record = SeqRecord(
                        Seq(seq),
                        id=protein_id,
                        description=f"Length: {len(seq)}",
                    )
                    records.append(record)

                    mean_b_factor = (
                        float(np.mean(b_factors)) if len(b_factors) > 0 else np.nan
                    )
                    b_factors_list = (
                        b_factors.tolist()
                        if len(b_factors) > 0
                        else [np.nan] * len(seq)
                    )

                    protein_data = {
                        "id": protein_id,
                        "pdb_id": fname.stem,
                        "chain_id": chain.id,
                        "length": len(seq),
                        "sequence": seq,
                        "residue_indices": residue_indices,
                        "b_factors": b_factors_list,
                        "mean_b_factor": mean_b_factor,
                        "coords_ca": ca_coords.tolist()
                        if len(ca_coords) > 0
                        else [[np.nan, np.nan, np.nan]] * len(seq),
                    }
                    protein_info.append(protein_data)

                    sequence_count += 1

                    logger.debug(
                        f"Extracted: {protein_id} -> {seq[:20]}... (length: {len(seq)}, CA atoms: {len(ca_coords)})"
                    )

    logger.info(f"Total sequences extracted: {len(records)}")

    SeqIO.write(records, args.output_fasta, "fasta")
    logger.info(f"Sequences written to: {args.output_fasta}")

    df = pd.DataFrame(protein_info)
    df.to_parquet(args.output_parquet, index=False)
    logger.info(f"Protein info written to: {args.output_parquet}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()
    extract_sequences_from_pdb_dir(args.config)
