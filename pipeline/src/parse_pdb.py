import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB import PDBParser, is_aa
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq1
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
    max_sequences: Optional[int] = None


def extract_chain_data(chain):
    sequences = []
    ca_coords = []
    b_factors = []
    residue_indices = []

    for residue in chain:
        if is_aa(residue, standard=True):
            if "CA" in residue:
                three_letter = residue.get_resname()
                one_letter = seq1(three_letter)
                sequences.append(one_letter)

                ca_atom = residue["CA"]
                ca_coords.append(ca_atom.get_coord())
                b_factors.append(ca_atom.get_bfactor())
                residue_indices.append(residue.get_id()[1])
            else:
                raise ValueError(
                    f"Standard AA {residue.get_resname()} without CA atom in chain {chain.id}, residue {residue.get_id()[1]}"
                )

    seq_len = len(sequences)
    ca_len = len(ca_coords)
    bf_len = len(b_factors)
    ri_len = len(residue_indices)

    if not (seq_len == ca_len == bf_len == ri_len):
        raise ValueError(
            f"Inconsistent data lengths: seq={seq_len}, CA={ca_len}, b_factors={bf_len}, residue_indices={ri_len}"
        )

    if seq_len == 0:
        logger.warning(f"No sequences found in chain {chain.id}")
        return None

    return {
        "sequence": "".join(sequences),
        "ca_coords": np.stack(ca_coords).reshape(-1),
        "b_factors": np.stack(b_factors),
        "residue_indices": residue_indices,
        "length": seq_len,
    }


def process_structure(structure, args, sequence_count):
    records = []
    protein_info = []

    model = structure[0]
    for chain in model:
        chain_data = extract_chain_data(chain)
        if chain_data is None:
            continue

        protein_id = f"{args.split}_{structure.id}_{chain.id}"

        record = SeqRecord(
            Seq(chain_data["sequence"]),
            id=protein_id,
            description=f"Length: {chain_data['length']}",
        )
        records.append(record)

        mean_b_factor = float(np.mean(chain_data["b_factors"]))

        protein_data = {
            "id": protein_id,
            "pdb_id": structure.id,
            "chain_id": chain.id,
            "length": chain_data["length"],
            "sequence": chain_data["sequence"],
            "residue_indices": chain_data["residue_indices"],
            "b_factors": chain_data["b_factors"].tolist(),
            "mean_b_factor": mean_b_factor,
            "coords_ca": chain_data["ca_coords"].tolist(),
        }
        protein_info.append(protein_data)

        sequence_count += 1

        logger.debug(
            f"Processed: {protein_id} -> {chain_data['sequence'][:20]}... "
            f"(length: {chain_data['length']}, mean_b_factor: {mean_b_factor:.2f})"
        )

    return records, protein_info, sequence_count


def extract_sequences_from_pdb_dir(args: Args):
    parser = PDBParser(QUIET=True)
    all_records = []
    all_protein_info = []
    sequence_count = 0

    for fname in tqdm(list(Path(args.pdb_dir).glob("*.pdb"))):
        try:
            if args.max_sequences and sequence_count >= args.max_sequences:
                break

            structure = parser.get_structure(fname.stem, fname)
            records, protein_info, sequence_count = process_structure(
                structure, args, sequence_count
            )

            all_records.extend(records)
            all_protein_info.extend(protein_info)

        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            continue

    logger.info(f"Total sequences extracted: {len(all_records)}")

    if len(all_records) == 0:
        logger.warning("No sequences extracted!")
        return

    SeqIO.write(all_records, args.output_fasta, "fasta")
    logger.info(f"Sequences written to: {args.output_fasta}")

    df = pd.DataFrame(all_protein_info)
    df.to_parquet(args.output_parquet, index=False)
    logger.info(f"Protein info written to: {args.output_parquet}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()
    extract_sequences_from_pdb_dir(args.config)
