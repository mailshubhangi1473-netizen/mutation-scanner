"""Core logic: physicochemical scoring of every possible amino-acid substitution."""
import io

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import substitution_matrices
from Bio.SeqUtils.ProtParamData import kd  # Kyte-Doolittle hydropathy

AA = "ACDEFGHIKLMNPQRSTVWY"

# Residue volumes in cubic angstroms (Zamyatnin, 1972)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

# Approximate side-chain charge at pH 7.4
CHARGE = {aa: 0.0 for aa in AA}
CHARGE.update({"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0})

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Maximum possible differences, used to scale each component to 0-1
_MAX_HYDRO = max(kd[a] for a in AA) - min(kd[a] for a in AA)  # 9.0
_MAX_CHARGE = 2.0
_MAX_VOLUME = max(VOLUME.values()) - min(VOLUME.values())
_BLOSUM_MIN, _BLOSUM_MAX = -4, 3  # off-diagonal range for the 20 standard AAs

DEFAULT_WEIGHTS = {"hydrophobicity": 0.25, "charge": 0.20, "volume": 0.25, "blosum62": 0.30}


def parse_sequence(text: str):
    """Accept FASTA or a raw sequence. Returns (name, sequence). Raises ValueError."""
    text = text.strip()
    if not text:
        raise ValueError("No sequence provided.")
    if text.startswith(">"):
        record = next(SeqIO.parse(io.StringIO(text), "fasta"), None)
        if record is None:
            raise ValueError("Could not parse FASTA input.")
        name, seq = record.id, str(record.seq)
    else:
        name, seq = "pasted_sequence", "".join(text.split())
    seq = seq.upper().replace("*", "")
    bad = sorted(set(seq) - set(AA))
    if bad:
        raise ValueError(
            f"Unsupported characters: {', '.join(bad)}. Use the 20 standard amino-acid letters."
        )
    if len(seq) < 2:
        raise ValueError("Sequence is too short.")
    return name, seq


def substitution_components(wt: str, mut: str) -> dict:
    """Each component is scaled 0 (no change) to 1 (most disruptive possible)."""
    return {
        "hydrophobicity": abs(kd[wt] - kd[mut]) / _MAX_HYDRO,
        "charge": abs(CHARGE[wt] - CHARGE[mut]) / _MAX_CHARGE,
        "volume": abs(VOLUME[wt] - VOLUME[mut]) / _MAX_VOLUME,
        "blosum62": (_BLOSUM_MAX - BLOSUM62[wt][mut]) / (_BLOSUM_MAX - _BLOSUM_MIN),
    }


def disruption_score(wt: str, mut: str, weights: dict) -> float:
    comps = substitution_components(wt, mut)
    total = sum(weights.values()) or 1.0
    return sum(weights[k] * comps[k] for k in comps) / total


def scan(seq: str, weights: dict = None):
    """Score all 19 substitutions at every position.

    Returns (matrix, positions):
      matrix: DataFrame, rows = mutant AA, columns = position (1-based), NaN on wild type
      positions: DataFrame with per-position mean/max disruption
    """
    weights = weights or DEFAULT_WEIGHTS
    data = np.full((len(AA), len(seq)), np.nan)
    for j, wt in enumerate(seq):
        for i, mut in enumerate(AA):
            if mut != wt:
                data[i, j] = disruption_score(wt, mut, weights)
    matrix = pd.DataFrame(data, index=list(AA), columns=range(1, len(seq) + 1))
    positions = pd.DataFrame(
        {
            "position": range(1, len(seq) + 1),
            "residue": list(seq),
            "mean_disruption": np.nanmean(data, axis=0),
            "max_disruption": np.nanmax(data, axis=0),
        }
    )
    return matrix, positions


def mutation_table(seq: str, weights: dict = None) -> pd.DataFrame:
    """Long-format table of every substitution (for download)."""
    matrix, _ = scan(seq, weights)
    rows = []
    for pos in matrix.columns:
        wt = seq[pos - 1]
        for mut in AA:
            if mut != wt:
                rows.append(
                    {"mutation": f"{wt}{pos}{mut}", "position": pos, "wt": wt,
                     "mut": mut, "score": matrix.loc[mut, pos]}
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
