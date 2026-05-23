"""
parse_jpl_mnf_stats.py

Parses a JPL MNF / ENVI statistics text file into structured Python objects.

Usage:
    from parse_jpl_mnf_stats import parse_jpl_mnf_stats
    basic_stats, covariance, correlation, eigenvectors, eigenvalues = \
        parse_jpl_mnf_stats("full_jpl_mnf_txt_stats.txt")

Returns
-------
basic_stats  : dict
    Keys are band names (e.g. 'Band 1').
    Values are dicts with keys 'Min', 'Max', 'Mean', 'StdDev' (floats).

covariance   : np.ndarray, shape (n_bands, n_bands)
correlation  : np.ndarray, shape (n_bands, n_bands)
    NaN where ENVI wrote '-NaN'.

eigenvectors : np.ndarray, shape (n_bands, n_bands)
    Row i  is eigenvector i  (i.e. eigenvectors[0] == first eigenvector).

eigenvalues  : np.ndarray, shape (n_bands,)
"""

import re
import numpy as np


# ---------------------------------------------------------------------------
# Row label patterns
# ---------------------------------------------------------------------------
# "Band 1", "Band 12", …
_BAND_ROW = re.compile(r"^\s*(Band\s+\d+)\s+(.*)", re.DOTALL)
# "Eig. 1", "Eig. 12", …
_EIG_ROW = re.compile(r"^\s*(Eig\.\s+\d+)\s+(.*)", re.DOTALL)


def _parse_data_row(line):
    """
    Parse one data row.  Returns (label: str, values: list[float]) or None.
    Handles '-NaN' / 'NaN' as np.nan.
    """
    m = _BAND_ROW.match(line) or _EIG_ROW.match(line)
    if not m:
        return None
    label, rest = m.group(1), m.group(2)
    vals = []
    for token in rest.split():
        if re.match(r"^-?nan$", token, re.IGNORECASE):
            vals.append(np.nan)
        else:
            try:
                vals.append(float(token))
            except ValueError:
                pass  # skip stray non-numeric tokens
    return label, vals


def _extract_rows(lines, section_start):
    """
    Read data rows that immediately follow the section header line.
    Stops on blank line or a line that starts with an alphabetic word
    that is not 'Band' or 'Eig.' (i.e. a new section header).
    """
    rows = []
    for line in lines[section_start + 1 :]:
        stripped = line.strip()
        # blank line → end of section
        if not stripped:
            break
        # new section header?
        first = stripped.split()[0]
        if first[0].isalpha() and first not in ("Band", "Eig."):
            break
        parsed = _parse_data_row(line)
        if parsed is not None:
            rows.append(parsed)
    return rows  # list of (label, [float, ...])


def parse_jpl_mnf_stats(filepath: str):
    with open(filepath, "r", errors="replace") as fh:
        lines = [ln.rstrip() for ln in fh.readlines()]

    # ------------------------------------------------------------------ #
    # Find section header lines
    # ------------------------------------------------------------------ #
    sec = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if not sec.get("basic_stats") and s.startswith("Basic Stats") and "Min" in s:
            sec["basic_stats"] = i
        elif not sec.get("covariance") and s.startswith("Covariance"):
            sec["covariance"] = i
        elif not sec.get("correlation") and s.startswith("Correlation"):
            sec["correlation"] = i
        elif not sec.get("eigenvectors") and s.startswith("Eigenvectors"):
            sec["eigenvectors"] = i
        elif not sec.get("eigenvalues") and s.startswith("Eigenvalues"):
            sec["eigenvalues"] = i

    # ------------------------------------------------------------------ #
    # Basic Stats
    # ------------------------------------------------------------------ #
    basic_stats = {}
    for label, vals in _extract_rows(lines, sec["basic_stats"]):
        if len(vals) >= 4:
            basic_stats[label] = {
                "Min": vals[0],
                "Max": vals[1],
                "Mean": vals[2],
                "StdDev": vals[3],
            }

    n = len(basic_stats)

    # ------------------------------------------------------------------ #
    # Helper: build (n x n) matrix from row list
    # ------------------------------------------------------------------ #
    def to_matrix(rows):
        mat = np.full((n, n), np.nan)
        for i, (_, vals) in enumerate(rows):
            cols = min(len(vals), n)
            mat[i, :cols] = vals[:cols]
        return mat

    # ------------------------------------------------------------------ #
    # Covariance / Correlation / Eigenvectors
    # ------------------------------------------------------------------ #
    covariance = to_matrix(_extract_rows(lines, sec["covariance"]))
    correlation = to_matrix(_extract_rows(lines, sec["correlation"]))
    eigenvectors = to_matrix(_extract_rows(lines, sec["eigenvectors"]))

    # ------------------------------------------------------------------ #
    # Eigenvalues
    # ------------------------------------------------------------------ #
    eigenvalues = np.array(
        [vals[0] for _, vals in _extract_rows(lines, sec["eigenvalues"]) if vals],
        dtype=float,
    )

    return basic_stats, covariance, correlation, eigenvectors, eigenvalues


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "full_jpl_mnf_txt_stats.txt"
    bs, cov, cor, evec, eval_ = parse_jpl_mnf_stats(path)

    print(f"Bands parsed : {len(bs)}")

    print("\n--- Basic Stats (first 3 bands) ---")
    for k in list(bs)[:3]:
        print(f"  {k}: {bs[k]}")

    print(f"\n--- Covariance   shape={cov.shape}  (top-left 3x3) ---")
    print(cov[:3, :3])

    print(f"\n--- Correlation  shape={cor.shape}  (top-left 3x3) ---")
    print(cor[:3, :3])

    print(f"\n--- Eigenvectors shape={evec.shape}  (rows 0-2, cols 0-2) ---")
    print(evec[:3, :3])

    print(f"\n--- Eigenvalues  shape={eval_.shape}  (first 5) ---")
    print(eval_[:5])
