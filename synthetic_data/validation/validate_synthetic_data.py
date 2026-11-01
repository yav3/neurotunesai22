"""
Validation script for synthetic neuro/bio dataset.
Checks structural integrity, distribution plausibility, and synthetic tagging.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def validate(quarantine_dir):
    quarantine_dir = Path(quarantine_dir)
    files = sorted(quarantine_dir.glob("SYNTHETIC_neuro_bio_disease_chunk_*.parquet"))
    if not files:
        print("FAIL: No parquet files found")
        return False

    manifest_path = quarantine_dir / "SYNTHETIC_manifest.json"
    if not manifest_path.exists():
        print("FAIL: No manifest found")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {manifest['total_samples']:,} samples, {len(files)} files")

    total_rows = 0
    disease_counts = {}
    errors = []

    for fp in files:
        pf = pq.ParquetFile(str(fp))
        meta = pf.schema_arrow.metadata or {}

        if meta.get(b"synthetic") != b"true":
            errors.append(f"{fp.name}: missing synthetic=true metadata tag")

        df = pd.read_parquet(fp)
        total_rows += len(df)

        if "is_synthetic" not in df.columns or not df["is_synthetic"].all():
            errors.append(f"{fp.name}: is_synthetic column missing or has False values")

        for ds in df["disease_code"].unique():
            disease_counts[ds] = disease_counts.get(ds, 0) + (df["disease_code"] == ds).sum()

        # Check value ranges
        if df["age"].dropna().min() < 0 or df["age"].dropna().max() > 120:
            errors.append(f"{fp.name}: age out of [0, 120]")
        if not set(df["sex"].dropna().unique()).issubset({"M", "F"}):
            errors.append(f"{fp.name}: unexpected sex values")
        if df["eeg_spectral_entropy"].dropna().min() < 0 or df["eeg_spectral_entropy"].dropna().max() > 1:
            errors.append(f"{fp.name}: spectral entropy out of [0,1]")

    print(f"\nTotal rows: {total_rows:,} (expected {manifest['total_samples']:,})")
    if total_rows != manifest["total_samples"]:
        errors.append(f"Row count mismatch: {total_rows} vs {manifest['total_samples']}")

    print(f"\nDisease state distribution:")
    for code, count in sorted(disease_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_rows
        print(f"  {code:5s}: {count:>10,}  ({pct:5.1f}%)")

    # Check missingness rates
    print(f"\nMissingness rates (sample from first file):")
    df0 = pd.read_parquet(files[0])
    for col in df0.columns:
        miss = df0[col].isna().mean()
        if miss > 0:
            print(f"  {col:40s}: {miss:6.2%}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\nAll checks passed.")
        return True


if __name__ == "__main__":
    qdir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "quarantine")
    ok = validate(qdir)
    sys.exit(0 if ok else 1)
