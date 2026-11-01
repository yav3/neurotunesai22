"""Convert downloaded GEO series_matrix.txt files to Merlin CSV format.

Reads GEO series_matrix files from /tmp/geo_dl/, validates them (human, ≥30
samples, ≥500 genes), and saves as Merlin-format CSVs (genes × samples) in
the committed data directory.

Evidence discipline: species declared from GEO Sample_organism_ch1 metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

MERLIN_DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "merlinaineuralplus2"
    / "backend"
    / "data"
    / "expression"
    / "human"
)
RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
TMP_DIR = Path("/tmp/geo_dl")

MIN_SAMPLES = 30
MIN_GENES = 500

PROGRAM_MAP = {
    "GSE26927": "neurodegeneration",
    "GSE15222": "neurodegeneration",
    "GSE5281": "neurodegeneration",
    "GSE30174": "drug_response",
}


def parse_series_matrix(filepath: Path) -> dict | None:
    """Parse a decompressed GEO series_matrix.txt."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    metadata: dict[str, str] = {}
    data_start = -1
    data_end = -1

    for i, line in enumerate(lines):
        if line.startswith("!Series_title"):
            vals = line.split("\t")[1:]
            metadata["title"] = vals[0].strip('"') if vals else ""
        elif line.startswith("!Series_platform_id"):
            vals = line.split("\t")[1:]
            metadata["platform"] = vals[0].strip('"') if vals else ""
        elif line.startswith("!Sample_organism_ch1"):
            vals = line.split("\t")[1:]
            metadata["organism"] = vals[0].strip('"').lower() if vals else ""
        elif line.startswith("!series_matrix_table_begin"):
            data_start = i + 1
        elif line.startswith("!series_matrix_table_end"):
            data_end = i
            break

    if data_start < 0 or data_end <= data_start:
        print(f"  No expression table in {filepath.name}")
        return None

    header_line = lines[data_start]
    sample_ids = [s.strip('"') for s in header_line.split("\t")[1:]]

    gene_names: list[str] = []
    matrix: list[list[float]] = []

    for line in lines[data_start + 1 : data_end]:
        if not line.strip():
            continue
        parts = line.split("\t")
        gene_id = parts[0].strip('"')
        if not gene_id:
            continue
        gene_names.append(gene_id)
        row: list[float] = []
        for v in parts[1:]:
            v = v.strip().strip('"')
            if not v or v.lower() in ("null", "nan"):
                row.append(0.0)
            else:
                try:
                    row.append(float(v))
                except ValueError:
                    row.append(0.0)
        while len(row) < len(sample_ids):
            row.append(0.0)
        matrix.append(row[: len(sample_ids)])

    return {
        "gene_names": gene_names,
        "sample_names": sample_ids,
        "matrix": matrix,
        "organism": metadata.get("organism", ""),
        "platform": metadata.get("platform", ""),
        "title": metadata.get("title", filepath.stem),
        "n_genes": len(gene_names),
        "n_samples": len(sample_ids),
    }


def main():
    print("GEO → Merlin CSV converter")

    txt_files = sorted(TMP_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in /tmp/geo_dl/")
        print("Run geo_matrix_downloader.py first, or download manually.")
        return

    print(f"Found {len(txt_files)} files to process\n")
    MERLIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for fpath in txt_files:
        acc = fpath.stem.upper()
        print(f"Processing {acc}...")

        parsed = parse_series_matrix(fpath)
        if parsed is None:
            continue

        if "homo sapiens" not in parsed["organism"]:
            print(f"  REJECTED: organism={parsed['organism']}")
            continue

        if parsed["n_samples"] < MIN_SAMPLES:
            print(f"  REJECTED: {parsed['n_samples']} samples < {MIN_SAMPLES}")
            continue

        if parsed["n_genes"] < MIN_GENES:
            print(f"  REJECTED: {parsed['n_genes']} genes < {MIN_GENES}")
            continue

        program = PROGRAM_MAP.get(acc, "general")
        safe_prog = re.sub(r"[^a-z0-9_]", "_", program)
        filename = f"{acc.lower()}_{safe_prog}_full.csv"
        out_path = MERLIN_DATA_DIR / filename

        h = hashlib.sha256()
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = ["gene"] + parsed["sample_names"]
            writer.writerow(header)
            h.update(",".join(header).encode("utf-8"))
            h.update(b"\n")
            for gene, row in zip(parsed["gene_names"], parsed["matrix"]):
                csv_row = [gene] + [str(v) for v in row]
                writer.writerow(csv_row)
                h.update(",".join(csv_row).encode("utf-8"))
                h.update(b"\n")

        sha = h.hexdigest()
        size_mb = out_path.stat().st_size / 1e6

        print(f"  SAVED: {filename}")
        print(f"    {parsed['n_genes']} genes × {parsed['n_samples']} samples ({size_mb:.1f} MB)")
        print(f"    {parsed['title'][:80]}")

        results.append({
            "accession": acc,
            "program": program,
            "n_genes": parsed["n_genes"],
            "n_samples": parsed["n_samples"],
            "organism": "human",
            "platform": parsed["platform"],
            "title": parsed["title"],
            "output_file": filename,
            "content_hash": sha,
        })

    print(f"\n{'='*60}")
    print(f"Converted: {len(results)} datasets")
    total_samples = sum(r["n_samples"] for r in results)
    total_genes_avg = sum(r["n_genes"] for r in results) / max(len(results), 1)
    print(f"Total samples: {total_samples}")
    print(f"Avg genes: {total_genes_avg:.0f}")

    if results:
        manifest = {
            "date": str(date.today()),
            "converter_version": "NT-004-data-layer",
            "species": "human",
            "datasets": results,
            "total_datasets": len(results),
            "total_samples": total_samples,
        }
        manifest["manifest_hash"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest()

        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        out = RECORDS_DIR / "geo_download_manifest.json"
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest: {out}")


if __name__ == "__main__":
    main()
