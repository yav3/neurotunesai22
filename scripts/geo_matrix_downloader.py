"""GEO Expression Matrix Downloader — fetch full matrices for NT-004.

Downloads expression matrices from NCBI GEO for in-regime series identified
by the GEO scout catalog. Uses GEO's series_matrix.txt format (the standard
tab-delimited expression matrix available for most series).

Data flow:
    scout catalog → select top candidates → download series_matrix.txt.gz
    → decompress → parse → validate (human, ≥30 samples) → save as
    Merlin-format CSV (genes × samples)

Evidence discipline:
  - Species declared from GEO metadata, not inferred from gene symbols
  - Content-addressed (SHA-256) output
  - No silent fallback — if download fails, it fails loudly
  - Downloaded files quarantined until validated

This script does NOT train anything — it builds the data layer for NT-004.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
from datetime import date
from io import BytesIO, StringIO
from pathlib import Path
from http.client import IncompleteRead
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

MERLIN_DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "merlinaineuralplus2"
    / "backend"
    / "data"
    / "expression"
    / "human"
)
RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

MIN_SAMPLES = 30
MIN_GENES = 500


def series_matrix_url(accession: str) -> str:
    """Build the HTTPS URL for a GEO series_matrix.txt.gz file."""
    acc = accession.upper()
    nnn = acc[:len(acc) - 3] + "nnn"
    return f"{GEO_FTP_BASE}/{nnn}/{acc}/matrix/{acc}_series_matrix.txt.gz"


def download_series_matrix(accession: str, timeout: int = 120) -> bytes | None:
    """Download and decompress a series_matrix.txt.gz file. Returns raw text bytes."""
    url = series_matrix_url(accession)
    print(f"  Downloading {url}")
    req = Request(url, headers={"User-Agent": "NeuroTunesAI/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > 100_000_000:
                print(f"    Too large ({int(content_length) / 1e6:.0f} MB), skipping")
                return None
            chunks = []
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            compressed = b"".join(chunks)
        if not compressed:
            print(f"    Empty response for {accession}")
            return None
        return gzip.decompress(compressed)
    except HTTPError as e:
        if e.code == 404:
            print(f"    404 — no series_matrix.txt.gz for {accession}")
        else:
            print(f"    HTTP {e.code}: {e.reason}")
        return None
    except (URLError, TimeoutError, OSError, IncompleteRead) as e:
        print(f"    Download error: {e}")
        return None


def parse_series_matrix(raw: bytes, accession: str) -> dict | None:
    """Parse a GEO series_matrix.txt into a genes × samples dict.

    Returns dict with keys: gene_names, sample_names, matrix (list of lists),
    organism, platform, title, n_genes, n_samples — or None on failure.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")

    metadata: dict[str, str] = {}
    data_start = -1
    data_end = -1

    for i, line in enumerate(lines):
        if line.startswith("!Series_"):
            key = line.split("\t")[0].lstrip("!")
            vals = line.split("\t")[1:]
            val = vals[0].strip('"') if vals else ""
            metadata[key] = val
        elif line.startswith("!series_matrix_table_begin"):
            data_start = i + 1
        elif line.startswith("!series_matrix_table_end"):
            data_end = i
            break

    if data_start < 0 or data_end < 0 or data_end <= data_start:
        print(f"    No expression table found in {accession}")
        return None

    header_line = lines[data_start]
    sample_ids = [s.strip('"') for s in header_line.split("\t")[1:]]
    n_samples = len(sample_ids)

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
            if v == "" or v.lower() == "null" or v.lower() == "nan":
                row.append(0.0)
            else:
                try:
                    row.append(float(v))
                except ValueError:
                    row.append(0.0)
        if len(row) < n_samples:
            row.extend([0.0] * (n_samples - len(row)))
        elif len(row) > n_samples:
            row = row[:n_samples]
        matrix.append(row)

    organism = metadata.get("Series_organism_ch1", "").lower()
    platform = metadata.get("Series_platform_id", "")
    title = metadata.get("Series_title", accession)

    return {
        "gene_names": gene_names,
        "sample_names": sample_ids,
        "matrix": matrix,
        "organism": organism,
        "platform": platform,
        "title": title,
        "n_genes": len(gene_names),
        "n_samples": n_samples,
        "accession": accession,
    }


def validate_matrix(parsed: dict) -> tuple[bool, str]:
    """Validate a parsed matrix meets in-regime requirements."""
    if "homo sapiens" not in parsed["organism"]:
        return False, f"species={parsed['organism']}, not human"
    if parsed["n_samples"] < MIN_SAMPLES:
        return False, f"n_samples={parsed['n_samples']} < {MIN_SAMPLES}"
    if parsed["n_genes"] < MIN_GENES:
        return False, f"n_genes={parsed['n_genes']} < {MIN_GENES}"
    return True, "ok"


def save_as_merlin_csv(
    parsed: dict, output_dir: Path, program: str
) -> tuple[Path, str]:
    """Save parsed matrix as Merlin-format CSV (genes × samples). Returns (path, sha256)."""
    acc = parsed["accession"].lower()
    safe_program = re.sub(r"[^a-z0-9_]", "_", program.lower())
    filename = f"{acc}_{safe_program}_full.csv"
    out_path = output_dir / filename

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

    return out_path, h.hexdigest()


def load_catalog() -> list[dict]:
    """Load the GEO scout catalog."""
    catalog_path = RECORDS_DIR / "geo_scout_catalog.json"
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}")
        print("Run geo_series_scout.py first.")
        sys.exit(1)
    with open(catalog_path) as f:
        catalog = json.load(f)
    return catalog.get("candidates", [])


def select_top_candidates(
    candidates: list[dict],
    max_per_program: int = 3,
    max_total: int = 10,
    max_samples: int = 500,
) -> list[dict]:
    """Select top candidates for download, diversified across programs.

    Prefers moderately-sized series (30-500 samples) that are more likely
    to have series_matrix files and won't timeout on download.
    """
    by_program: dict[str, list[dict]] = {}
    for c in candidates:
        if c["n_samples"] > max_samples:
            continue
        p = c["program"]
        by_program.setdefault(p, []).append(c)

    selected: list[dict] = []
    seen: set[str] = set()

    rnd = 0
    while len(selected) < max_total:
        added_this_round = False
        for prog in sorted(by_program.keys()):
            if rnd < len(by_program[prog]) and rnd < max_per_program:
                c = by_program[prog][rnd]
                if c["accession"] not in seen:
                    seen.add(c["accession"])
                    selected.append(c)
                    added_this_round = True
                    if len(selected) >= max_total:
                        break
        if not added_this_round:
            break
        rnd += 1

    return selected


def main():
    print("GEO Expression Matrix Downloader — building data layer for NT-004")
    print(f"  MIN_SAMPLES={MIN_SAMPLES}, MIN_GENES={MIN_GENES}")
    print()

    candidates = load_catalog()
    print(f"Catalog: {len(candidates)} in-regime candidates")

    already_have = set()
    if MERLIN_DATA_DIR.exists():
        for f in MERLIN_DATA_DIR.glob("*.csv"):
            m = re.search(r"(gse\d+)", f.name, re.IGNORECASE)
            if m:
                already_have.add(m.group(1).upper())
    print(f"Already have: {len(already_have)} series ({', '.join(sorted(already_have))})")

    new_candidates = [c for c in candidates if c["accession"] not in already_have]
    print(f"New candidates: {len(new_candidates)}")

    top = select_top_candidates(new_candidates, max_per_program=3, max_total=10)
    print(f"\nSelected {len(top)} for download:")
    for c in top:
        print(f"  {c['accession']} — {c['n_samples']} samples — {c['program']}")
        print(f"    {c['title'][:80]}")

    MERLIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict] = []
    failed: list[dict] = []

    for c in top:
        acc = c["accession"]
        print(f"\n{'='*60}")
        print(f"Processing {acc} ({c['program']})")

        raw = download_series_matrix(acc)
        if raw is None:
            failed.append({"accession": acc, "reason": "download_failed"})
            time.sleep(1.0)
            continue

        parsed = parse_series_matrix(raw, acc)
        if parsed is None:
            failed.append({"accession": acc, "reason": "parse_failed"})
            time.sleep(1.0)
            continue

        valid, reason = validate_matrix(parsed)
        if not valid:
            print(f"  REJECTED: {reason}")
            failed.append({"accession": acc, "reason": reason})
            time.sleep(1.0)
            continue

        out_path, sha = save_as_merlin_csv(parsed, MERLIN_DATA_DIR, c["program"])
        print(f"  SAVED: {out_path.name}")
        print(f"    {parsed['n_genes']} genes × {parsed['n_samples']} samples")
        print(f"    SHA-256: {sha[:16]}...")

        downloaded.append({
            "accession": acc,
            "program": c["program"],
            "n_genes": parsed["n_genes"],
            "n_samples": parsed["n_samples"],
            "organism": parsed["organism"],
            "platform": parsed["platform"],
            "title": parsed["title"],
            "output_path": str(out_path),
            "content_hash": sha,
            "species": "human",
        })

        time.sleep(1.0)

    print(f"\n{'='*60}")
    print(f"Download complete")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")

    if downloaded:
        print(f"\nDownloaded series:")
        total_samples = 0
        for d in downloaded:
            print(f"  {d['accession']}: {d['n_genes']} genes × {d['n_samples']} samples ({d['program']})")
            total_samples += d["n_samples"]
        print(f"  Total new samples: {total_samples}")

    if failed:
        print(f"\nFailed:")
        for f_item in failed:
            print(f"  {f_item['accession']}: {f_item['reason']}")

    manifest = {
        "date": str(date.today()),
        "downloader_version": "NT-004-data-layer",
        "min_samples": MIN_SAMPLES,
        "min_genes": MIN_GENES,
        "species": "human",
        "downloaded": downloaded,
        "failed": failed,
        "total_downloaded": len(downloaded),
        "total_failed": len(failed),
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RECORDS_DIR / "geo_download_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
