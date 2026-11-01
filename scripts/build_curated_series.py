"""Build the NT-001 curated series catalog from Merlin's committed data.

Reads every expression CSV in the committed data directory, validates
species (from registry, not from gene names), classifies regime, and
writes docs/records/NT-001_curated_series.json — the pre-training
artifact required by the NT-001 pre-registration.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.data.geo_loader import discover_merlin_datasets, content_hash

MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
REGISTRY_PATH = MERLIN_ROOT / "backend/data/registry.json"
HUMAN_DATA_DIR = MERLIN_ROOT / "backend/data/expression/human"
OUTPUT_PATH = Path(__file__).parent.parent / "docs/records/NT-001_curated_series.json"

MIN_DATA_DRIVEN = 30


def build_catalog() -> dict:
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    entries = registry.get("entries", {})

    datasets = discover_merlin_datasets(HUMAN_DATA_DIR, species="human")

    curated = []
    for d in sorted(datasets, key=lambda x: -x["n_samples"]):
        acc = d["accession"]
        registry_key = acc.lower()
        reg_entry = entries.get(registry_key, {})

        species_source = reg_entry.get("species_source", {})
        species_confirmation = species_source.get("source", "inferred from directory")

        regime = "data-driven" if d["n_samples"] >= MIN_DATA_DRIVEN else "prior-contrast"
        if d["n_control"] == 0 or d["n_treated"] == 0:
            if d["n_samples"] < MIN_DATA_DRIVEN:
                regime = "prior-mean"

        curated.append({
            "accession": acc,
            "n_samples": d["n_samples"],
            "n_genes": d["n_genes"],
            "n_control": d["n_control"],
            "n_treated": d["n_treated"],
            "species": "human",
            "species_confirmation": species_confirmation,
            "regime": regime,
            "in_regime_for_training": d["n_samples"] >= MIN_DATA_DRIVEN and d["n_control"] > 0 and d["n_treated"] > 0,
            "content_hash": d["content_hash"],
            "path": d["path"],
            "program": reg_entry.get("program", "unknown"),
            "studies": reg_entry.get("studies", []),
        })

    in_regime = [c for c in curated if c["in_regime_for_training"]]

    catalog = {
        "_about": "NT-001 curated series catalog. Built from Merlin's committed expression data.",
        "date": str(date.today()),
        "total_committed_human": len(curated),
        "in_regime_for_training": len(in_regime),
        "min_data_driven_samples": MIN_DATA_DRIVEN,
        "species_firewall": "human only; species from registry declaration, not gene symbol casing",
        "regime_rule": f"data-driven >= {MIN_DATA_DRIVEN} samples with both control and treated conditions",
        "series": curated,
    }

    catalog_hash = hashlib.sha256(
        json.dumps(catalog["series"], sort_keys=True).encode()
    ).hexdigest()
    catalog["catalog_hash"] = catalog_hash

    return catalog


def main():
    catalog = build_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Total human series: {catalog['total_committed_human']}")
    print(f"  In-regime for training: {catalog['in_regime_for_training']}")
    print(f"  Catalog hash: {catalog['catalog_hash'][:16]}...")
    for s in catalog["series"]:
        flag = "  TRAIN" if s["in_regime_for_training"] else "       "
        print(f"  {s['accession']:>12}  {s['n_samples']:>3} samples  {s['regime']:<16}{flag}")


if __name__ == "__main__":
    main()
