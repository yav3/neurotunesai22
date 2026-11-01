"""GEO Series Scout — discover in-regime datasets for NT-004.

Queries NCBI GEO via E-utilities to find expression series matching
our training requirements:
  - Species: human (firewall-enforced)
  - Samples: ≥30 (data-driven regime)
  - Type: Expression profiling by high throughput sequencing or array
  - Has supplementary expression matrix files

This script does NOT download data — it produces a prioritized catalog
of candidate series for manual review before download.

Evidence discipline: species declared from GEO metadata (organism field),
not inferred from gene symbols.
"""
from __future__ import annotations

import json
import sys
import time
import hashlib
from datetime import date
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

RECORDS_DIR = Path(__file__).parent.parent / "docs" / "records"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PROGRAMS = {
    "autophagy": '("autophagy"[MeSH Terms] OR autophagy[Title/Abstract])',
    "drug_response": '("drug response"[Title/Abstract] OR "perturbation"[Title/Abstract] OR "treatment response"[Title/Abstract])',
    "reprogramming": '("cellular reprogramming"[MeSH Terms] OR "iPSC"[Title/Abstract] OR "reprogramming"[Title/Abstract])',
    "neurodegeneration": '("neurodegeneration"[MeSH Terms] OR "Alzheimer"[Title/Abstract] OR "Parkinson"[Title/Abstract])',
    "immune_response": '("immune response"[MeSH Terms] OR "T cell"[Title/Abstract] OR "immunotherapy"[Title/Abstract])',
}

MIN_SAMPLES = 30


def esearch(query: str, retmax: int = 200) -> list[str]:
    """Search GEO DataSets and return UIDs."""
    params = urlencode({
        "db": "gds",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "n",
    })
    url = f"{EUTILS_BASE}/esearch.fcgi?{params}"
    req = Request(url, headers={"User-Agent": "NeuroTunesAI/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"  esearch error: {e}")
        return []


def esummary(uids: list[str]) -> list[dict]:
    """Fetch summaries for GDS UIDs."""
    if not uids:
        return []
    params = urlencode({
        "db": "gds",
        "id": ",".join(uids[:50]),
        "retmode": "json",
    })
    url = f"{EUTILS_BASE}/esummary.fcgi?{params}"
    req = Request(url, headers={"User-Agent": "NeuroTunesAI/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        result = data.get("result", {})
        summaries = []
        for uid in result.get("uids", []):
            entry = result.get(uid, {})
            if entry:
                summaries.append(entry)
        return summaries
    except Exception as e:
        print(f"  esummary error: {e}")
        return []


def parse_series_info(summary: dict) -> dict | None:
    """Extract relevant fields from a GDS/GSE summary."""
    accession = summary.get("accession", "")
    if not accession.startswith("GSE"):
        gse = summary.get("gse", "")
        if gse:
            accession = f"GSE{gse}"
        else:
            return None

    n_samples = int(summary.get("n_samples", 0))
    organism = summary.get("taxon", "")
    title = summary.get("title", "")
    summary_text = summary.get("summary", "")
    pdat = summary.get("pdat", "")
    gpl = summary.get("gpl", "")
    entrytype = summary.get("entrytype", "")

    if "homo sapiens" not in organism.lower():
        return None

    return {
        "accession": accession,
        "title": title,
        "n_samples": n_samples,
        "organism": organism,
        "platform": gpl,
        "date": pdat,
        "entry_type": entrytype,
        "summary_excerpt": summary_text[:200] if summary_text else "",
    }


def scout_program(program: str, query_fragment: str) -> list[dict]:
    """Scout GEO for in-regime series matching a biological program."""
    base_query = (
        f'{query_fragment} AND "Homo sapiens"[Organism] '
        f'AND ("Expression profiling by high throughput sequencing"[Filter] '
        f'OR "Expression profiling by array"[Filter])'
    )

    print(f"\n  Searching: {program}")
    uids = esearch(base_query, retmax=200)
    print(f"    Found {len(uids)} GDS entries")

    if not uids:
        return []

    time.sleep(0.4)

    summaries = esummary(uids)
    print(f"    Retrieved {len(summaries)} summaries")

    candidates = []
    for s in summaries:
        info = parse_series_info(s)
        if info and info["n_samples"] >= MIN_SAMPLES:
            info["program"] = program
            candidates.append(info)

    candidates.sort(key=lambda x: -x["n_samples"])
    print(f"    In-regime candidates (≥{MIN_SAMPLES} samples): {len(candidates)}")
    return candidates


def main():
    print("GEO Series Scout — discovering in-regime datasets for NT-004")
    print(f"  MIN_SAMPLES={MIN_SAMPLES}, species=human")
    print()

    all_candidates = []
    seen_accessions = set()

    for program, query in PROGRAMS.items():
        candidates = scout_program(program, query)
        for c in candidates:
            if c["accession"] not in seen_accessions:
                seen_accessions.add(c["accession"])
                all_candidates.append(c)
        time.sleep(1.0)

    all_candidates.sort(key=lambda x: -x["n_samples"])

    print(f"\n{'='*60}")
    print(f"Total unique in-regime candidates: {len(all_candidates)}")
    print(f"\nTop 20 by sample count:")
    for i, c in enumerate(all_candidates[:20]):
        print(f"  {i+1:2d}. {c['accession']} — {c['n_samples']} samples — {c['program']}")
        print(f"      {c['title'][:80]}")

    catalog = {
        "date": str(date.today()),
        "scout_version": "NT-003-follow-up",
        "min_samples": MIN_SAMPLES,
        "species": "human",
        "programs_queried": list(PROGRAMS.keys()),
        "total_candidates": len(all_candidates),
        "candidates": all_candidates,
    }
    catalog["catalog_hash"] = hashlib.sha256(
        json.dumps(catalog, sort_keys=True).encode()
    ).hexdigest()

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECORDS_DIR / "geo_scout_catalog.json"
    with open(out, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"\nCatalog written: {out}")

    by_program = {}
    for c in all_candidates:
        p = c["program"]
        by_program.setdefault(p, []).append(c)

    print(f"\nBreakdown by program:")
    for p, cs in sorted(by_program.items(), key=lambda x: -len(x[1])):
        print(f"  {p}: {len(cs)} series, {sum(c['n_samples'] for c in cs)} total samples")


if __name__ == "__main__":
    main()
