# SYNTHETIC DATA — DO NOT USE FOR CLINICAL DECISIONS

**Classification**: SYNTHETIC / SIMULATED  
**Generated**: 2026-10-31  
**Generator version**: 1.0.0  

## Warning

All data in this directory tree (`synthetic_data/`) is **entirely synthetic**.
It was generated algorithmically to approximate the statistical structure of
real neurological and biological datasets. **No real patient data** was used
in, or is contained within, any file under this directory.

## Intended Use

- Algorithm development and benchmarking
- Pipeline testing and validation
- Educational demonstrations
- Software integration testing

## Prohibited Use

- Clinical decision-making
- Regulatory submissions without explicit disclosure
- Representation as real patient data
- Training clinical diagnostic models without disclosure

## Data Provenance

Generated via `synthetic_data/generators/generate_neuro_bio_dataset.py`
using parameterized statistical distributions derived from published
literature ranges. Disease-state parameters are documented in
`synthetic_data/schemas/disease_parameters.json`.

## Quarantine Policy

This directory is the **sole** location for synthetic data in this repository.
Synthetic data must never be moved to or mixed with real data directories.
Every output file contains a `synthetic=true` metadata flag and a
`SYNTHETIC_` filename prefix.
