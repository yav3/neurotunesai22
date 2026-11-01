"""PyTorch datasets for cross-sectional and longitudinal synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from models.config import MODALITY_DIMS, DISEASE_CODES


# Column name mappings from parquet → modality groups
_COL_MAP: dict[str, list[str]] = {
    "eeg_spectral": [
        "eeg_delta_1_4hz", "eeg_theta_4_8hz", "eeg_alpha_8_13hz",
        "eeg_beta_13_30hz", "eeg_gamma_30_100hz",
    ],
    "eeg_derived": [
        "eeg_theta_alpha_ratio", "eeg_spectral_entropy", "eeg_alpha_peak_freq_hz",
    ],
    "fmri_bold": [
        "fmri_bold_dmn", "fmri_bold_pfc", "fmri_bold_hippocampus",
        "fmri_bold_amygdala", "fmri_bold_motor_cortex", "fmri_bold_cerebellum",
        "fmri_bold_thalamus", "fmri_bold_basal_ganglia",
    ],
    "fmri_connectivity": [
        "fmri_conn_dmn_pfcx", "fmri_conn_hipp_amyg",
        "fmri_conn_motor_bg", "fmri_conn_thal_cortical",
    ],
    "blood": [
        "blood_nfl_pg_ml", "blood_gfap_pg_ml", "blood_bdnf_ng_ml",
        "blood_s100b_ug_l", "blood_crp_mg_l", "blood_il6_pg_ml",
        "blood_tnf_alpha_pg_ml",
    ],
    "csf": [
        "csf_amyloid_beta42_pg_ml", "csf_total_tau_pg_ml",
        "csf_p_tau181_pg_ml", "csf_nfl_pg_ml",
    ],
    "genomic": ["apoe_e4_carrier", "polygenic_risk_score"],
    "clinical": [
        "mmse_score", "moca_score", "hamilton_depression_score",
        "updrs_motor_score",
    ],
    "demographics": ["age", "sex_female", "bmi"],
}

_LABEL_COL = "disease_code"


class NeuroDataset(Dataset):
    """Cross-sectional dataset from parquet files.

    Each sample is a dict of modality tensors plus a disease label.
    Handles NaN as missing data: modality mask is False where all features
    in a group are NaN.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_samples: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        dfs = [pq.read_table(f).to_pandas() for f in files]
        self.df = pd.concat(dfs, ignore_index=True)

        if max_samples is not None:
            self.df = self.df.head(max_samples)

        # Encode labels
        self.label_map = {code: i for i, code in enumerate(DISEASE_CODES)}

        # Precompute per-feature statistics for normalization
        self._stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mod, cols in _COL_MAP.items():
            present = [c for c in cols if c in self.df.columns]
            if present:
                vals = self.df[present].values.astype(np.float32)
                self._stats[mod] = (
                    np.nanmean(vals, axis=0),
                    np.nanstd(vals, axis=0) + 1e-8,
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sample: dict[str, torch.Tensor] = {}
        mask: dict[str, torch.Tensor] = {}

        for mod, cols in _COL_MAP.items():
            present = [c for c in cols if c in self.df.columns]
            if not present:
                continue
            vals = np.array([row.get(c, np.nan) for c in present], dtype=np.float32)
            is_valid = ~np.isnan(vals)
            if is_valid.any() and mod in self._stats:
                mu, sigma = self._stats[mod]
                vals = np.where(is_valid, (vals - mu[:len(vals)]) / sigma[:len(vals)], 0.0)
            else:
                vals = np.zeros_like(vals)
            sample[mod] = torch.from_numpy(vals)
            mask[mod] = torch.tensor(is_valid.all(), dtype=torch.bool)

        label_str = row.get(_LABEL_COL, "HC")
        sample["label"] = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        sample["_mask"] = mask
        return sample


class LongitudinalDataset(Dataset):
    """Longitudinal dataset for progression / treatment response modeling.

    Each sample is a sequence of visits for one patient, returned as a dict
    of (T, d_m) tensors per modality plus visit-level metadata.
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_visits: int = 20,
        max_patients: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        files = sorted(self.data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
        self.max_visits = max_visits

        # Group by patient
        pid_col = "patient_id" if "patient_id" in df.columns else df.columns[0]
        self.patients = []
        for pid, group in df.groupby(pid_col):
            group = group.sort_values("visit_month" if "visit_month" in group.columns
                                      else group.columns[1])
            if len(group) > max_visits:
                group = group.head(max_visits)
            self.patients.append(group)
            if max_patients and len(self.patients) >= max_patients:
                break

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        group = self.patients[idx]
        T = len(group)
        sample: dict[str, torch.Tensor] = {}

        for mod, cols in _COL_MAP.items():
            present = [c for c in cols if c in group.columns]
            if not present:
                continue
            vals = group[present].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            sample[mod] = torch.from_numpy(vals)

        if "visit_month" in group.columns:
            sample["time"] = torch.from_numpy(
                group["visit_month"].values.astype(np.float32)
            )

        sample["seq_len"] = torch.tensor(T, dtype=torch.long)
        return sample
