"""Evaluation metrics and benchmarking for NeuroFoundation downstream tasks."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)

from models.config import DISEASE_CODES


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float | str]:
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    for batch in loader:
        x_dict = {k: v.to(device) for k, v in batch["x_dict"].items()}
        mask_dict = {k: v.to(device) for k, v in batch.get("mask_dict", {}).items()}
        labels = batch["labels"]

        logits = model(x_dict, mask_dict)
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": classification_report(y_true, y_pred, target_names=DISEASE_CODES),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


@torch.no_grad()
def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    model.eval()
    model.to(device)
    all_preds, all_targets = [], []

    for batch in loader:
        x_dict = {k: v.to(device) for k, v in batch["x_dict"].items()}
        mask_dict = {k: v.to(device) for k, v in batch.get("mask_dict", {}).items()}
        time_months = batch.get("time_months", torch.zeros(len(batch["labels"]))).to(device)
        targets = batch.get("targets", batch["labels"]).float()

        preds = model(x_dict, time_months, mask_dict).cpu()
        all_preds.append(preds)
        all_targets.append(targets)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    valid = ~np.isnan(y_true)
    if valid.ndim > 1:
        valid = valid.all(axis=1)

    return {
        "mse": mean_squared_error(y_true[valid], y_pred[valid]),
        "rmse": mean_squared_error(y_true[valid], y_pred[valid], squared=False),
        "r2": r2_score(y_true[valid].ravel(), y_pred[valid].ravel()),
    }


@torch.no_grad()
def evaluate_imputation(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate imputation by artificially masking modalities and measuring
    reconstruction error on the held-out values."""
    model.eval()
    model.to(device)
    per_modality_mse: dict[str, list[float]] = {}

    for batch in loader:
        x_dict = {k: v.to(device) for k, v in batch["x_dict"].items()}
        mask_dict = {k: v.to(device) for k, v in batch.get("mask_dict", {}).items()}

        imputed = model(x_dict, mask_dict)
        if isinstance(imputed, dict) and "imputed" in imputed:
            imputed = imputed["imputed"]

        for name in imputed:
            if name in x_dict:
                mse = ((imputed[name] - x_dict[name]) ** 2).mean().item()
                per_modality_mse.setdefault(name, []).append(mse)

    avg = {name: np.mean(vals) for name, vals in per_modality_mse.items()}
    avg["mean_mse"] = np.mean(list(avg.values()))
    return avg
