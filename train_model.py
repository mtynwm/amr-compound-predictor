"""Offline training script: ChEMBL bioactivity data -> trained classifier.

Run manually (not from the Streamlit app, training is a heavy one-off step):
    python train_model.py --target saureus_mic

Produces in models/:
    model_<id>.joblib          - sklearn Pipeline (StandardScaler + RandomForestClassifier)
    metadata_<id>.json         - dataset stats, evaluation metrics, threshold, feature names
    pca_<id>.joblib            - fitted PCA(2) for the SAR Insight applicability-domain plot
    pca_training_points_<id>.csv - training set projected into PCA space, with labels
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_TARGET_ID, get_target
from featurize import DESCRIPTOR_NAMES, FEATURE_NAMES, featurize, smiles_to_mol

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"


CHEMBL_API_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"
PAGE_SIZE = 1000
MAX_RETRIES = 6
RETRY_BACKOFF_SECONDS = 10
# Server ChEMBL cukup sering flaky (timeout/500 intermiten). Data QSAR beberapa ribu
# senyawa sudah lebih dari cukup untuk model yang solid, jadi dibatasi (bukan tarik
# semua ~160rb record) supaya proses fetch lebih cepat dan lebih tahan gangguan.
MAX_RECORDS_TO_FETCH = 30000


def _fetch_page(target_chembl_id, standard_type, offset):
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": standard_type,
        "format": "json",
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(CHEMBL_API_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            print(f"  Gagal ambil offset={offset} (percobaan {attempt + 1}), retry dalam {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Gagal mengambil data ChEMBL di offset={offset} setelah {MAX_RETRIES} retry.")


def fetch_raw_activities(target_cfg):
    """Fetch activities, caching each page to disk so a rerun after a crash
    resumes from the last successful offset instead of starting over."""
    cache_path = DATA_DIR / f"raw_activities_cache_{target_cfg['id']}.jsonl"
    DATA_DIR.mkdir(exist_ok=True)

    records = []
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        print(f"  Cache lokal ditemukan: {len(records)} record.")

    if len(records) >= MAX_RECORDS_TO_FETCH:
        print("  Cache sudah lengkap - melatih ulang offline tanpa menghubungi ChEMBL.")
        records = [r for r in records if r.get("standard_units") == target_cfg["standard_units"]]
        print(f"  {len(records)} record dengan unit {target_cfg['standard_units']}.")
        return records

    print(f"Mengambil data dari ChEMBL untuk target {target_cfg['chembl_target_id']}...")
    first_page = _fetch_page(target_cfg["chembl_target_id"], target_cfg["standard_type"], offset=0)
    total_count = min(first_page["page_meta"]["total_count"], MAX_RECORDS_TO_FETCH)
    print(f"  Total record tersedia: {first_page['page_meta']['total_count']} (dibatasi ambil {total_count})")

    offset = len(records)
    with open(cache_path, "a", encoding="utf-8") as cache_file:
        while offset < total_count:
            page = _fetch_page(target_cfg["chembl_target_id"], target_cfg["standard_type"], offset=offset)
            new_records = page["activities"]
            if not new_records:
                break
            for r in new_records:
                cache_file.write(json.dumps(r) + "\n")
            cache_file.flush()
            records.extend(new_records)
            offset += len(new_records)
            print(f"  Progres: {len(records)}/{total_count}")

    records = [
        r for r in records
        if r.get("standard_units") == target_cfg["standard_units"]
    ]
    print(f"  {len(records)} record dengan unit {target_cfg['standard_units']}.")
    return records


def clean_and_label(records, threshold_um):
    df = pd.DataFrame(records)
    df = df.dropna(subset=["molecule_chembl_id", "standard_value", "canonical_smiles"])
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]

    # dedup: median MIC per molecule (some compounds have multiple assay entries)
    df = df.groupby(["molecule_chembl_id", "canonical_smiles"], as_index=False)[
        "standard_value"
    ].median()
    print(f"  {len(df)} senyawa unik setelah dedup.")

    mws = []
    valid_idx = []
    for i, smiles in enumerate(df["canonical_smiles"]):
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue
        from rdkit.Chem import Descriptors
        mws.append(Descriptors.MolWt(mol))
        valid_idx.append(i)
    df = df.iloc[valid_idx].reset_index(drop=True)
    df["mol_wt"] = mws
    print(f"  {len(df)} senyawa dengan SMILES valid.")

    # convert MIC ug/mL -> uM using molecular weight
    df["mic_um"] = (df["standard_value"] * 1000) / df["mol_wt"]
    df["active"] = (df["mic_um"] <= threshold_um).astype(int)

    return df


def build_features(df):
    vectors, keep_idx = [], []
    for i, smiles in enumerate(df["canonical_smiles"]):
        vec, _ = featurize(smiles)
        if vec is not None:
            vectors.append(vec)
            keep_idx.append(i)
    df = df.iloc[keep_idx].reset_index(drop=True)
    X = np.vstack(vectors)
    y = df["active"].values
    return X, y, df


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET_ID)
    args = parser.parse_args()

    target_cfg = get_target(args.target)
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    records = fetch_raw_activities(target_cfg)
    df = clean_and_label(records, target_cfg["active_threshold_um"])
    df.to_csv(DATA_DIR / f"raw_dataset_{target_cfg['id']}.csv", index=False)

    X, y, df = build_features(df)
    print(f"Feature matrix: {X.shape}, kelas aktif: {int(y.sum())}/{len(y)}")

    pipeline, metrics = train_and_evaluate(X, y)
    print("Evaluasi:", json.dumps(metrics, indent=2))

    # compress=3: file model jauh lebih kecil sehingga muat di GitHub/Streamlit Cloud
    joblib.dump(pipeline, MODELS_DIR / f"model_{target_cfg['id']}.joblib", compress=3)

    # PCA for applicability-domain visualization (fit on descriptor columns only,
    # fingerprint bits are too high-dimensional/sparse for a meaningful 2D projection)
    descriptor_idx = list(range(len(DESCRIPTOR_NAMES)))
    X_desc = X[:, descriptor_idx]
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_desc)
    joblib.dump(pca, MODELS_DIR / f"pca_{target_cfg['id']}.joblib")
    pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "active": y}).to_csv(
        MODELS_DIR / f"pca_training_points_{target_cfg['id']}.csv", index=False
    )

    metadata = {
        "target_id": target_cfg["id"],
        "target_label": target_cfg["label"],
        "chembl_target_id": target_cfg["chembl_target_id"],
        "active_threshold_um": target_cfg["active_threshold_um"],
        "n_compounds": len(df),
        "n_active": int(y.sum()),
        "n_inactive": int(len(y) - y.sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "descriptor_names": DESCRIPTOR_NAMES,
        "metrics": metrics,
    }
    with open(MODELS_DIR / f"metadata_{target_cfg['id']}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Selesai. Model tersimpan di {MODELS_DIR}/model_{target_cfg['id']}.joblib")


if __name__ == "__main__":
    main()
