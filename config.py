"""Target definitions for AMR compound activity prediction.

Extensible by design: add a new dict to TARGETS to support another
organism/strain. Each target gets its own trained model
(models/model_<id>.joblib) via `python train_model.py --target <id>`.
"""

TARGETS = [
    {
        "id": "saureus_mic",
        "label": "Staphylococcus aureus (termasuk MRSA)",
        "chembl_target_id": "CHEMBL352",
        "standard_type": "MIC",
        "standard_units": "ug.mL-1",
        "active_threshold_um": 10.0,  # MIC <= 10 uM dianggap aktif
    }
]


def get_target(target_id):
    for t in TARGETS:
        if t["id"] == target_id:
            return t
    raise ValueError(f"Target '{target_id}' tidak ditemukan di config.py")


DEFAULT_TARGET_ID = TARGETS[0]["id"]
