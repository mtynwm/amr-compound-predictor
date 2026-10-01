"""SMILES -> feature vector: interpretable RDKit descriptors + Morgan fingerprint.

Used identically at training time and at prediction time so the feature
space always matches what the model was trained on.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 1024

DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3",
]

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_SIZE
)

FEATURE_NAMES = DESCRIPTOR_NAMES + [f"fp_{i}" for i in range(FINGERPRINT_SIZE)]


def smiles_to_mol(smiles):
    """Parse + sanitize a SMILES string. Returns None if invalid."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol


def compute_descriptors(mol):
    """Return a dict of interpretable physicochemical descriptors."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
    }


def compute_fingerprint(mol):
    """Return a Morgan fingerprint as a numpy array of 0/1 ints."""
    fp = _fp_gen.GetFingerprint(mol)
    return np.array(fp, dtype=int)


def featurize(smiles):
    """SMILES -> (feature_vector: np.ndarray, descriptors: dict) or (None, None) if invalid."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, None
    descriptors = compute_descriptors(mol)
    fingerprint = compute_fingerprint(mol)
    vector = np.concatenate([
        np.array([descriptors[name] for name in DESCRIPTOR_NAMES], dtype=float),
        fingerprint,
    ])
    return vector, descriptors
