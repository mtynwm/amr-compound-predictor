"""SQLite storage for the prediction history."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "prediction_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles TEXT,
    canonical_smiles TEXT,
    target_id TEXT,
    predicted_label TEXT,
    confidence REAL,
    mol_wt REAL,
    log_p REAL,
    tpsa REAL,
    num_h_donors INTEGER,
    num_h_acceptors INTEGER,
    pc1 REAL,
    pc2 REAL,
    predicted_at TEXT DEFAULT (datetime('now'))
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def insert_prediction(conn, record):
    conn.execute(
        """INSERT INTO predictions
           (smiles, canonical_smiles, target_id, predicted_label, confidence,
            mol_wt, log_p, tpsa, num_h_donors, num_h_acceptors, pc1, pc2)
           VALUES (:smiles, :canonical_smiles, :target_id, :predicted_label, :confidence,
                   :mol_wt, :log_p, :tpsa, :num_h_donors, :num_h_acceptors, :pc1, :pc2)""",
        record,
    )
    conn.commit()


def fetch_all(conn, target_id=None):
    if target_id:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE target_id = ? ORDER BY predicted_at DESC",
            (target_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM predictions ORDER BY predicted_at DESC").fetchall()
    return [dict(r) for r in rows]
