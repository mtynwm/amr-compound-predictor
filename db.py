"""SQLite storage: prediction history and the user's saved compound library."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "prediction_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_name TEXT,
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

CREATE TABLE IF NOT EXISTS compounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    smiles TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def _migrate(conn):
    """Add columns introduced after a database was first created."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if existing and "compound_name" not in existing:
        conn.execute("ALTER TABLE predictions ADD COLUMN compound_name TEXT")
        conn.commit()


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# ---------------------------------------------------------------- riwayat prediksi
def insert_prediction(conn, record):
    conn.execute(
        """INSERT INTO predictions
           (compound_name, smiles, canonical_smiles, target_id, predicted_label, confidence,
            mol_wt, log_p, tpsa, num_h_donors, num_h_acceptors, pc1, pc2)
           VALUES (:compound_name, :smiles, :canonical_smiles, :target_id, :predicted_label,
                   :confidence, :mol_wt, :log_p, :tpsa, :num_h_donors, :num_h_acceptors,
                   :pc1, :pc2)""",
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


def delete_predictions(conn, ids):
    if not ids:
        return
    conn.executemany("DELETE FROM predictions WHERE id = ?", [(i,) for i in ids])
    conn.commit()


def clear_predictions(conn, target_id):
    conn.execute("DELETE FROM predictions WHERE target_id = ?", (target_id,))
    conn.commit()


# ---------------------------------------------------------------- pustaka senyawa
def save_compound(conn, name, smiles):
    conn.execute(
        "INSERT INTO compounds (name, smiles) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET smiles = excluded.smiles",
        (name, smiles),
    )
    conn.commit()


def fetch_compounds(conn):
    rows = conn.execute("SELECT * FROM compounds ORDER BY name COLLATE NOCASE").fetchall()
    return [dict(r) for r in rows]


def delete_compound(conn, name):
    conn.execute("DELETE FROM compounds WHERE name = ?", (name,))
    conn.commit()
