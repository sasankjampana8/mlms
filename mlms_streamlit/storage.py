# storage.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Base directory is the folder where this file lives (mlms_streamlit/)
_BASE_DIR = Path(__file__).resolve().parent

# Persist under repo folder (works on local + Streamlit Cloud)
_DATA_DIR_PATH = _BASE_DIR / "data"
_ARTIFACTS_DIR_PATH = _BASE_DIR / "artifacts"
_DB_PATH = _DATA_DIR_PATH / "mlms.db"

# Export strings for backward compatibility with your app code
BASE_DIR = str(_BASE_DIR)
DB_PATH = str(_DB_PATH)
DATA_DIR = str(_DATA_DIR_PATH)
ARTIFACTS_DIR = str(_ARTIFACTS_DIR_PATH)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dirs() -> None:
    _DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    _ARTIFACTS_DIR_PATH.mkdir(parents=True, exist_ok=True)


def abs_path(path_str: str) -> str:
    """
    Convert a stored relative path (e.g., 'data/project_1/x.csv') into an absolute path.
    If already absolute, returns as-is.
    """
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((_BASE_DIR / p).resolve())


def rel_path(path_str: str) -> str:
    """
    Convert an absolute path into a path relative to BASE_DIR for portability.
    If already relative, returns as-is.
    """
    if not path_str:
        return path_str
    p = Path(path_str)
    if not p.is_absolute():
        return path_str
    try:
        return str(p.resolve().relative_to(_BASE_DIR))
    except Exception:
        # If it can't be made relative, keep absolute (won't crash)
        return str(p)


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    ensure_dirs()
    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            summary_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            dataset_id INTEGER NOT NULL,
            problem_type TEXT NOT NULL,
            target_col TEXT DEFAULT '',
            metric TEXT NOT NULL,
            algorithms_json TEXT NOT NULL,
            config_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            status TEXT NOT NULL,
            params_json TEXT DEFAULT '{}',
            metrics_json TEXT DEFAULT '{}',
            logs_text TEXT DEFAULT '',
            model_path TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            run_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(project_id) REFERENCES projects(id),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """
    )

    conn.commit()
    conn.close()


# ---------------------------
# Projects
# ---------------------------
def create_project(name: str, description: str = "") -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (name, description, created_at) VALUES (?, ?, ?)",
        (name.strip(), description.strip(), _now()),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()
    return int(pid)


def list_projects() -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------
# Datasets
# ---------------------------
def _dataset_folder(project_id: int) -> Path:
    path = _DATA_DIR_PATH / f"project_{project_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataset_csv(project_id: int, dataset_name: str, csv_bytes: bytes) -> Tuple[str, int]:
    """
    Saves uploaded CSV as a new dataset version.
    Returns (RELATIVE file_path, version).
    """
    conn = connect()
    cur = conn.cursor()

    row = cur.execute(
        """
        SELECT MAX(version) AS max_v
        FROM datasets
        WHERE project_id = ? AND name = ?
        """,
        (project_id, dataset_name),
    ).fetchone()

    max_v = int(row["max_v"]) if row and row["max_v"] is not None else 0
    version = max_v + 1

    folder = _dataset_folder(project_id)
    file_path_abs = folder / f"{dataset_name}_v{version}.csv"
    file_path_abs.write_bytes(csv_bytes)

    conn.close()
    return rel_path(str(file_path_abs)), version


def create_dataset_record(
    project_id: int,
    name: str,
    version: int,
    file_path: str,
    summary: Dict[str, Any],
) -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO datasets (project_id, name, version, file_path, summary_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project_id, name, version, file_path, json.dumps(summary), _now()),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return int(did)


def delete_dataset(dataset_id: int) -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    conn.close()


def list_datasets(project_id: int) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM datasets WHERE project_id = ? ORDER BY id DESC",
        (project_id,),
    ).fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["summary"] = json.loads(d.get("summary_json") or "{}")
        except Exception:
            d["summary"] = {}
        out.append(d)
    return out


def get_dataset(dataset_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    r = cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
    conn.close()
    if not r:
        return None
    d = dict(r)
    try:
        d["summary"] = json.loads(d.get("summary_json") or "{}")
    except Exception:
        d["summary"] = {}
    return d


# ---------------------------
# Experiments
# ---------------------------
def create_experiment(
    project_id: int,
    name: str,
    dataset_id: int,
    problem_type: str,
    target_col: str,
    metric: str,
    algorithms: List[str],
    config: Dict[str, Any],
) -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO experiments
        (project_id, name, dataset_id, problem_type, target_col, metric, algorithms_json, config_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            name.strip(),
            dataset_id,
            problem_type,
            (target_col or "").strip(),
            metric,
            json.dumps(algorithms),
            json.dumps(config or {}),
            _now(),
        ),
    )
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    return int(eid)


def list_experiments(project_id: int) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM experiments WHERE project_id = ? ORDER BY id DESC",
        (project_id,),
    ).fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["algorithms"] = json.loads(d.get("algorithms_json") or "[]")
        d["config"] = json.loads(d.get("config_json") or "{}")
        out.append(d)
    return out


def get_experiment(experiment_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    r = cur.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    conn.close()
    if not r:
        return None
    d = dict(r)
    d["algorithms"] = json.loads(d.get("algorithms_json") or "[]")
    d["config"] = json.loads(d.get("config_json") or "{}")
    return d


# ---------------------------
# Runs
# ---------------------------
def create_run(experiment_id: int, algorithm: str, params: Dict[str, Any]) -> int:
    conn = connect()
    cur = conn.cursor()
    now = _now()
    cur.execute(
        """
        INSERT INTO runs (experiment_id, algorithm, status, params_json, metrics_json, logs_text, model_path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (experiment_id, algorithm, "CREATED", json.dumps(params or {}), "{}", "", "", now, now),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)


def update_run(
    run_id: int,
    *,
    status: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    logs_text: Optional[str] = None,
    model_path: Optional[str] = None,
) -> None:
    conn = connect()
    cur = conn.cursor()

    fields = []
    vals: List[Any] = []

    if status is not None:
        fields.append("status = ?")
        vals.append(status)
    if metrics is not None:
        fields.append("metrics_json = ?")
        vals.append(json.dumps(metrics))
    if logs_text is not None:
        fields.append("logs_text = ?")
        vals.append(logs_text)
    if model_path is not None:
        fields.append("model_path = ?")
        vals.append(rel_path(model_path))

    fields.append("updated_at = ?")
    vals.append(_now())

    vals.append(run_id)
    cur.execute(f"UPDATE runs SET {', '.join(fields)} WHERE id = ?", tuple(vals))
    conn.commit()
    conn.close()


def list_runs(experiment_id: int) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM runs WHERE experiment_id = ? ORDER BY id DESC",
        (experiment_id,),
    ).fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d.get("params_json") or "{}")
        d["metrics"] = json.loads(d.get("metrics_json") or "{}")
        out.append(d)
    return out


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    r = cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if not r:
        return None
    d = dict(r)
    d["params"] = json.loads(d.get("params_json") or "{}")
    d["metrics"] = json.loads(d.get("metrics_json") or "{}")
    return d


# ---------------------------
# Deployments
# ---------------------------
def create_deployment(project_id: int, name: str, run_id: int) -> int:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO deployments (project_id, name, run_id, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (project_id, name.strip(), run_id, "ACTIVE", _now()),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return int(did)


def list_deployments(project_id: int) -> List[Dict[str, Any]]:
    conn = connect()
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM deployments WHERE project_id = ? ORDER BY id DESC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
