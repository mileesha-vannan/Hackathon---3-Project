"""
music_grader.py  ·  AI Music Grading System v2.0
=================================================
Flask API + ML pipeline.  All config lives in config.json.
Open dashboard.html in a browser after starting this server.

Install:
    pip install flask flask-cors librosa numpy pandas scikit-learn joblib

Run:
    python music_grader.py

Endpoints:
    GET  /api/status          — system state, dataset stats, active model
    GET  /api/history         — grading history (last N entries)
    GET  /api/models          — last training run model comparison table
    POST /api/grade           — upload audio + optional human score → predict / store
    POST /api/retrain         — force immediate retraining
    GET  /api/config          — return public config subset to dashboard
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import json
import time
import hashlib
import warnings
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib

# ── librosa — explicit sub-module imports resolve IDE "feature" stub warnings ─
import librosa
from librosa import get_duration
from librosa.beat import beat_track
from librosa.effects import harmonic as librosa_harmonic
from librosa.feature import (
    zero_crossing_rate,
    rms,
    spectral_centroid,
    spectral_bandwidth,
    spectral_rolloff,
    spectral_contrast,
    chroma_stft,
    mfcc as librosa_mfcc,
    melspectrogram,
    tonnetz,
)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFIG LOADER  (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════
CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return json.load(fh)


CFG = load_config()

# Convenience shortcuts
SYS    = CFG["system"]
GRADE  = CFG["grading"]
TRAIN  = CFG["training"]
MODELS = CFG["models"]
FEATS  = CFG["features"]
DB_CFG = CFG["database"]

DATA_FILE    = SYS["data_file"]
MODEL_FILE   = SYS["model_file"]
HISTORY_FILE = SYS["history_file"]
UPLOAD_DIR   = Path(SYS["upload_folder"])
UPLOAD_DIR.mkdir(exist_ok=True)

SCORE_MIN: float = GRADE["score_min"]
SCORE_MAX: float = GRADE["score_max"]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATABASE LAYER — CSV primary  |  SQL / Snowflake on standby
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. CSV (always active) ───────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


def save_to_dataset(
    features: dict,
    score: float,
    filename: str,
    source: str = "human",
    predicted: Optional[float] = None,
) -> pd.DataFrame:
    dataset = load_dataset()
    entry = {
        "filename": filename,
        "score": score,
        "predicted_score": predicted,
        "source": source,
        "timestamp": time.time(),
        **features,
    }
    dataset = pd.concat([dataset, pd.DataFrame([entry])], ignore_index=True)
    dataset.to_csv(DATA_FILE, index=False)
    _db_insert_grade(entry)
    return dataset


def get_feature_columns(dataset: pd.DataFrame) -> list:
    meta = {"filename", "score", "predicted_score", "source", "timestamp"}
    return [c for c in dataset.columns if c not in meta]


# ── 3b. Snowflake STANDBY ─────────────────────────────────────────────────────
# Activate: set database.enabled=true, database.type="snowflake" in config.json
# then: pip install snowflake-connector-python

def _snowflake_connect() -> Optional[Any]:
    """Return a live Snowflake connection, or None if not configured."""
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY — uncomment to activate                               │
    # └─────────────────────────────────────────────────────────────────┘
    # import snowflake.connector
    # sf = DB_CFG["snowflake"]
    # return snowflake.connector.connect(
    #     account   = sf["account"],
    #     user      = sf["user"],
    #     password  = sf["password"],
    #     database  = sf["database"],
    #     schema    = sf["schema"],
    #     warehouse = sf["warehouse"],
    #     role      = sf["role"],
    # )
    return None


def _snowflake_ensure_tables(_conn: Any) -> None:
    """Create Snowflake tables if they don't exist."""
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY                                                        │
    # └─────────────────────────────────────────────────────────────────┘
    # sf  = DB_CFG["snowflake"]
    # cur = _conn.cursor()
    # cur.execute(f"""
    #     CREATE TABLE IF NOT EXISTS {sf['table_grades']} (
    #         id         INTEGER AUTOINCREMENT PRIMARY KEY,
    #         filename   VARCHAR,
    #         score      FLOAT,
    #         predicted  FLOAT,
    #         source     VARCHAR,
    #         ts         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP
    #     )
    # """)
    # _conn.commit()


# ── 3c. SQLite STANDBY ────────────────────────────────────────────────────────
def _sqlite_connect() -> Optional[Any]:
    """Return a live SQLite connection, or None if not configured."""
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY — set database.type = "sqlite" to activate            │
    # └─────────────────────────────────────────────────────────────────┘
    # import sqlite3
    # db_path = DB_CFG["sqlite"]["path"]
    # conn = sqlite3.connect(db_path)
    # _sqlite_ensure_tables(conn)
    # return conn
    return None


def _sqlite_ensure_tables(_conn: Any) -> None:
    """Create SQLite tables if they don't exist."""
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY                                                        │
    # └─────────────────────────────────────────────────────────────────┘
    # sq = DB_CFG["sqlite"]
    # _conn.execute(f"""
    #     CREATE TABLE IF NOT EXISTS {sq['table_grades']} (
    #         id INTEGER PRIMARY KEY AUTOINCREMENT,
    #         filename TEXT, score REAL, predicted REAL,
    #         source TEXT, ts REAL
    #     )
    # """)
    # _conn.commit()


# ── 3d. PostgreSQL STANDBY ────────────────────────────────────────────────────
def _postgres_connect() -> Optional[Any]:
    """Return a live psycopg2 connection, or None if not configured."""
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY — set database.type = "postgres" to activate          │
    # └─────────────────────────────────────────────────────────────────┘
    # import psycopg2
    # pg = DB_CFG["postgres"]
    # return psycopg2.connect(
    #     host     = pg["host"],
    #     port     = pg["port"],
    #     dbname   = pg["dbname"],
    #     user     = pg["user"],
    #     password = pg["password"],
    # )
    return None


# ── 3e. Unified DB insert (routes to active backend) ─────────────────────────
def _db_insert_grade(_row: dict) -> None:
    """Insert a grade row into the active SQL backend (if enabled)."""
    if not DB_CFG.get("enabled", False):
        return
    db_type = DB_CFG.get("type", "sqlite")
    try:
        if db_type == "snowflake":
            conn = _snowflake_connect()
        elif db_type == "postgres":
            conn = _postgres_connect()
        else:
            conn = _sqlite_connect()
        if conn is None:
            return
        # Generic parameterised insert — adapt column list to your schema:
        # cur = conn.cursor()
        # cur.execute(
        #     "INSERT INTO graded_tracks (filename, score, predicted, source, ts) "
        #     "VALUES (%s, %s, %s, %s, %s)",
        #     (_row["filename"], _row["score"], _row.get("predicted_score"),
        #      _row["source"], _row.get("timestamp"))
        # )
        # conn.commit()
        # conn.close()
    except Exception as exc:
        print(f"[DB WARN] Insert failed: {exc}")


def _db_insert_model_run(
    _results: dict, _best_name: str, _n_samples: int
) -> None:
    """Log a model comparison run to SQL (if enabled)."""
    if not DB_CFG.get("enabled", False):
        return
    # ┌─────────────────────────────────────────────────────────────────┐
    # │  STANDBY — same pattern as _db_insert_grade                    │
    # └─────────────────────────────────────────────────────────────────┘


# ══════════════════════════════════════════════════════════════════════════════
# 4.  AUDIO FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(filepath: str) -> Optional[dict]:
    duration = SYS["audio_load_duration_seconds"]
    try:
        y, sr = librosa.load(filepath, duration=duration, mono=True)
    except Exception as exc:
        print(f"[EXTRACT ERROR] {filepath}: {exc}")
        return None

    feat: dict = {}

    # Tempo
    tempo, _ = beat_track(y=y, sr=sr)
    feat["tempo"] = float(np.atleast_1d(tempo)[0])

    # Zero-crossing rate
    zcr_arr = zero_crossing_rate(y)
    feat["zcr_mean"] = float(np.mean(zcr_arr))
    feat["zcr_std"]  = float(np.std(zcr_arr))

    # RMS energy
    rms_arr = rms(y=y)
    feat["rms_mean"] = float(np.mean(rms_arr))
    feat["rms_std"]  = float(np.std(rms_arr))
    feat["rms_max"]  = float(np.max(rms_arr))

    # Spectral centroid
    sc_arr = spectral_centroid(y=y, sr=sr)
    feat["spectral_centroid_mean"] = float(np.mean(sc_arr))
    feat["spectral_centroid_std"]  = float(np.std(sc_arr))

    # Spectral bandwidth
    sb_arr = spectral_bandwidth(y=y, sr=sr)
    feat["spectral_bandwidth_mean"] = float(np.mean(sb_arr))
    feat["spectral_bandwidth_std"]  = float(np.std(sb_arr))

    # Spectral rolloff
    sro_arr = spectral_rolloff(y=y, sr=sr)
    feat["spectral_rolloff_mean"] = float(np.mean(sro_arr))

    # Spectral contrast
    if FEATS["include_spectral_contrast"]:
        contrast_arr = spectral_contrast(y=y, sr=sr)
        for i, val in enumerate(np.mean(contrast_arr, axis=1)):
            feat[f"spectral_contrast_{i}"] = float(val)

    # Chroma
    if FEATS["include_chroma"]:
        chroma_arr = chroma_stft(y=y, sr=sr)
        feat["chroma_mean"] = float(np.mean(chroma_arr))
        feat["chroma_std"]  = float(np.std(chroma_arr))
        for i, val in enumerate(np.mean(chroma_arr, axis=1)):
            feat[f"chroma_{i}"] = float(val)

    # MFCCs
    mfcc_arr = librosa_mfcc(y=y, sr=sr, n_mfcc=FEATS["n_mfcc"])
    for i in range(FEATS["n_mfcc"]):
        feat[f"mfcc_{i}_mean"] = float(np.mean(mfcc_arr[i]))
        feat[f"mfcc_{i}_std"]  = float(np.std(mfcc_arr[i]))

    # Mel spectrogram summary
    if FEATS["include_mel_summary"]:
        mel_arr = melspectrogram(y=y, sr=sr)
        feat["mel_mean"] = float(np.mean(mel_arr))
        feat["mel_std"]  = float(np.std(mel_arr))
        feat["mel_max"]  = float(np.max(mel_arr))

    # Tonnetz
    if FEATS["include_tonnetz"]:
        harm_y   = librosa_harmonic(y)
        tntz_arr = tonnetz(y=harm_y, sr=sr)
        for i, val in enumerate(np.mean(tntz_arr, axis=1)):
            feat[f"tonnetz_{i}"] = float(val)

    feat["duration"] = float(get_duration(y=y, sr=sr))
    return feat


def file_hash(filepath: str) -> str:
    """Return a short MD5 fingerprint for deduplication."""
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ML MODEL DEFINITIONS  (built from config)
# ══════════════════════════════════════════════════════════════════════════════
def build_models() -> dict:
    pipelines: dict = {}
    mc = MODELS

    if mc["RandomForest"]["enabled"]:
        c = mc["RandomForest"]
        pipelines["RandomForest"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=c["n_estimators"],
                random_state=c["random_state"],
                n_jobs=c["n_jobs"],
            )),
        ])

    if mc["GradientBoosting"]["enabled"]:
        c = mc["GradientBoosting"]
        pipelines["GradientBoosting"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=c["n_estimators"],
                learning_rate=c["learning_rate"],
                max_depth=c["max_depth"],
                random_state=c["random_state"],
            )),
        ])

    if mc["ExtraTrees"]["enabled"]:
        c = mc["ExtraTrees"]
        pipelines["ExtraTrees"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ExtraTreesRegressor(
                n_estimators=c["n_estimators"],
                random_state=c["random_state"],
                n_jobs=c["n_jobs"],
            )),
        ])

    if mc["SVR"]["enabled"]:
        c = mc["SVR"]
        pipelines["SVR"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel=c["kernel"], C=c["C"], epsilon=c["epsilon"])),
        ])

    if mc["NeuralNet"]["enabled"]:
        c = mc["NeuralNet"]
        pipelines["NeuralNet"] = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=tuple(c["hidden_layer_sizes"]),
                max_iter=c["max_iter"],
                learning_rate_init=c["learning_rate_init"],
                random_state=c["random_state"],
                early_stopping=c["early_stopping"],
                validation_fraction=c["validation_fraction"],
            )),
        ])

    return pipelines


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL TRAINING, EVALUATION & SELECTION
# ══════════════════════════════════════════════════════════════════════════════
_last_training_results: dict = {}   # kept in memory for /api/models endpoint


def train_and_select(dataset: pd.DataFrame) -> tuple:
    """Train all enabled models, select best, return (pipeline, name, results)."""
    global _last_training_results
    feature_cols = get_feature_columns(dataset)
    X = dataset[feature_cols].values.astype(float)
    y = dataset["score"].values.astype(float)

    model_map = build_models()
    results: dict = {}
    use_cv  = len(dataset) >= TRAIN["cv_min_samples"]
    n_folds = TRAIN["cv_folds"]

    print(f"\n[TRAIN] {len(model_map)} models · {len(dataset)} samples · CV={'yes' if use_cv else 'no'}")

    for name, pipeline in model_map.items():
        try:
            if use_cv:
                cv_scores = cross_val_score(
                    pipeline, X, y, cv=n_folds, scoring="r2", n_jobs=-1
                )
                r2   = float(np.mean(cv_scores))
                r2sd = float(np.std(cv_scores))
            else:
                pipeline.fit(X, y)
                preds = pipeline.predict(X)
                r2    = float(r2_score(y, preds))
                r2sd  = 0.0

            pipeline.fit(X, y)
            preds_full = pipeline.predict(X)
            mae = float(mean_absolute_error(y, preds_full))

            results[name] = {"r2": r2, "r2_std": r2sd, "mae": mae,
                             "pipeline": pipeline, "ok": True}
            print(f"  {name:<20}  R²={r2:+.4f} ±{r2sd:.4f}  MAE={mae:.4f}")
        except Exception as exc:
            results[name] = {"r2": -999, "r2_std": 0, "mae": 999,
                             "pipeline": None, "ok": False, "error": str(exc)}
            print(f"  {name:<20}  FAILED: {exc}")

    valid = {k: v for k, v in results.items() if v["ok"]}
    if not valid:
        raise RuntimeError("All models failed to train.")

    best_name = max(valid, key=lambda k: valid[k]["r2"])
    print(f"[SELECT] Best: {best_name}  R²={valid[best_name]['r2']:+.4f}")

    _last_training_results = {
        k: {
            "r2": v["r2"], "r2_std": v["r2_std"], "mae": v["mae"],
            "ok": v["ok"], "selected": (k == best_name),
        }
        for k, v in results.items()
    }

    _db_insert_model_run(_last_training_results, best_name, len(dataset))
    return valid[best_name]["pipeline"], best_name, results


def save_model(pipeline: Any, model_name: str, feature_cols: list) -> None:
    payload = {
        "pipeline": pipeline,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "trained_at": time.time(),
    }
    joblib.dump(payload, MODEL_FILE)
    print(f"[SAVED] {MODEL_FILE}")


def load_model_payload() -> Optional[dict]:
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)  # type: ignore[return-value]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
_model_payload: Optional[dict] = None   # in-memory cache


def get_model_payload() -> Optional[dict]:
    global _model_payload
    if _model_payload is None:
        _model_payload = load_model_payload()
    return _model_payload


def predict_score(features: dict, payload: dict) -> float:
    feat_cols = payload["feature_cols"]
    pipeline  = payload["pipeline"]
    row_vals  = [features.get(col, 0.0) for col in feat_cols]
    X_input   = np.array(row_vals, dtype=float).reshape(1, -1)
    pred      = float(pipeline.predict(X_input)[0])
    return float(np.clip(pred, SCORE_MIN, SCORE_MAX))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SESSION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as fh:
            return json.load(fh)
    return []


def append_history(entry: dict) -> None:
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as fh:
        json.dump(history, fh, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  RETRAINING LOGIC  (fires after every grade — configurable in config.json)
# ══════════════════════════════════════════════════════════════════════════════
_grades_since_retrain: int = 0


def retrain_if_needed(dataset: pd.DataFrame, force: bool = False) -> bool:
    global _grades_since_retrain, _model_payload
    _grades_since_retrain += 1
    threshold   = TRAIN["retrain_after_every_n_grades"]
    min_samples = TRAIN["min_samples_to_train"]

    if len(dataset) < min_samples:
        return False

    if force or _grades_since_retrain >= threshold:
        try:
            feature_cols = get_feature_columns(dataset)
            pipeline, best_name, _ = train_and_select(dataset)
            save_model(pipeline, best_name, feature_cols)
            _model_payload = {
                "pipeline": pipeline,
                "model_name": best_name,
                "feature_cols": feature_cols,
                "trained_at": time.time(),
            }
            _grades_since_retrain = 0
            return True
        except Exception as exc:
            print(f"[RETRAIN ERROR] {exc}")
            traceback.print_exc()
            return False
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 10.  FLASK API
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=".")
CORS(app)   # allow dashboard.html to call the API from file://


@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


# ── GET /api/config ──────────────────────────────────────────────────────────
@app.route("/api/config")
def api_config():
    return jsonify({
        "scoreMin":   SCORE_MIN,
        "scoreMax":   SCORE_MAX,
        "minSamples": TRAIN["min_samples_to_train"],
        "systemName": SYS["name"],
        "dbEnabled":  DB_CFG.get("enabled", False),
        "dbType":     DB_CFG.get("type", "none"),
        "dashboard":  CFG["dashboard"],
    })


# ── GET /api/status ──────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    dataset = load_dataset()
    payload = get_model_payload()
    n       = len(dataset)

    stats: dict = {}
    if n > 0:
        stats = {
            "count":     n,
            "mean":      round(float(dataset["score"].mean()), 3),
            "std":       round(float(dataset["score"].std()), 3),
            "min":       round(float(dataset["score"].min()), 3),
            "max":       round(float(dataset["score"].max()), 3),
            "sources":   dataset["source"].value_counts().to_dict()
                         if "source" in dataset else {},
            "histogram": _score_histogram(dataset),
        }

    model_info: Optional[dict] = None
    if payload:
        model_info = {
            "name":      payload["model_name"],
            "trainedAt": payload.get("trained_at"),
            "features":  len(payload["feature_cols"]),
        }

    cold_start = n < TRAIN["min_samples_to_train"]

    return jsonify({
        "ok":                 True,
        "coldStart":          cold_start,
        "remaining":          max(0, TRAIN["min_samples_to_train"] - n),
        "stats":              stats,
        "model":              model_info,
        "gradesSinceRetrain": _grades_since_retrain,
        "retrainEvery":       TRAIN["retrain_after_every_n_grades"],
    })


def _score_histogram(dataset: pd.DataFrame, bins: int = 10) -> list:
    counts, edges = np.histogram(
        dataset["score"].dropna(), bins=bins, range=(SCORE_MIN, SCORE_MAX)
    )
    return [
        {"label": f"{edges[i]:.1f}–{edges[i + 1]:.1f}", "count": int(counts[i])}
        for i in range(len(counts))
    ]


# ── GET /api/history ─────────────────────────────────────────────────────────
@app.route("/api/history")
def api_history():
    history  = load_history()
    max_rows = CFG["dashboard"]["history_max_rows"]
    return jsonify({"ok": True, "history": history[-max_rows:][::-1]})


# ── GET /api/models ──────────────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    return jsonify({"ok": True, "results": _last_training_results})


# ── POST /api/grade ──────────────────────────────────────────────────────────
@app.route("/api/grade", methods=["POST"])
def api_grade():
    """
    Multipart form fields:
        file   — audio file
        score  — (optional) human score; omit to get AI prediction only
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    upload = request.files["file"]
    ext    = Path(upload.filename).suffix.lower().lstrip(".")
    if ext not in SYS["allowed_extensions"]:
        return jsonify({"ok": False, "error": f"Unsupported format: .{ext}"}), 400

    safe_name = f"{int(time.time())}_{upload.filename}"
    fpath     = UPLOAD_DIR / safe_name
    upload.save(str(fpath))

    features = extract_features(str(fpath))
    if features is None:
        return jsonify({"ok": False, "error": "Feature extraction failed"}), 500

    payload     = get_model_payload()
    human_score = request.form.get("score", "").strip()
    dataset     = load_dataset()
    cold_start  = len(dataset) < TRAIN["min_samples_to_train"]

    # ── Cold-start: model not ready yet ──────────────────────────────────────
    if cold_start:
        if not human_score:
            return jsonify({
                "ok":        True,
                "coldStart": True,
                "remaining": max(0, TRAIN["min_samples_to_train"] - len(dataset)),
                "message":   "Provide a score to build the training set.",
            })

        score   = float(human_score)
        dataset = save_to_dataset(features, score, upload.filename, source="human")
        append_history({
            "filename":    upload.filename,
            "final_score": score,
            "source":      "human",
            "timestamp":   time.time(),
        })
        retrain_if_needed(dataset)
        return jsonify({
            "ok":        True,
            "coldStart": len(dataset) < TRAIN["min_samples_to_train"],
            "remaining": max(0, TRAIN["min_samples_to_train"] - len(dataset)),
            "score":     score,
            "source":    "human",
            "total":     len(dataset),
        })

    # ── AI prediction (no score submitted yet) ────────────────────────────────
    predicted = predict_score(features, payload)

    if not human_score:
        return jsonify({
            "ok":              True,
            "coldStart":       False,
            "predicted":       round(predicted, 2),
            "awaitingConfirm": True,
            "modelName":       payload["model_name"],
            "filename":        upload.filename,
            "_featuresDump":   {k: round(v, 4) for k, v in list(features.items())[:10]},
        })

    # ── Score submitted (acceptance or override) ──────────────────────────────
    final  = float(human_score)
    source = "ai_accepted" if abs(final - predicted) < 0.01 else "human_corrected"

    dataset = save_to_dataset(
        features, final, upload.filename, source=source, predicted=predicted
    )
    append_history({
        "filename":        upload.filename,
        "predicted_score": predicted,
        "final_score":     final,
        "source":          source,
        "model":           payload["model_name"],
        "timestamp":       time.time(),
    })

    retrained      = retrain_if_needed(dataset)
    active_payload = get_model_payload()

    return jsonify({
        "ok":         True,
        "coldStart":  False,
        "predicted":  round(predicted, 2),
        "finalScore": round(final, 2),
        "source":     source,
        "total":      len(dataset),
        "retrained":  retrained,
        "modelName":  active_payload["model_name"] if active_payload else None,
    })


# ── POST /api/retrain ────────────────────────────────────────────────────────
@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    dataset = load_dataset()
    if len(dataset) < TRAIN["min_samples_to_train"]:
        return jsonify({
            "ok":    False,
            "error": f"Need ≥{TRAIN['min_samples_to_train']} samples",
        }), 400

    success        = retrain_if_needed(dataset, force=True)
    active_payload = get_model_payload()
    return jsonify({
        "ok":      success,
        "model":   active_payload["model_name"] if active_payload else None,
        "results": _last_training_results,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n══════════════════════════════════════════")
    print("  🎵  AI Music Grading System  v2.0")
    print("══════════════════════════════════════════")

    boot_dataset = load_dataset()

    if len(boot_dataset) >= TRAIN["min_samples_to_train"] and load_model_payload() is None:
        print(f"[BOOT] Dataset has {len(boot_dataset)} entries but no model. Training now...")
        retrain_if_needed(boot_dataset, force=True)
    elif load_model_payload() is not None:
        _model_payload = load_model_payload()
        print(f"[BOOT] Loaded model: {_model_payload['model_name']}")  # type: ignore[index]
    else:
        print(f"[BOOT] Cold-start mode. Grade {TRAIN['min_samples_to_train']} tracks to begin.")

    host = SYS["flask_host"]
    port = SYS["flask_port"]
    print(f"\n  Dashboard → http://{host}:{port}/")
    print(f"  API base  → http://{host}:{port}/api/\n")
    app.run(host=host, port=port, debug=SYS["flask_debug"])
