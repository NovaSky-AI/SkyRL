"""TViz tracking backend for local visualization of SkyRL training runs."""

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any


METRIC_MAPPING = {
    "reward/avg_raw_reward": "reward_mean",
    "reward/avg_reward": "reward_mean",
    "reward/mean_positive_reward": "reward_mean",
    "reward_mean": "reward_mean",
    "policy/loss": "loss",
    "critic/loss": "loss",
    "loss": "loss",
    "policy/kl_divergence": "kl_divergence",
    "kl_divergence": "kl_divergence",
    "kl": "kl_divergence",
    "policy/entropy": "entropy",
    "entropy": "entropy",
    "trainer/learning_rate": "learning_rate",
    "learning_rate": "learning_rate",
    "lr": "learning_rate",
    "generate/avg_num_tokens": "ac_tokens_per_turn",
    "generate/avg_tokens_non_zero_rewards": "ac_tokens_per_turn",
    "ac_tokens_per_turn": "ac_tokens_per_turn",
    "timing/total": "time_total",
    "timing/generate": "sampling_time_mean",
    "time_total": "time_total",
}

# Schema for training visualization tables (added to tinker.db)
SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT DEFAULT 'rl',
    modality TEXT DEFAULT 'text',
    config TEXT,
    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS training_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    reward_mean REAL,
    reward_std REAL,
    loss REAL,
    kl_divergence REAL,
    entropy REAL,
    learning_rate REAL,
    ac_tokens_per_turn REAL,
    ob_tokens_per_turn REAL,
    total_ac_tokens INTEGER,
    total_turns INTEGER,
    sampling_time_mean REAL,
    time_total REAL,
    frac_mixed REAL,
    frac_all_good REAL,
    frac_all_bad REAL,
    extras TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES training_runs(id),
    UNIQUE(run_id, step)
);

CREATE INDEX IF NOT EXISTS idx_training_steps_run_id ON training_steps(run_id);
"""


def get_tinker_db_path() -> Path:
    """Get the tinker database path from environment or default."""
    if db_url := os.environ.get("TX_DATABASE_URL"):
        if db_url.startswith("sqlite:///"):
            return Path(db_url.replace("sqlite:///", ""))
    return Path(__file__).parent.parent.parent.parent / "skyrl-tx" / "tx" / "tinker" / "tinker.db"


class TvizTracker:
    """TViz tracking backend that writes metrics to the tinker database."""

    def __init__(self, experiment_name: str, config: dict[str, Any] | None = None):
        self.db_path = get_tinker_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = experiment_name
        self._config = config
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        conn = self._get_conn()
        conn.executescript(SCHEMA)

        config_json = None
        if self._config is not None:
            try:
                from omegaconf import OmegaConf
                if hasattr(self._config, "_content"):
                    self._config = OmegaConf.to_container(self._config, resolve=True)
            except ImportError:
                pass
            config_json = json.dumps(self._config)

        conn.execute(
            "INSERT INTO training_runs (id, name, type, modality, config) VALUES (?, ?, ?, ?, ?)",
            (self.run_id, self.run_name, "rl", "text", config_json),
        )
        conn.commit()
        self._initialized = True
        print(f"TViz dashboard: http://localhost:3003/training-run/{self.run_id}")

    def _map_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        mapped = {}
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            if key in METRIC_MAPPING:
                tviz_key = METRIC_MAPPING[key]
                if tviz_key not in mapped:
                    mapped[tviz_key] = value
            else:
                mapped[key] = value
        return mapped

    def log(self, data: dict[str, Any], step: int) -> None:
        self._ensure_initialized()
        metrics = self._map_metrics(data)

        known_cols = [
            "reward_mean", "reward_std", "loss", "kl_divergence", "entropy",
            "learning_rate", "ac_tokens_per_turn", "ob_tokens_per_turn",
            "total_ac_tokens", "total_turns", "sampling_time_mean", "time_total",
            "frac_mixed", "frac_all_good", "frac_all_bad",
        ]

        values = {col: metrics.pop(col, None) for col in known_cols}
        extras = json.dumps(metrics) if metrics else None

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO training_steps
            (run_id, step, reward_mean, reward_std, loss, kl_divergence, entropy,
             learning_rate, ac_tokens_per_turn, ob_tokens_per_turn, total_ac_tokens,
             total_turns, sampling_time_mean, time_total, frac_mixed, frac_all_good,
             frac_all_bad, extras)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id, step,
                values["reward_mean"], values["reward_std"], values["loss"],
                values["kl_divergence"], values["entropy"], values["learning_rate"],
                values["ac_tokens_per_turn"], values["ob_tokens_per_turn"],
                values["total_ac_tokens"], values["total_turns"],
                values["sampling_time_mean"], values["time_total"],
                values["frac_mixed"], values["frac_all_good"], values["frac_all_bad"],
                extras,
            ),
        )
        conn.commit()

    def finish(self) -> None:
        if self._conn is not None:
            self._conn.execute(
                "UPDATE training_runs SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
                (self.run_id,),
            )
            self._conn.commit()
            self._conn.close()
            self._conn = None
