"""A run-local transactional checkpoint and cache, safe to read while evaluating."""

import json
import sqlite3
from pathlib import Path

from human_bench.config import canonical


class Store:
    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.db = sqlite3.connect(folder / "run.sqlite", timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS results (
                prompt_id TEXT NOT NULL, sample_id TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (prompt_id, sample_id));
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, stage TEXT, cache_key TEXT, value TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def metadata(self) -> dict:
        return {key: json.loads(value) for key, value in self.db.execute("SELECT * FROM metadata")}

    def set_meta(self, key: str, value):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?, ?)", (key, canonical(value))
            )

    def initialize(self, metadata: dict):
        # Commit the initial identity, catalog, and manifest as one checkpoint.
        with self.db:
            self.db.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [(key, canonical(value)) for key, value in metadata.items()],
            )

    def cached(self, key: str):
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache(self, key: str, value):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO cache VALUES (?, ?)", (key, canonical(value)))

    def result(self, prompt_id: str, sample_id: str):
        row = self.db.execute(
            "SELECT value FROM results WHERE prompt_id=? AND sample_id=?", (prompt_id, sample_id)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_result(self, value: dict):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO results VALUES (?, ?, ?)",
                (value["prompt_id"], value["sample_id"], canonical(value)),
            )

    def results(self) -> list[dict]:
        return [
            json.loads(row[0])
            for row in self.db.execute("SELECT value FROM results ORDER BY prompt_id, sample_id")
        ]

    def attempt(self, stage: str, key: str, value: dict):
        with self.db:
            self.db.execute(
                "INSERT INTO attempts (stage, cache_key, value) VALUES (?, ?, ?)",
                (stage, key, canonical(value)),
            )

    def attempts(self) -> list[dict]:
        return [
            dict(stage=stage, cache_key=key, **json.loads(value))
            for stage, key, value in self.db.execute(
                "SELECT stage, cache_key, value FROM attempts ORDER BY id"
            )
        ]
