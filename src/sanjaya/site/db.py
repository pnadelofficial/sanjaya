"""
db.py — SQLite schema creation and row writing for sanjaya.

Two responsibilities:
  1. create_schema()       — create all structural and annotation tables once at
                             build start. Idempotent; safe to call on an existing DB.
  2. write_* methods       — insert individual rows. Writes are not auto-committed;
                             call commit() explicitly (e.g. once per chunk) so bulk
                             inserts stay efficient.

Usable independently of Generator:

    from sanjaya.site.db import AnnotationDB

    with AnnotationDB(Path("output/thucydides/annotations.db")) as db:
        db.create_schema()
        db.register_annotators(annotators)

        db.write_chunk("1.1", sequence=0, source_path="chunks/1.1.xml")
        db.write_sentence("1.1-0", chunk_id="1.1", position=0)
        db.write_token("tk-1.1-0-3", sentence_id="1.1-0", chunk_id="1.1",
                       position=3, form="οἱ")
        db.write_word_annotation("tk-1.1-0-3", annotator="gloss", value="the")
        db.write_sentence_annotation("1.1-0", annotator="translation",
                                     value="The men who …")
        db.commit()

Write ordering must respect foreign-key dependencies:
    chunks → sentences → tokens → word_annotations
                                → word_annotation_features
                                → sentence_annotations
"""

import json
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional


def normalize_form(form: str) -> str:
    """
    Canonicalise a surface form for storage and matching.

    Applies Unicode NFC so that visually identical Greek written with
    decomposed (NFD) vs precomposed (NFC) codepoints counts as one form.
    Display-only differences beyond NFC (case, sigma variants) are left
    intact deliberately: NFC is non-destructive and safe to render.
    """
    return unicodedata.normalize("NFC", form)


class AnnotationDB:
    """
    SQLite-backed store for sanjaya's structural and annotation data.

    Use as a context manager to ensure the connection is closed on exit.
    Individual write_* calls do not commit — batch them and call commit()
    at a cadence that makes sense for the caller (typically once per chunk).
    """

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """Create all tables. Idempotent — safe to call on an existing DB."""
        self.conn.executescript("""
            -- structural tables: fixed regardless of which annotators are used
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id    TEXT PRIMARY KEY,
                sequence    INTEGER,
                source_path TEXT
            );

            CREATE TABLE IF NOT EXISTS sentences (
                sentence_id TEXT PRIMARY KEY,
                chunk_id    TEXT REFERENCES chunks(chunk_id),
                position    INTEGER,
                speaker     TEXT
            );

            CREATE TABLE IF NOT EXISTS tokens (
                token_id    TEXT PRIMARY KEY,
                sentence_id TEXT REFERENCES sentences(sentence_id),
                chunk_id    TEXT REFERENCES chunks(chunk_id),
                position    INTEGER,
                form        TEXT NOT NULL
            );

            -- annotation tables: purely additive; no schema migration needed
            -- when annotators are added or removed between builds
            CREATE TABLE IF NOT EXISTS word_annotations (
                id          INTEGER PRIMARY KEY,
                token_id    TEXT REFERENCES tokens(token_id),
                annotator   TEXT NOT NULL,
                value       TEXT NOT NULL,
                confidence  REAL,
                UNIQUE(token_id, annotator)
            );

            CREATE TABLE IF NOT EXISTS sentence_annotations (
                id          INTEGER PRIMARY KEY,
                sentence_id TEXT REFERENCES sentences(sentence_id),
                annotator   TEXT NOT NULL,
                value       TEXT NOT NULL,
                confidence  REAL,
                UNIQUE(sentence_id, annotator)
            );

            -- extra per-token fields an annotator returns beyond the canonical
            -- "label" (e.g. lemma, part_of_speech, morphology). Key/value so any
            -- field is queryable without a schema migration when annotators change.
            CREATE TABLE IF NOT EXISTS word_annotation_features (
                id          INTEGER PRIMARY KEY,
                token_id    TEXT REFERENCES tokens(token_id),
                annotator   TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                UNIQUE(token_id, annotator, key)
            );

            -- build-time metadata: one row per annotator, populated by
            -- register_annotators() before any data rows are written
            CREATE TABLE IF NOT EXISTS annotators (
                name          TEXT PRIMARY KEY,
                level         TEXT NOT NULL CHECK(level IN ('word', 'sentence')),
                output_schema JSON
            );

            -- indexes for the search page's frequency queries (form <-> value)
            CREATE INDEX IF NOT EXISTS idx_tokens_form ON tokens(form);
            CREATE INDEX IF NOT EXISTS idx_word_annotations_value ON word_annotations(value);
            CREATE INDEX IF NOT EXISTS idx_word_annotations_annotator ON word_annotations(annotator);
            -- (key, value) leads so lemma lookups (key='lemma' AND value LIKE ?) hit
            -- the index; token_id lead so per-token feature joins stay cheap.
            CREATE INDEX IF NOT EXISTS idx_waf_key_value ON word_annotation_features(key, value);
            CREATE INDEX IF NOT EXISTS idx_waf_token ON word_annotation_features(token_id, annotator);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Annotator registration
    # ------------------------------------------------------------------

    def register_annotators(self, annotators: list) -> None:
        """
        Populate the annotators table from a list of WordAnnotator /
        SentenceAnnotator instances. Called once per build, after create_schema().

        output_schema is serialised as {"field": "typename"} JSON, or NULL when
        the annotator's output_schema is None (plain-string output).
        """
        from sanjaya.llm.annotations import WordAnnotator, SentenceAnnotator

        rows = []
        for a in annotators:
            if isinstance(a, WordAnnotator):
                level = "word"
            elif isinstance(a, SentenceAnnotator):
                level = "sentence"
            else:
                raise ValueError(
                    f"{a!r} is not a WordAnnotator or SentenceAnnotator"
                )
            schema_json = (
                json.dumps({k: v.__name__ for k, v in a.output_schema.items()})
                if a.output_schema is not None
                else None
            )
            rows.append((a.role, level, schema_json))

        self.conn.executemany(
            "INSERT OR REPLACE INTO annotators (name, level, output_schema)"
            " VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Structural writes
    # ------------------------------------------------------------------

    def write_chunk(self, chunk_id: str, sequence: int, source_path: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, sequence, source_path)"
            " VALUES (?, ?, ?)",
            (chunk_id, sequence, str(source_path)),
        )

    def write_sentence(
        self,
        sentence_id: str,
        chunk_id: str,
        position: int,
        speaker: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sentences"
            " (sentence_id, chunk_id, position, speaker) VALUES (?, ?, ?, ?)",
            (sentence_id, chunk_id, position, speaker),
        )

    def write_token(
        self,
        token_id: str,
        sentence_id: str,
        chunk_id: str,
        position: int,
        form: str,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO tokens"
            " (token_id, sentence_id, chunk_id, position, form)"
            " VALUES (?, ?, ?, ?, ?)",
            (token_id, sentence_id, chunk_id, position, normalize_form(form)),
        )

    # ------------------------------------------------------------------
    # Annotation writes
    # ------------------------------------------------------------------

    def write_word_annotation(
        self,
        token_id: str,
        annotator: str,
        value: str,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Write one word-level annotation. value must be the canonical "label" string.
        confidence is optional; supply it when the annotator's output_schema
        declares a confidence field and the annotation dict contains one.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO word_annotations"
            " (token_id, annotator, value, confidence) VALUES (?, ?, ?, ?)",
            (token_id, annotator, value, confidence),
        )

    def write_word_annotation_features(
        self,
        token_id: str,
        annotator: str,
        features: dict,
    ) -> None:
        """
        Write extra per-token fields (lemma, part_of_speech, morphology, …) for one
        token/annotator. Complements write_word_annotation, which stores only the
        canonical "label". Values are coerced to text; empty/None values are skipped.

        Requires the token row to exist first (foreign key on token_id).
        """
        rows = [
            (token_id, annotator, key, str(value))
            for key, value in features.items()
            if value is not None and str(value) != ""
        ]
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO word_annotation_features"
                " (token_id, annotator, key, value) VALUES (?, ?, ?, ?)",
                rows,
            )

    def write_sentence_annotation(
        self,
        sentence_id: str,
        annotator: str,
        value: str,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Write one sentence-level annotation. value must be the canonical "summary"
        string. confidence is optional; supply it when the annotator's output_schema
        declares a confidence field and the annotation dict contains one.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO sentence_annotations"
            " (sentence_id, annotator, value, confidence) VALUES (?, ?, ?, ?)",
            (sentence_id, annotator, value, confidence),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def chunk_exists(self, chunk_id: str) -> bool:
        """Return True if this chunk already has a row in the chunks table."""
        return self.conn.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone() is not None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def commit(self) -> None:
        """Commit the current transaction. Call after each logical batch (e.g. one chunk)."""
        self.conn.commit()

    def rollback(self) -> None:
        """Roll back the current transaction, discarding any uncommitted writes."""
        self.conn.rollback()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
