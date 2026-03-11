from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras

from config.settings import DatabaseConfig
from src.kg.models.column import Column
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.models.relationship import Relationship
from src.kg.models.table import Table

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# DDL  –  six tables, all idempotent
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS kg_metadata (
    kg_id           UUID        PRIMARY KEY,
    source_db_host  TEXT        NOT NULL,
    source_db_port  INTEGER     NOT NULL,
    source_db_name  TEXT        NOT NULL,
    source_db_hash  TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'building',
    created_at      TIMESTAMPTZ NOT NULL,
    last_updated    TIMESTAMPTZ NOT NULL,
    UNIQUE (source_db_hash)
);

CREATE TABLE IF NOT EXISTS kg_tables (
    table_id            UUID        PRIMARY KEY,
    kg_id               UUID        NOT NULL
                            REFERENCES kg_metadata(kg_id) ON DELETE CASCADE,
    table_name          TEXT        NOT NULL,
    schema_name         TEXT        NOT NULL DEFAULT 'public',
    qualified_name      TEXT        NOT NULL,
    table_type          TEXT        NOT NULL DEFAULT 'base_table',
    row_count_estimate  BIGINT,
    description         TEXT,
    business_domain     TEXT,
    typical_use_cases   TEXT[],
    UNIQUE (kg_id, qualified_name)
);

CREATE TABLE IF NOT EXISTS kg_columns (
    column_id        UUID        PRIMARY KEY,
    table_id         UUID        NOT NULL
                         REFERENCES kg_tables(table_id) ON DELETE CASCADE,
    column_name      TEXT        NOT NULL,
    qualified_name   TEXT        NOT NULL,
    data_type        TEXT        NOT NULL,
    is_nullable      BOOLEAN     NOT NULL DEFAULT TRUE,
    is_primary_key   BOOLEAN     NOT NULL DEFAULT FALSE,
    is_unique        BOOLEAN     NOT NULL DEFAULT FALSE,
    is_foreign_key   BOOLEAN     NOT NULL DEFAULT FALSE,
    column_position  INTEGER,
    description      TEXT,
    business_meaning TEXT,
    sample_values    TEXT[],
    enum_values      TEXT[],
    cardinality      TEXT,
    null_percentage  REAL,
    is_pii           BOOLEAN     NOT NULL DEFAULT FALSE,
    UNIQUE (table_id, column_name)
);

CREATE TABLE IF NOT EXISTS kg_relationships (
    relationship_id   UUID    PRIMARY KEY,
    kg_id             UUID    NOT NULL
                          REFERENCES kg_metadata(kg_id)  ON DELETE CASCADE,
    from_table_id     UUID    NOT NULL
                          REFERENCES kg_tables(table_id) ON DELETE CASCADE,
    to_table_id       UUID    NOT NULL
                          REFERENCES kg_tables(table_id) ON DELETE CASCADE,
    from_table_name   TEXT    NOT NULL,
    to_table_name     TEXT    NOT NULL,
    from_column       TEXT    NOT NULL,
    to_column         TEXT    NOT NULL,
    relationship_type TEXT    NOT NULL,
    constraint_name   TEXT,
    join_condition    TEXT    NOT NULL,
    business_meaning  TEXT,
    is_self_reference BOOLEAN NOT NULL DEFAULT FALSE
);

-- Embeddings live in dedicated tables so they can be updated independently
-- of the schema metadata.  ON DELETE CASCADE keeps them in sync automatically
-- if a parent table or column is removed.

CREATE TABLE IF NOT EXISTS kg_table_embeddings (
    table_id    UUID        PRIMARY KEY
                    REFERENCES kg_tables(table_id)   ON DELETE CASCADE,
    embedding   REAL[]      NOT NULL,
    model       TEXT        NOT NULL DEFAULT 'text-embedding-3-small',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS kg_column_embeddings (
    column_id   UUID        PRIMARY KEY
                    REFERENCES kg_columns(column_id) ON DELETE CASCADE,
    embedding   REAL[]      NOT NULL,
    model       TEXT        NOT NULL DEFAULT 'text-embedding-3-small',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# DML  –  upserts and queries
# ---------------------------------------------------------------------------

# ---- kg_metadata -----------------------------------------------------------

_UPSERT_KG_METADATA = """
INSERT INTO kg_metadata
    (kg_id, source_db_host, source_db_port, source_db_name,
     source_db_hash, status, created_at, last_updated)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source_db_hash) DO UPDATE SET
    source_db_host = EXCLUDED.source_db_host,
    source_db_port = EXCLUDED.source_db_port,
    source_db_name = EXCLUDED.source_db_name,
    status         = EXCLUDED.status,
    last_updated   = EXCLUDED.last_updated
    -- kg_id is intentionally NOT updated so all child FKs remain valid
RETURNING kg_id;
"""

# ---- kg_tables -------------------------------------------------------------

_UPSERT_TABLE = """
INSERT INTO kg_tables
    (table_id, kg_id, table_name, schema_name, qualified_name,
     table_type, row_count_estimate, description,
     business_domain, typical_use_cases)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (kg_id, qualified_name) DO UPDATE SET
    table_name         = EXCLUDED.table_name,
    schema_name        = EXCLUDED.schema_name,
    table_type         = EXCLUDED.table_type,
    row_count_estimate = EXCLUDED.row_count_estimate,
    description        = EXCLUDED.description,
    business_domain    = EXCLUDED.business_domain,
    typical_use_cases  = EXCLUDED.typical_use_cases
    -- table_id NOT updated → kg_table_embeddings FK stays valid
RETURNING table_id;
"""

# ---- kg_columns ------------------------------------------------------------

_UPSERT_COLUMN = """
INSERT INTO kg_columns
    (column_id, table_id, column_name, qualified_name,
     data_type, is_nullable, is_primary_key, is_unique, is_foreign_key,
     column_position, description, business_meaning,
     sample_values, enum_values, cardinality, null_percentage, is_pii)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (table_id, column_name) DO UPDATE SET
    qualified_name   = EXCLUDED.qualified_name,
    data_type        = EXCLUDED.data_type,
    is_nullable      = EXCLUDED.is_nullable,
    is_primary_key   = EXCLUDED.is_primary_key,
    is_unique        = EXCLUDED.is_unique,
    is_foreign_key   = EXCLUDED.is_foreign_key,
    column_position  = EXCLUDED.column_position,
    description      = EXCLUDED.description,
    business_meaning = EXCLUDED.business_meaning,
    sample_values    = EXCLUDED.sample_values,
    enum_values      = EXCLUDED.enum_values,
    cardinality      = EXCLUDED.cardinality,
    null_percentage  = EXCLUDED.null_percentage,
    is_pii           = EXCLUDED.is_pii
    -- column_id NOT updated → kg_column_embeddings FK stays valid
RETURNING column_id;
"""

# ---- kg_relationships ------------------------------------------------------
# Relationships have no natural stable key beyond the UUID produced by the
# extractor (which changes on every run).  We therefore DELETE all existing
# relationships for the KG and re-insert on every save.

_DELETE_RELATIONSHIPS = """
DELETE FROM kg_relationships WHERE kg_id = %s;
"""

_INSERT_RELATIONSHIP = """
INSERT INTO kg_relationships
    (relationship_id, kg_id, from_table_id, to_table_id,
     from_table_name, to_table_name, from_column, to_column,
     relationship_type, constraint_name, join_condition,
     business_meaning, is_self_reference)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# ---- embeddings ------------------------------------------------------------

_UPSERT_TABLE_EMBEDDING = """
INSERT INTO kg_table_embeddings (table_id, embedding, model, updated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (table_id) DO UPDATE SET
    embedding  = EXCLUDED.embedding,
    model      = EXCLUDED.model,
    updated_at = NOW();
"""

_UPSERT_COLUMN_EMBEDDING = """
INSERT INTO kg_column_embeddings (column_id, embedding, model, updated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (column_id) DO UPDATE SET
    embedding  = EXCLUDED.embedding,
    model      = EXCLUDED.model,
    updated_at = NOW();
"""

# ---- selects ---------------------------------------------------------------

_SELECT_KG_BY_HASH = """
SELECT kg_id, source_db_host, source_db_port, source_db_name,
       source_db_hash, status, created_at, last_updated
FROM   kg_metadata
WHERE  source_db_hash = %s;
"""

_SELECT_KG_BY_ID = """
SELECT kg_id, source_db_host, source_db_port, source_db_name,
       source_db_hash, status, created_at, last_updated
FROM   kg_metadata
WHERE  kg_id = %s;
"""

_SELECT_TABLES = """
SELECT
    t.table_id,  t.kg_id,       t.table_name,    t.schema_name,
    t.qualified_name,            t.table_type,    t.row_count_estimate,
    t.description,               t.business_domain,
    t.typical_use_cases,
    te.embedding                 AS embedding,
    te.model                     AS embedding_model
FROM      kg_tables           t
LEFT JOIN kg_table_embeddings te ON te.table_id = t.table_id
WHERE t.kg_id = %s
ORDER BY  t.table_name;
"""

_SELECT_COLUMNS = """
SELECT
    c.column_id,     c.table_id,      c.column_name,   c.qualified_name,
    c.data_type,     c.is_nullable,   c.is_primary_key, c.is_unique,
    c.is_foreign_key, c.column_position,
    c.description,   c.business_meaning,
    c.sample_values, c.enum_values,   c.cardinality,
    c.null_percentage, c.is_pii,
    ce.embedding     AS embedding,
    ce.model         AS embedding_model
FROM      kg_columns            c
JOIN      kg_tables             t  ON t.table_id  = c.table_id
LEFT JOIN kg_column_embeddings ce  ON ce.column_id = c.column_id
WHERE t.kg_id = %s
ORDER BY  c.table_id, c.column_position NULLS LAST, c.column_name;
"""

_SELECT_RELATIONSHIPS = """
SELECT
    relationship_id, kg_id,
    from_table_id,   to_table_id,
    from_table_name, to_table_name,
    from_column,     to_column,
    relationship_type, constraint_name,
    join_condition,  business_meaning, is_self_reference
FROM  kg_relationships
WHERE kg_id = %s
ORDER BY from_table_name, from_column;
"""

_EXISTS_BY_HASH = """
SELECT 1 FROM kg_metadata WHERE source_db_hash = %s LIMIT 1;
"""


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class KGRepository:
    """
    Persistence layer for the KnowledgeGraph and all its components.

    Six PostgreSQL tables are managed:

    +------------------------+--------------------------------------------------+
    | Table                  | Contents                                         |
    +========================+==================================================+
    | kg_metadata            | One row per source database (KG identity)        |
    | kg_tables              | One row per table in the KG                      |
    | kg_columns             | One row per column                               |
    | kg_relationships       | One row per FK relationship                      |
    | kg_table_embeddings    | Embedding vector for each table (separate table) |
    | kg_column_embeddings   | Embedding vector for each column                 |
    +------------------------+--------------------------------------------------+

    Embeddings are stored separately so they can be updated independently of
    the schema metadata.  A fresh extraction (which re-generates UUIDs) will
    NOT delete previously stored embeddings because the upsert logic stabilises
    UUIDs via ``ON CONFLICT … RETURNING``.

    UUID stability
    --------------
    The first time a table or column is saved its UUID (from the extractor) is
    written to the DB.  On subsequent saves the upsert hits the unique
    constraint ``(kg_id, qualified_name)`` for tables and
    ``(table_id, column_name)`` for columns, updates all mutable fields, and
    returns the *original* UUID via ``RETURNING``.  The in-memory Pydantic
    objects are updated with these stable UUIDs so that a following
    ``save_embeddings()`` call always references the correct rows.

    Usage
    -----
    Use as a context manager so the connection is always closed::

        with KGRepository(DatabaseConfig.repo_db_from_env()) as repo:
            repo.create_schema()
            repo.save_kg(kg)
            repo.save_embeddings(kg)

    Parameters
    ----------
    config:
        :class:`~config.settings.DatabaseConfig` for the *repository*
        database (not the source database being introspected).
        Build it from ``REPO_DB_*`` env variables via
        ``DatabaseConfig.repo_db_from_env()``.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self._conn: Any = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a connection to the repository database."""
        logger.debug("Connecting to repository DB %s", self.config.safe_repr())
        self._conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
        )
        # Native UUID support
        psycopg2.extras.register_uuid()
        logger.debug("Repository DB connection established.")

    def close(self) -> None:
        """Close the connection if it is open."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.debug("Repository DB connection closed.")

    def __enter__(self) -> "KGRepository":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def create_schema(self) -> None:
        """
        Create all six KG tables if they do not already exist.

        Safe to call on every startup — all statements use
        ``CREATE TABLE IF NOT EXISTS``.
        """
        self._ensure_connected()
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(_DDL)
        logger.info("KG schema created / verified.")

    # ------------------------------------------------------------------
    # Save  –  schema metadata
    # ------------------------------------------------------------------

    def save_kg(self, kg: KnowledgeGraph) -> None:
        """
        Persist the full KnowledgeGraph (metadata + tables + columns +
        relationships) in a single transaction.

        This method is idempotent.  Calling it multiple times with the same
        *kg* (e.g. after descriptions have been generated) will update all
        mutable fields without touching UUIDs or embeddings.

        UUID stabilisation
        ------------------
        Every table and column upsert uses ``ON CONFLICT … RETURNING`` to
        retrieve the *stable* UUID already stored in the DB.  The in-memory
        ``Table.table_id`` and ``Column.column_id`` are updated to these
        stable values so that a subsequent call to :meth:`save_embeddings`
        references the correct rows.

        Parameters
        ----------
        kg:
            The :class:`~src.kg.models.knowledge_graph.KnowledgeGraph` to
            persist.
        """
        self._ensure_connected()
        logger.info(
            "Saving KG for '%s' (%d tables, %d relationships) …",
            kg.source_db_name,
            len(kg.tables),
            len(kg.relationships),
        )

        with self._conn:  # auto-commit on success, rollback on exception
            with self._conn.cursor() as cur:
                stable_kg_id = self._upsert_kg_metadata(cur, kg)

                # Ensure in-memory kg_id matches the DB's stable value
                kg.kg_id = stable_kg_id

                # Tables → stabilise table_ids
                for table in kg.tables.values():
                    stable_table_id = self._upsert_table(cur, table, stable_kg_id)
                    table.table_id = stable_table_id  # pin in-memory UUID

                    # Columns → stabilise column_ids
                    for column in table.columns.values():
                        stable_col_id = self._upsert_column(
                            cur, column, stable_table_id
                        )
                        column.column_id = stable_col_id  # pin in-memory UUID

                # Relationships: delete-all + re-insert (no stable natural key)
                self._replace_relationships(cur, kg)

        logger.info(
            "KG saved: kg_id=%s  tables=%d  relationships=%d",
            kg.kg_id,
            len(kg.tables),
            len(kg.relationships),
        )

    # ------------------------------------------------------------------
    # Save  –  embeddings  (independent of schema metadata)
    # ------------------------------------------------------------------

    def save_embeddings(
        self,
        kg: KnowledgeGraph,
        model: str = _EMBEDDING_MODEL,
    ) -> None:
        """
        Persist embeddings for every table and column in *kg* that has a
        non-``None`` ``embedding`` field.

        This method is designed to run **after** :meth:`save_kg` so that
        all ``table_id`` and ``column_id`` values in the in-memory objects
        already match the stable UUIDs in the database.

        Embeddings are upserted independently — re-running this method
        after regenerating vectors updates the stored vectors without
        touching any schema-metadata rows.

        Parameters
        ----------
        kg:
            The enriched :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.
        model:
            The embedding model name to record alongside each vector.
        """
        self._ensure_connected()

        table_count = sum(1 for t in kg.tables.values() if t.embedding is not None)
        col_count = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.embedding is not None
        )
        logger.info(
            "Saving embeddings: %d table vector(s), %d column vector(s) …",
            table_count,
            col_count,
        )

        with self._conn:
            with self._conn.cursor() as cur:
                for table in kg.tables.values():
                    if table.embedding is not None:
                        self._upsert_table_embedding(cur, table, model)

                    for column in table.columns.values():
                        if column.embedding is not None:
                            self._upsert_column_embedding(cur, column, model)

        logger.info(
            "Embeddings saved: %d table(s), %d column(s).",
            table_count,
            col_count,
        )

    def save_table_embedding(
        self,
        table: Table,
        model: str = _EMBEDDING_MODEL,
    ) -> None:
        """
        Upsert the embedding for a single *table*.

        Useful for refreshing one table's vector without re-running the
        full pipeline.
        """
        if table.embedding is None:
            logger.warning(
                "save_table_embedding: Table '%s' has no embedding.", table.table_name
            )
            return
        self._ensure_connected()
        with self._conn:
            with self._conn.cursor() as cur:
                self._upsert_table_embedding(cur, table, model)
        logger.debug("Saved embedding for table '%s'.", table.table_name)

    def save_column_embedding(
        self,
        column: Column,
        model: str = _EMBEDDING_MODEL,
    ) -> None:
        """
        Upsert the embedding for a single *column*.

        Useful for refreshing one column's vector without re-running the
        full pipeline.
        """
        if column.embedding is None:
            logger.warning(
                "save_column_embedding: Column '%s' has no embedding.",
                column.qualified_name,
            )
            return
        self._ensure_connected()
        with self._conn:
            with self._conn.cursor() as cur:
                self._upsert_column_embedding(cur, column, model)
        logger.debug("Saved embedding for column '%s'.", column.qualified_name)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_kg(self, source_db_hash: str) -> KnowledgeGraph | None:
        """
        Load a full :class:`KnowledgeGraph` (including embeddings) by the
        stable ``source_db_hash`` of the source database.

        Returns ``None`` if no KG has been saved for that hash yet.

        Parameters
        ----------
        source_db_hash:
            SHA-256 hex digest of ``host:port/dbname`` produced by
            :class:`~src.kg.extractors.schema_extractor.SchemaExtractor`.
        """
        self._ensure_connected()
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_KG_BY_HASH, (source_db_hash,))
            row = cur.fetchone()

        if row is None:
            logger.debug("No KG found for hash '%s'.", source_db_hash)
            return None

        return self._load_kg_from_meta_row(dict(row))

    def load_kg_by_id(self, kg_id: UUID) -> KnowledgeGraph | None:
        """
        Load a full :class:`KnowledgeGraph` (including embeddings) by its
        ``kg_id`` UUID.

        Returns ``None`` if not found.
        """
        self._ensure_connected()
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_KG_BY_ID, (kg_id,))
            row = cur.fetchone()

        if row is None:
            logger.debug("No KG found for kg_id=%s.", kg_id)
            return None

        return self._load_kg_from_meta_row(dict(row))

    def exists(self, source_db_hash: str) -> bool:
        """
        Return ``True`` if a KG has already been saved for the given
        ``source_db_hash``.

        Use this to skip re-extraction when the KG is already up-to-date::

            if not repo.exists(hash):
                kg = SchemaExtractor(source_config).extract()
                repo.save_kg(kg)
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(_EXISTS_BY_HASH, (source_db_hash,))
            return cur.fetchone() is not None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_kg(self, kg_id: UUID) -> None:
        """
        Delete a KG and all its associated rows from every table.

        The ``ON DELETE CASCADE`` constraints ensure that tables, columns,
        relationships, and embeddings are all removed automatically when
        the ``kg_metadata`` row is deleted.

        Parameters
        ----------
        kg_id:
            The UUID of the KG to delete.
        """
        self._ensure_connected()
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM kg_metadata WHERE kg_id = %s;",
                    (kg_id,),
                )
        logger.info("Deleted KG kg_id=%s.", kg_id)

    # ------------------------------------------------------------------
    # Private – upsert helpers
    # ------------------------------------------------------------------

    def _upsert_kg_metadata(
        self,
        cur: Any,
        kg: KnowledgeGraph,
    ) -> UUID:
        """
        Upsert the kg_metadata row and return the stable ``kg_id``.

        If a row already exists for ``source_db_hash`` the existing
        ``kg_id`` is returned (NOT the one from the Pydantic object),
        ensuring all child-table FKs remain valid across re-extractions.
        """
        cur.execute(
            _UPSERT_KG_METADATA,
            (
                kg.kg_id,
                kg.source_db_host,
                kg.source_db_port,
                kg.source_db_name,
                kg.source_db_hash,
                kg.status,
                kg.created_at,
                kg.last_updated,
            ),
        )
        row = cur.fetchone()
        stable_kg_id: UUID = row[0]
        logger.debug("  kg_metadata upserted  kg_id=%s", stable_kg_id)
        return stable_kg_id

    def _upsert_table(
        self,
        cur: Any,
        table: Table,
        stable_kg_id: UUID,
    ) -> UUID:
        """
        Upsert a kg_tables row and return the stable ``table_id``.

        On conflict the existing ``table_id`` is returned so that child
        column rows and embeddings continue to reference the right UUID.
        """
        cur.execute(
            _UPSERT_TABLE,
            (
                table.table_id,
                stable_kg_id,
                table.table_name,
                table.schema_name,
                table.qualified_name,
                table.table_type,
                table.row_count_estimate,
                table.description,
                table.business_domain,
                table.typical_use_cases,
            ),
        )
        row = cur.fetchone()
        stable_table_id: UUID = row[0]
        logger.debug("    table '%s'  table_id=%s", table.table_name, stable_table_id)
        return stable_table_id

    def _upsert_column(
        self,
        cur: Any,
        column: Column,
        stable_table_id: UUID,
    ) -> UUID:
        """
        Upsert a kg_columns row and return the stable ``column_id``.

        Uses *stable_table_id* (returned by :meth:`_upsert_table`) rather
        than ``column.table_id`` so that the FK is always valid even when
        the extractor produced a fresh UUID for the parent table.
        """
        cur.execute(
            _UPSERT_COLUMN,
            (
                column.column_id,
                stable_table_id,
                column.column_name,
                column.qualified_name,
                column.data_type,
                column.is_nullable,
                column.is_primary_key,
                column.is_unique,
                column.is_foreign_key,
                column.column_position,
                column.description,
                column.business_meaning,
                column.sample_values,
                column.enum_values,
                column.cardinality,
                column.null_percentage,
                column.is_pii,
            ),
        )
        row = cur.fetchone()
        stable_col_id: UUID = row[0]
        logger.debug(
            "      column '%s'  column_id=%s", column.column_name, stable_col_id
        )
        return stable_col_id

    def _replace_relationships(self, cur: Any, kg: KnowledgeGraph) -> None:
        """
        Delete all existing relationships for this KG and insert the
        current set.

        Relationships reference ``table_id`` values that were stabilised by
        earlier upserts, so the FKs are always correct at this point.
        """
        cur.execute(_DELETE_RELATIONSHIPS, (kg.kg_id,))

        for rel in kg.relationships:
            cur.execute(
                _INSERT_RELATIONSHIP,
                (
                    rel.relationship_id,
                    kg.kg_id,
                    rel.from_table_id,
                    rel.to_table_id,
                    rel.from_table_name,
                    rel.to_table_name,
                    rel.from_column,
                    rel.to_column,
                    rel.relationship_type,
                    rel.constraint_name,
                    rel.join_condition,
                    rel.business_meaning,
                    rel.is_self_reference,
                ),
            )
        logger.debug(
            "    relationships replaced: %d row(s) inserted.", len(kg.relationships)
        )

    def _upsert_table_embedding(
        self,
        cur: Any,
        table: Table,
        model: str,
    ) -> None:
        cur.execute(
            _UPSERT_TABLE_EMBEDDING,
            (table.table_id, table.embedding, model),
        )
        logger.debug("      embedding saved for table '%s'.", table.table_name)

    def _upsert_column_embedding(
        self,
        cur: Any,
        column: Column,
        model: str,
    ) -> None:
        cur.execute(
            _UPSERT_COLUMN_EMBEDDING,
            (column.column_id, column.embedding, model),
        )
        logger.debug("      embedding saved for column '%s'.", column.qualified_name)

    # ------------------------------------------------------------------
    # Private – load helpers
    # ------------------------------------------------------------------

    def _load_kg_from_meta_row(self, meta: dict) -> KnowledgeGraph:
        """
        Given a kg_metadata dict, fetch all child rows and assemble a
        fully-populated :class:`KnowledgeGraph`.
        """
        kg_id: UUID = meta["kg_id"]
        logger.info("Loading KG kg_id=%s …", kg_id)

        kg = KnowledgeGraph(
            kg_id=kg_id,
            source_db_host=meta["source_db_host"],
            source_db_port=meta["source_db_port"],
            source_db_name=meta["source_db_name"],
            source_db_hash=meta["source_db_hash"],
            status=meta["status"],
            created_at=meta["created_at"],
            last_updated=meta["last_updated"],
        )

        # ---- tables (with embeddings via LEFT JOIN) ----------------------
        tables_by_id: dict[UUID, Table] = {}

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_TABLES, (kg_id,))
            table_rows = cur.fetchall()

        for row in table_rows:
            row = dict(row)
            table = Table(
                table_id=row["table_id"],
                kg_id=kg_id,
                table_name=row["table_name"],
                schema_name=row["schema_name"],
                qualified_name=row["qualified_name"],
                table_type=row["table_type"],
                row_count_estimate=row["row_count_estimate"],
                description=row["description"],
                business_domain=row["business_domain"],
                typical_use_cases=list(row["typical_use_cases"])
                if row["typical_use_cases"]
                else None,
                embedding=list(row["embedding"]) if row["embedding"] else None,
            )
            tables_by_id[table.table_id] = table
            kg.add_table(table)

        logger.debug("  Loaded %d table(s).", len(tables_by_id))

        # ---- columns (with embeddings via LEFT JOIN) ---------------------
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_COLUMNS, (kg_id,))
            column_rows = cur.fetchall()

        for row in column_rows:
            row = dict(row)
            column = Column(
                column_id=row["column_id"],
                table_id=row["table_id"],
                column_name=row["column_name"],
                qualified_name=row["qualified_name"],
                data_type=row["data_type"],
                is_nullable=row["is_nullable"],
                is_primary_key=row["is_primary_key"],
                is_unique=row["is_unique"],
                is_foreign_key=row["is_foreign_key"],
                column_position=row["column_position"],
                description=row["description"],
                business_meaning=row["business_meaning"],
                sample_values=list(row["sample_values"])
                if row["sample_values"]
                else None,
                enum_values=list(row["enum_values"]) if row["enum_values"] else None,
                cardinality=row["cardinality"],
                null_percentage=row["null_percentage"],
                is_pii=row["is_pii"],
                embedding=list(row["embedding"]) if row["embedding"] else None,
            )
            # Attach to parent table
            parent_table = tables_by_id.get(column.table_id)
            if parent_table is not None:
                parent_table.columns[column.column_name] = column
            else:
                logger.warning(
                    "  Column '%s' references unknown table_id=%s — skipped.",
                    column.qualified_name,
                    column.table_id,
                )

        total_cols = sum(len(t.columns) for t in kg.tables.values())
        logger.debug("  Loaded %d column(s) across all tables.", total_cols)

        # ---- relationships ----------------------------------------------
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SELECT_RELATIONSHIPS, (kg_id,))
            rel_rows = cur.fetchall()

        for row in rel_rows:
            row = dict(row)
            relationship = Relationship(
                relationship_id=row["relationship_id"],
                kg_id=kg_id,
                from_table_id=row["from_table_id"],
                to_table_id=row["to_table_id"],
                from_table_name=row["from_table_name"],
                to_table_name=row["to_table_name"],
                from_column=row["from_column"],
                to_column=row["to_column"],
                relationship_type=row["relationship_type"],
                constraint_name=row["constraint_name"],
                join_condition=row["join_condition"],
                business_meaning=row["business_meaning"],
                is_self_reference=row["is_self_reference"],
            )
            kg.add_relationship(relationship)

        logger.info(
            "KG loaded: kg_id=%s  tables=%d  columns=%d  relationships=%d",
            kg_id,
            len(kg.tables),
            total_cols,
            len(kg.relationships),
        )
        return kg

    # ------------------------------------------------------------------
    # Private – connection guard
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise a clear error if :meth:`connect` has not been called."""
        if self._conn is None or self._conn.closed:
            raise RuntimeError(
                "KGRepository is not connected. "
                "Call connect() or use it as a context manager."
            )
