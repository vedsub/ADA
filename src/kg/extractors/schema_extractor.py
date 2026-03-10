from __future__ import annotations

import hashlib
import logging
from typing import Any

import psycopg2
import psycopg2.extras

from config.settings import DatabaseConfig
from src.kg.extractors.column_extractor import ColumnExtractor
from src.kg.extractors.relationship_extractor import RelationshipExtractor
from src.kg.extractors.table_extractor import TableExtractor
from src.kg.models.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class SchemaExtractor:
    """
    Orchestrates a full schema extraction pass against a PostgreSQL database
    and returns a fully-populated :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.

    Extraction pipeline
    -------------------
    1. Open a single ``psycopg2`` connection (read-only intent; no writes).
    2. **TableExtractor**      – discover all tables / views + row-count estimates.
    3. **ColumnExtractor**     – for every table, extract column metadata,
                                 constraint flags, stats, enum values, and
                                 non-PII sample values.
    4. **RelationshipExtractor** – extract every FK edge and infer its cardinality.
    5. Assemble results into a :class:`KnowledgeGraph` and set ``status = "ready"``.
    6. On any unrecoverable error set ``status = "error"`` and re-raise.

    The connection is always closed in the ``finally`` block regardless of
    success or failure, so callers do not need to manage it.

    Parameters
    ----------
    config:
        A :class:`~config.settings.DatabaseConfig` instance.  Build one from
        environment variables via ``DatabaseConfig.from_env()``, or construct
        it directly in tests.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> KnowledgeGraph:
        """
        Run the full extraction pipeline and return the completed
        :class:`KnowledgeGraph`.

        Raises
        ------
        psycopg2.OperationalError
            When the database cannot be reached.
        Exception
            Any unhandled error from an individual extractor; the KG's
            ``status`` is set to ``"error"`` before re-raising.
        """
        logger.info(
            "Starting schema extraction for %s",
            self.config.safe_repr(),
        )

        conn = self._open_connection()

        kg = KnowledgeGraph(
            source_db_host=self.config.host,
            source_db_port=self.config.port,
            source_db_name=self.config.dbname,
            source_db_hash=self._make_db_hash(),
            status="building",
        )

        try:
            self._extract_tables(conn, kg)
            self._extract_columns(conn, kg)
            self._extract_relationships(conn, kg)

            kg.status = "ready"
            logger.info(
                "Schema extraction complete — %d table(s), %d relationship(s).",
                len(kg.tables),
                len(kg.relationships),
            )

        except Exception:
            kg.status = "error"
            logger.exception("Schema extraction failed for %s", self.config.safe_repr())
            raise

        finally:
            try:
                conn.close()
                logger.debug("Database connection closed.")
            except Exception:  # noqa: BLE001
                pass

        return kg

    # ------------------------------------------------------------------
    # Private – pipeline steps
    # ------------------------------------------------------------------

    def _extract_tables(self, conn: Any, kg: KnowledgeGraph) -> None:
        """
        Step 1 – extract all tables and register them in the KG.
        """
        logger.info("Step 1/3 — Extracting tables …")

        extractor = TableExtractor(
            conn=conn,
            kg_id=kg.kg_id,
            schema_name=self.config.schema_name,
        )
        tables = extractor.extract_tables()

        for table in tables.values():
            kg.add_table(table)

        logger.info(
            "  → %d table(s) registered: %s",
            len(tables),
            sorted(tables.keys()),
        )

    def _extract_columns(self, conn: Any, kg: KnowledgeGraph) -> None:
        """
        Step 2 – for every table in the KG, extract its columns and attach
        them to the Table object in-place.
        """
        logger.info("Step 2/3 — Extracting columns for %d table(s) …", len(kg.tables))

        extractor = ColumnExtractor(
            conn=conn,
            schema_name=self.config.schema_name,
        )

        total_columns = 0
        for table_name, table in kg.tables.items():
            columns = extractor.extract_columns(
                table_name=table_name,
                table_id=table.table_id,
            )
            table.columns = columns
            total_columns += len(columns)
            logger.info(
                "  → %-40s %3d column(s)  (PK=%d  FK=%d  PII=%d)",
                f"{table_name}:",
                len(columns),
                sum(1 for c in columns.values() if c.is_primary_key),
                sum(1 for c in columns.values() if c.is_foreign_key),
                sum(1 for c in columns.values() if c.is_pii),
            )

        logger.info("  → %d column(s) extracted in total.", total_columns)

    def _extract_relationships(self, conn: Any, kg: KnowledgeGraph) -> None:
        """
        Step 3 – extract FK relationships and register them in the KG.
        """
        logger.info("Step 3/3 — Extracting relationships …")

        extractor = RelationshipExtractor(
            conn=conn,
            kg_id=kg.kg_id,
            schema_name=self.config.schema_name,
        )
        relationships = extractor.extract_relationships(tables=kg.tables)

        for rel in relationships:
            kg.add_relationship(rel)

        if relationships:
            logger.info("  → %d relationship(s):", len(relationships))
            for rel in relationships:
                logger.info(
                    "      %-30s  %s.%s → %s.%s  [%s]",
                    rel.constraint_name or "(unnamed)",
                    rel.from_table_name,
                    rel.from_column,
                    rel.to_table_name,
                    rel.to_column,
                    rel.relationship_type,
                )
        else:
            logger.info(
                "  → No foreign-key relationships found in schema '%s'.",
                self.config.schema_name,
            )

    # ------------------------------------------------------------------
    # Private – helpers
    # ------------------------------------------------------------------

    def _open_connection(self) -> Any:
        """
        Open and return a ``psycopg2`` connection using the stored config.

        ``autocommit`` is set to ``True`` because:
        - All queries are read-only; no transaction semantics are needed.
        - It prevents ``DISTINCT`` sample-value queries from blocking inside
          an implicit transaction when a per-column error triggers a rollback
          in :class:`ColumnExtractor`.
        """
        logger.debug(
            "Connecting to %s:%s/%s as %s …",
            self.config.host,
            self.config.port,
            self.config.dbname,
            self.config.user,
        )
        conn = psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.dbname,
            user=self.config.user,
            password=self.config.password,
        )
        conn.autocommit = True
        return conn

    def _make_db_hash(self) -> str:
        """
        Return a stable SHA-256 hex digest that uniquely identifies this
        database endpoint.  Used as ``KnowledgeGraph.source_db_hash`` so
        that the same physical database always maps to the same KG identity.

        The hash is computed from ``host:port/dbname`` – deliberately
        excluding the username so that credential rotation does not
        invalidate a stored KG.
        """
        raw = f"{self.config.host}:{self.config.port}/{self.config.dbname}"
        return hashlib.sha256(raw.encode()).hexdigest()
