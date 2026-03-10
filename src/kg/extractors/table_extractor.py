from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg2.extras

from src.kg.models.table import Table

logger = logging.getLogger(__name__)

# Normalize information_schema table_type values to the model's convention
_TABLE_TYPE_MAP: dict[str, str] = {
    "BASE TABLE": "base_table",
    "VIEW": "view",
    "FOREIGN": "foreign",
    "LOCAL TEMPORARY": "local_temporary",
}


class TableExtractor:
    """
    Extracts table-level metadata for every user table (and view) that lives
    inside a given PostgreSQL schema.

    What is pulled
    --------------
    - Table name and normalized table type  (information_schema.tables)
    - Estimated row count                   (pg_class + pg_namespace)

    The columns dict on each Table is left empty here; it is populated
    separately by ColumnExtractor so that each extractor stays focused on
    its own concern.

    Parameters
    ----------
    conn        An open psycopg2 connection to the target database.
    kg_id       UUID of the KnowledgeGraph these tables belong to.
    schema_name PostgreSQL schema to introspect (default: "public").
    """

    def __init__(self, conn: Any, kg_id: UUID, schema_name: str = "public") -> None:
        self.conn = conn
        self.kg_id = kg_id
        self.schema_name = schema_name

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract_tables(self) -> dict[str, Table]:
        """
        Return a mapping of ``table_name -> Table`` for every table / view
        found in the configured schema.
        """
        logger.debug("Fetching table list for schema %r", self.schema_name)
        raw_tables = self._fetch_raw_table_info()

        logger.debug("Fetching row-count estimates for schema %r", self.schema_name)
        row_counts = self._fetch_row_counts()

        tables: dict[str, Table] = {}

        for row in raw_tables:
            table_name: str = row["table_name"] or ""
            raw_type: str = row["table_type"] or "BASE TABLE"
            table_type: str = _TABLE_TYPE_MAP.get(
                raw_type, raw_type.lower().replace(" ", "_")
            )
            row_count = row_counts.get(table_name)

            table = Table(
                kg_id=self.kg_id,
                table_name=table_name,
                schema_name=self.schema_name,
                qualified_name=f"{self.schema_name}.{table_name}",
                table_type=table_type,
                row_count_estimate=row_count,
            )
            tables[table_name] = table
            logger.debug(
                "  Registered table %r (type=%r, ~%s rows)",
                table_name,
                table_type,
                row_count if row_count is not None else "unknown",
            )

        logger.info(
            "TableExtractor: found %d table(s) in schema %r",
            len(tables),
            self.schema_name,
        )
        return tables

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_raw_table_info(self) -> list[dict]:
        """
        Query information_schema.tables for every BASE TABLE and VIEW in the
        target schema, ordered by name for deterministic output.
        """
        sql = """
            SELECT
                table_name,
                table_type
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type   IN ('BASE TABLE', 'VIEW')
            ORDER BY table_name
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (self.schema_name,))
            return [dict(row) for row in cur.fetchall()]

    def _fetch_row_counts(self) -> dict[str, int]:
        """
        Return a mapping of table_name -> estimated row count by reading the
        pg_class tuple-count statistics (reltuples).

        These numbers are estimates maintained by VACUUM / ANALYZE, so they
        may be 0 for brand-new tables that have never been analyzed, and
        slightly off for frequently updated tables.  They are accurate enough
        for schema metadata purposes and cost nothing at query time.
        """
        sql = """
            SELECT
                c.relname            AS table_name,
                c.reltuples::bigint  AS row_count
            FROM pg_class       c
            JOIN pg_namespace   n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relkind  IN ('r', 'v', 'p')   -- regular, view, partitioned
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (self.schema_name,))
            return {
                row["table_name"]: int(row["row_count"])
                for row in cur.fetchall()
                if row["row_count"] is not None
            }
