from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg2
import psycopg2.extras

from src.kg.models.relationship import Relationship
from src.kg.models.table import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Every FK in the schema: constraint name, referencing side, referenced side.
# We join referential_constraints to also get the update/delete action rules,
# which are used only for logging but could be surfaced later.
_SQL_ALL_FKS = """
SELECT
    tc.constraint_name,
    tc.table_name          AS from_table,
    kcu.column_name        AS from_column,
    ccu.table_name         AS to_table,
    ccu.column_name        AS to_column,
    rc.update_rule,
    rc.delete_rule
FROM information_schema.table_constraints      AS tc
JOIN information_schema.key_column_usage       AS kcu
  ON  tc.constraint_name = kcu.constraint_name
  AND tc.table_schema    = kcu.table_schema
  AND tc.table_name      = kcu.table_name
JOIN information_schema.constraint_column_usage AS ccu
  ON  tc.constraint_name = ccu.constraint_name
  AND tc.table_schema    = ccu.table_schema
JOIN information_schema.referential_constraints  AS rc
  ON  tc.constraint_name = rc.constraint_name
  AND tc.table_schema    = rc.constraint_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND tc.table_schema    = %s
ORDER BY tc.constraint_name, kcu.ordinal_position;
"""

# Is a given column a PRIMARY KEY in its table?
_SQL_IS_PK = """
SELECT 1
FROM information_schema.table_constraints      AS tc
JOIN information_schema.key_column_usage       AS kcu
  ON  tc.constraint_name = kcu.constraint_name
  AND tc.table_schema    = kcu.table_schema
WHERE tc.constraint_type = 'PRIMARY KEY'
  AND tc.table_schema    = %s
  AND tc.table_name      = %s
  AND kcu.column_name    = %s
LIMIT 1;
"""

# Is a given column covered by a single-column UNIQUE constraint?
_SQL_IS_UNIQUE = """
SELECT 1
FROM information_schema.table_constraints      AS tc
JOIN information_schema.key_column_usage       AS kcu
  ON  tc.constraint_name = kcu.constraint_name
  AND tc.table_schema    = kcu.table_schema
WHERE tc.constraint_type = 'UNIQUE'
  AND tc.table_schema    = %s
  AND tc.table_name      = %s
  AND kcu.column_name    = %s
LIMIT 1;
"""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class RelationshipExtractor:
    """
    Extracts every foreign-key relationship defined in the given PostgreSQL
    schema and returns a list of :class:`~src.kg.models.relationship.Relationship`
    objects, one per FK column (composite FKs produce one object per column
    pair, matching how information_schema exposes them).

    Relationship type inference
    ---------------------------
    The type is derived from the uniqueness of the referencing column:

    * ``one-to-one``  – ``from_column`` is a PRIMARY KEY or UNIQUE in
                        ``from_table`` (so at most one row on each side).
    * ``many-to-one`` – ``from_column`` is a plain FK column (the common
                        case: many child rows point to one parent row).

    Note: ``one-to-many`` is the *mirror* of many-to-one and is not stored
    as a separate edge; callers can derive it by reading relationships from
    the referenced table's perspective.

    Parameters
    ----------
    conn:
        An open ``psycopg2`` connection.  The caller owns the connection
        lifecycle (open/close/commit).
    kg_id:
        UUID of the :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`
        that owns these relationships.
    schema_name:
        PostgreSQL schema to introspect (default ``"public"``).
    """

    def __init__(
        self,
        conn: Any,
        kg_id: UUID,
        schema_name: str = "public",
    ) -> None:
        self.conn = conn
        self.kg_id = kg_id
        self.schema_name = schema_name

        # Cache PK / UNIQUE lookups so we don't re-query the same
        # (table, column) pair multiple times across many FKs.
        self._pk_cache: dict[tuple[str, str], bool] = {}
        self._unique_cache: dict[tuple[str, str], bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_relationships(
        self,
        tables: dict[str, Table],
    ) -> list[Relationship]:
        """
        Return all FK relationships found in the schema.

        Parameters
        ----------
        tables:
            Mapping of ``table_name -> Table`` produced by
            :class:`~src.kg.extractors.table_extractor.TableExtractor`.
            Used to resolve ``table_id`` UUIDs for both sides of each FK.

        Returns
        -------
        list[Relationship]
            One :class:`Relationship` per FK column pair.  Rows whose
            referencing or referenced table is not present in *tables*
            are skipped with a warning (e.g. cross-schema FKs).
        """
        raw_rows = self._fetch_raw_fk_rows()
        logger.debug("Raw FK rows fetched from information_schema: %d", len(raw_rows))

        relationships: list[Relationship] = []

        for row in raw_rows:
            from_table_name: str = row["from_table"]
            to_table_name: str = row["to_table"]
            from_col: str = row["from_column"]
            to_col: str = row["to_column"]

            from_table = tables.get(from_table_name)
            to_table = tables.get(to_table_name)

            if from_table is None or to_table is None:
                missing = from_table_name if from_table is None else to_table_name
                logger.warning(
                    "Skipping FK '%s' – table '%s' not found in the extracted "
                    "table map (possible cross-schema reference).",
                    row["constraint_name"],
                    missing,
                )
                continue

            rel_type = self._infer_relationship_type(
                from_table_name,
                from_col,
                to_table_name,
                to_col,
            )
            is_self_ref = from_table_name == to_table_name

            relationship = Relationship(
                kg_id=self.kg_id,
                from_table_id=from_table.table_id,
                to_table_id=to_table.table_id,
                from_table_name=from_table_name,
                to_table_name=to_table_name,
                from_column=from_col,
                to_column=to_col,
                relationship_type=rel_type,
                constraint_name=row["constraint_name"],
                join_condition=(
                    f"{from_table_name}.{from_col} = {to_table_name}.{to_col}"
                ),
                is_self_reference=is_self_ref,
            )
            relationships.append(relationship)

            logger.debug(
                "  FK %-40s  %s.%s -> %s.%s  [%s]%s",
                row["constraint_name"],
                from_table_name,
                from_col,
                to_table_name,
                to_col,
                rel_type,
                "  ⟲ self-ref" if is_self_ref else "",
            )

        logger.info(
            "Extracted %d relationship(s) from schema '%s'.",
            len(relationships),
            self.schema_name,
        )
        return relationships

    # ------------------------------------------------------------------
    # Private – data fetching
    # ------------------------------------------------------------------

    def _fetch_raw_fk_rows(self) -> list[dict]:
        """
        Query information_schema for every FK in the target schema.

        Returns a list of dicts with keys:
          constraint_name, from_table, from_column,
          to_table, to_column, update_rule, delete_rule
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SQL_ALL_FKS, (self.schema_name,))
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Private – relationship type inference
    # ------------------------------------------------------------------

    def _infer_relationship_type(
        self,
        from_table: str,
        from_col: str,
        to_table: str,
        to_col: str,
    ) -> str:
        """
        Determine cardinality of the FK edge.

        Logic
        -----
        1. If ``from_col`` is a PRIMARY KEY **or** is covered by a single-
           column UNIQUE constraint → ``"one-to-one"``.
        2. Otherwise → ``"many-to-one"`` (the typical FK pattern).

        The referenced side (``to_col``) is almost always a PK so we do
        not re-check it, but we accept any uniquely-constrained column.
        """
        if self._is_primary_key(from_table, from_col):
            return "one-to-one"

        if self._is_unique(from_table, from_col):
            return "one-to-one"

        return "many-to-one"

    # ------------------------------------------------------------------
    # Private – constraint look-ups (cached)
    # ------------------------------------------------------------------

    def _is_primary_key(self, table_name: str, column_name: str) -> bool:
        """Return True if *column_name* is part of the PK for *table_name*."""
        key = (table_name, column_name)
        if key not in self._pk_cache:
            with self.conn.cursor() as cur:
                cur.execute(
                    _SQL_IS_PK,
                    (self.schema_name, table_name, column_name),
                )
                self._pk_cache[key] = cur.fetchone() is not None
        return self._pk_cache[key]

    def _is_unique(self, table_name: str, column_name: str) -> bool:
        """
        Return True if *column_name* appears alone in a UNIQUE constraint
        for *table_name*.
        """
        key = (table_name, column_name)
        if key not in self._unique_cache:
            with self.conn.cursor() as cur:
                cur.execute(
                    _SQL_IS_UNIQUE,
                    (self.schema_name, table_name, column_name),
                )
                self._unique_cache[key] = cur.fetchone() is not None
        return self._unique_cache[key]
