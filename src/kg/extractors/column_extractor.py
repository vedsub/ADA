from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

import psycopg2.extras
from psycopg2 import sql

from src.kg.models.column import Column

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII heuristics
# ---------------------------------------------------------------------------

_PII_KEYWORDS: frozenset[str] = frozenset(
    {
        "email",
        "phone",
        "mobile",
        "ssn",
        "social_security",
        "passport",
        "address",
        "ip_address",
        "ip",
        "credit_card",
        "card_number",
        "dob",
        "date_of_birth",
        "birth_date",
        "birthdate",
        "salary",
        "income",
        "tax_id",
        "national_id",
        "license",
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "first_name",
        "last_name",
        "full_name",
        "gender",
        "race",
        "ethnicity",
        "religion",
        "biometric",
    }
)

# ---------------------------------------------------------------------------
# Cardinality thresholds  (applied to the resolved distinct-value count)
# ---------------------------------------------------------------------------

_CARDINALITY_LOW_MAX = 10
_CARDINALITY_MED_MAX = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_pii(column_name: str) -> bool:
    """Return True when any PII keyword appears as a whole token in the column name."""
    parts = set(column_name.lower().replace("-", "_").split("_"))
    return bool(parts & _PII_KEYWORDS)


def _classify_cardinality(n_distinct: float, row_count: int) -> str:
    """
    Convert pg_stats.n_distinct to a low/medium/high label.

    PostgreSQL convention
    --------------------
    n_distinct > 0  →  exact count of distinct values
    n_distinct < 0  →  fraction of total rows  (e.g. -1.0 means all unique)
    n_distinct == 0 →  not enough data (treat as unknown → return "low")
    """
    if n_distinct == 0 or row_count <= 0:
        return "low"

    count = n_distinct if n_distinct > 0 else abs(n_distinct) * row_count

    if count <= _CARDINALITY_LOW_MAX:
        return "low"
    if count <= _CARDINALITY_MED_MAX:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class ColumnExtractor:
    """
    Extracts column-level metadata for a single table.

    Each public method is self-contained and issues its own SQL so that the
    class can also be used à la carte (e.g. refresh stats for one table only).

    Parameters
    ----------
    conn        An open psycopg2 connection to the target database.
    schema_name The Postgres schema to introspect (default: ``"public"``).
    """

    def __init__(self, conn: Any, schema_name: str = "public") -> None:
        self.conn = conn
        self.schema_name = schema_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_columns(self, table_name: str, table_id: UUID) -> dict[str, Column]:
        """
        Build and return a ``{column_name: Column}`` mapping for *table_name*.

        Extraction order
        ----------------
        1. Raw column info (information_schema.columns)
        2. Primary-key, unique, and foreign-key flags  (constraint tables)
        3. Per-column statistics from pg_stats          (may be empty before ANALYZE)
        4. Enum values for USER-DEFINED typed columns   (pg_enum / pg_type)
        5. Sample values                                (direct DISTINCT query, skipped for PII)
        """
        logger.debug(
            "Extracting columns for table '%s.%s'", self.schema_name, table_name
        )

        raw_rows = self._fetch_raw_column_info(table_name)
        if not raw_rows:
            logger.warning(
                "No columns found for '%s.%s' – table may not exist in this schema.",
                self.schema_name,
                table_name,
            )
            return {}

        pk_cols = self._fetch_primary_key_columns(table_name)
        unique_cols = self._fetch_unique_columns(table_name)
        fk_cols = self._fetch_foreign_key_columns(table_name)
        stats_map = self._fetch_column_stats(table_name)
        row_count = self._fetch_row_count(table_name) or 0

        # Enum values keyed by udt_name (the underlying type name)
        udt_names: list[str] = [
            r["udt_name"] for r in raw_rows if r["data_type"] == "USER-DEFINED"
        ]
        enum_values_map = self._fetch_enum_values(udt_names)

        # Determine which columns are safe to sample (skip PII)
        safe_cols = [
            r["column_name"] for r in raw_rows if not _is_pii(r["column_name"])
        ]
        sample_map = self._fetch_sample_values(table_name, safe_cols)

        columns: dict[str, Column] = {}
        for row in raw_rows:
            col_name: str = row["column_name"]
            stat = stats_map.get(col_name, {})

            # --- cardinality -------------------------------------------------
            cardinality: str | None = None
            n_distinct = stat.get("n_distinct")
            if n_distinct is not None:
                cardinality = _classify_cardinality(float(n_distinct), row_count)

            # --- null percentage ---------------------------------------------
            null_percentage: float | None = None
            null_frac = stat.get("null_frac")
            if null_frac is not None:
                null_percentage = round(float(null_frac) * 100, 2)

            # --- enum values -------------------------------------------------
            enum_vals: list[str] | None = None
            if row["data_type"] == "USER-DEFINED":
                enum_vals = enum_values_map.get(row["udt_name"])

            # --- sample values (omitted for PII columns) ---------------------
            is_pii = _is_pii(col_name)
            sample_vals: list[str] | None = None if is_pii else sample_map.get(col_name)

            col = Column(
                table_id=table_id,
                column_name=col_name,
                qualified_name=f"{table_name}.{col_name}",
                data_type=row["data_type"],
                is_nullable=(row["is_nullable"] == "YES"),
                is_primary_key=(col_name in pk_cols),
                is_unique=(col_name in unique_cols),
                is_foreign_key=(col_name in fk_cols),
                column_position=row["ordinal_position"],
                sample_values=sample_vals,
                enum_values=enum_vals,
                cardinality=cardinality,
                null_percentage=null_percentage,
                is_pii=is_pii,
            )
            columns[col_name] = col

        logger.debug(
            "  %s.%s → %d columns (%d PK, %d FK)",
            self.schema_name,
            table_name,
            len(columns),
            sum(1 for c in columns.values() if c.is_primary_key),
            sum(1 for c in columns.values() if c.is_foreign_key),
        )
        return columns

    # ------------------------------------------------------------------
    # Private – raw column info
    # ------------------------------------------------------------------

    def _fetch_raw_column_info(self, table_name: str) -> list[dict]:
        """
        Pull core column attributes from information_schema.columns.

        ``udt_name`` is included so we can resolve enum types via pg_type later.
        """
        query = """
            SELECT
                column_name,
                ordinal_position,
                data_type,
                udt_name,
                is_nullable,
                character_maximum_length,
                numeric_precision,
                numeric_scale,
                column_default
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name   = %s
            ORDER BY ordinal_position
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (self.schema_name, table_name))
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Private – constraint flags
    # ------------------------------------------------------------------

    def _fetch_primary_key_columns(self, table_name: str) -> set[str]:
        """Return the set of column names that participate in the PK."""
        return self._fetch_constrained_columns(table_name, "PRIMARY KEY")

    def _fetch_unique_columns(self, table_name: str) -> set[str]:
        """Return the set of column names covered by any UNIQUE constraint."""
        return self._fetch_constrained_columns(table_name, "UNIQUE")

    def _fetch_foreign_key_columns(self, table_name: str) -> set[str]:
        """Return the set of column names that are FK participants."""
        return self._fetch_constrained_columns(table_name, "FOREIGN KEY")

    def _fetch_constrained_columns(
        self, table_name: str, constraint_type: str
    ) -> set[str]:
        query = """
            SELECT kcu.column_name
            FROM information_schema.table_constraints  tc
            JOIN information_schema.key_column_usage   kcu
              ON  tc.constraint_name = kcu.constraint_name
             AND  tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = %s
              AND tc.table_schema    = %s
              AND tc.table_name      = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (constraint_type, self.schema_name, table_name))
            return {row[0] for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Private – pg_stats
    # ------------------------------------------------------------------

    def _fetch_column_stats(self, table_name: str) -> dict[str, dict]:
        """
        Return per-column statistics from pg_stats.

        The dict is keyed by ``attname`` (column name) and each value
        contains ``n_distinct`` and ``null_frac``.

        Note: this table is populated by ANALYZE / autovacuum; it may be
        empty for freshly-created or very small tables.
        """
        query = """
            SELECT
                attname,
                n_distinct,
                null_frac
            FROM pg_stats
            WHERE schemaname = %s
              AND tablename  = %s
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (self.schema_name, table_name))
            return {row["attname"]: dict(row) for row in cur.fetchall()}

    # ------------------------------------------------------------------
    # Private – row count estimate
    # ------------------------------------------------------------------

    def _fetch_row_count(self, table_name: str) -> int | None:
        """
        Return the fast row-count estimate from pg_class.reltuples.

        This is updated by ANALYZE / autovacuum and may not reflect recent
        bulk inserts. Returns None when the table is not found in pg_class.
        """
        query = """
            SELECT c.reltuples::bigint
            FROM   pg_class     c
            JOIN   pg_namespace n ON n.oid = c.relnamespace
            WHERE  n.nspname = %s
              AND  c.relname  = %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (self.schema_name, table_name))
            row = cur.fetchone()
            return int(row[0]) if row else None

    # ------------------------------------------------------------------
    # Private – enum values
    # ------------------------------------------------------------------

    def _fetch_enum_values(self, udt_names: list[str]) -> dict[str, list[str]]:
        """
        Resolve enum labels for every USER-DEFINED type name in *udt_names*.

        Returns a ``{udt_name: [label, ...]}`` mapping ordered by pg_enum.enumsortorder.
        """
        if not udt_names:
            return {}

        query = """
            SELECT
                t.typname          AS udt_name,
                e.enumlabel        AS label
            FROM pg_type      t
            JOIN pg_enum      e  ON e.enumtypid  = t.oid
            JOIN pg_namespace n  ON n.oid         = t.typnamespace
            WHERE n.nspname  = %s
              AND t.typname  = ANY(%s)
            ORDER BY t.typname, e.enumsortorder
        """
        result: dict[str, list[str]] = {}
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (self.schema_name, list(udt_names)))
            for row in cur.fetchall():
                result.setdefault(row["udt_name"], []).append(row["label"])
        return result

    # ------------------------------------------------------------------
    # Private – sample values
    # ------------------------------------------------------------------

    def _fetch_sample_values(
        self,
        table_name: str,
        column_names: list[str],
        limit: int = 5,
    ) -> dict[str, list[str]]:
        """
        Fetch up to *limit* distinct, non-null sample values for each column.

        Identifiers are quoted with ``psycopg2.sql`` to prevent SQL injection.
        Any per-column error (e.g. unsupported cast) is caught and the column
        receives an empty list; the connection is rolled back so subsequent
        columns succeed.
        """
        result: dict[str, list[str]] = {}

        for col_name in column_names:
            query = sql.SQL(
                "SELECT DISTINCT {col} FROM {schema}.{table} "
                "WHERE {col} IS NOT NULL LIMIT {limit}"
            ).format(
                col=sql.Identifier(col_name),
                schema=sql.Identifier(self.schema_name),
                table=sql.Identifier(table_name),
                limit=sql.Literal(limit),
            )

            try:
                with self.conn.cursor() as cur:
                    cur.execute(query)
                    result[col_name] = [str(row[0]) for row in cur.fetchall()]
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Could not fetch sample values for '%s.%s': %s",
                    table_name,
                    col_name,
                    exc,
                )
                self.conn.rollback()
                result[col_name] = []

        return result
