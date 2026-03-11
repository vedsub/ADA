from __future__ import annotations

import logging
from typing import Optional

from src.kg.models.column import Column
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.models.table import Table
from src.openai_client import OpenAILLMClient, get_default_llm_client

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embedding text builders  (pure functions – easy to test in isolation)
# ---------------------------------------------------------------------------


def _build_table_embedding_text(table: Table) -> str:
    """
    Build the rich natural-language document that will be embedded for *table*.

    The document is optimised for semantic retrieval: it layers the table's
    technical identity (name, schema, type), its LLM-generated business
    context (description, domain, use-cases), and a compact column roster
    so that queries mentioning specific column concepts also surface the
    right table.

    Fields that have not yet been populated (e.g. description is still None
    because DescriptionGenerator has not run) are silently omitted so the
    method remains safe to call at any pipeline stage.
    """
    lines: list[str] = []

    # --- identity -----------------------------------------------------------
    lines.append(f"Table: {table.table_name}")
    lines.append(f"Schema: {table.schema_name}")
    lines.append(f"Qualified name: {table.qualified_name}")
    lines.append(f"Type: {table.table_type}")

    if table.row_count_estimate is not None:
        lines.append(f"Approximate row count: {table.row_count_estimate:,}")

    # --- LLM-generated context ----------------------------------------------
    if table.description:
        lines.append(f"Description: {table.description}")

    if table.business_domain:
        lines.append(f"Business domain: {table.business_domain}")

    if table.typical_use_cases:
        lines.append("Typical use cases: " + "; ".join(table.typical_use_cases))

    # --- column roster ------------------------------------------------------
    if table.columns:
        col_parts: list[str] = []
        for col in sorted(table.columns.values(), key=lambda c: c.column_position or 0):
            flags: list[str] = []
            if col.is_primary_key:
                flags.append("PK")
            if col.is_foreign_key:
                flags.append("FK")
            if col.is_unique:
                flags.append("UNIQUE")
            if not col.is_nullable:
                flags.append("NOT NULL")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            desc_str = f": {col.description}" if col.description else ""
            col_parts.append(f"{col.column_name} ({col.data_type}){flag_str}{desc_str}")

        lines.append("Columns: " + " | ".join(col_parts))

    return "\n".join(lines)


def _build_column_embedding_text(column: Column, table: Table) -> str:
    """
    Build the rich natural-language document that will be embedded for *column*.

    The document embeds the parent table's identity and description so that
    the vector captures both the column's own semantics and its role within
    the broader table context.  This is essential for queries like
    "customer email address" to surface the right column even when the column
    name alone is ambiguous (e.g. ``email`` exists in multiple tables).

    PII columns never expose sample values in the text.
    """
    lines: list[str] = []

    # --- identity -----------------------------------------------------------
    lines.append(f"Column: {column.column_name}")
    lines.append(f"Table: {table.table_name} (schema: {table.schema_name})")
    lines.append(f"Qualified name: {column.qualified_name}")
    lines.append(f"Data type: {column.data_type}")
    lines.append(f"Position: {column.column_position}")

    # --- parent table context -----------------------------------------------
    if table.description:
        lines.append(f"Table description: {table.description}")

    if table.business_domain:
        lines.append(f"Business domain: {table.business_domain}")

    # --- constraint flags ---------------------------------------------------
    flags: list[str] = []
    if column.is_primary_key:
        flags.append("PRIMARY KEY")
    if column.is_foreign_key:
        flags.append("FOREIGN KEY")
    if column.is_unique:
        flags.append("UNIQUE")
    if not column.is_nullable:
        flags.append("NOT NULL")
    if column.is_pii:
        flags.append("PII")
    if flags:
        lines.append("Constraints: " + ", ".join(flags))

    # --- statistics ---------------------------------------------------------
    if column.cardinality:
        lines.append(f"Cardinality: {column.cardinality}")

    if column.null_percentage is not None:
        lines.append(f"Null percentage: {column.null_percentage:.1f}%")

    # --- value hints (enum or sample, never both, never PII) ----------------
    if column.enum_values:
        lines.append("Enum values: " + ", ".join(column.enum_values))
    elif column.sample_values and not column.is_pii:
        lines.append(
            "Sample values: " + ", ".join(str(v) for v in column.sample_values)
        )

    # --- LLM-generated context ----------------------------------------------
    if column.description:
        lines.append(f"Description: {column.description}")

    if column.business_meaning:
        lines.append(f"Business meaning: {column.business_meaning}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class EmbeddingGenerator:
    """
    Generates and stores ``text-embedding-3-small`` vectors for every table
    and column in a :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.

    Each object's embedding is written back in-place to its ``embedding``
    field (``Table.embedding`` / ``Column.embedding``) so the KG is fully
    enriched after a single call to :meth:`generate_all`.

    Batching strategy
    -----------------
    * **Tables** – all tables are embedded in a single API call.  The
      number of tables in a typical schema is small enough that one batch
      is always safe within the API's per-request limits.
    * **Columns** – columns are batched *per table*; one API call is made
      for every table's column list.  This keeps error boundaries at the
      table level and avoids hitting per-request token limits for very
      wide schemas.

    Call order
    ----------
    :meth:`generate_all` is designed to run **after**
    :class:`~src.kg.generators.description_generator.DescriptionGenerator`
    has enriched the KG, so that embedding texts include LLM-generated
    descriptions and business context.  The generator degrades gracefully
    when descriptions are absent — it embeds whatever metadata is available.

    Per-item errors are caught, logged, and skipped; a single failed table
    or column never aborts the whole run.

    Parameters
    ----------
    llm_client:
        The :class:`~src.openai_client.OpenAILLMClient` to use.  Defaults
        to the shared singleton returned by
        :func:`~src.openai_client.get_default_llm_client`.
    model:
        OpenAI embedding model identifier.  Defaults to
        ``"text-embedding-3-small"`` (1 536-dimensional vectors).
    """

    def __init__(
        self,
        llm_client: Optional[OpenAILLMClient] = None,
        model: str = _EMBEDDING_MODEL,
    ) -> None:
        self.llm_client = llm_client or get_default_llm_client()
        self.model = model

    # ------------------------------------------------------------------
    # Public API – single-item helpers
    # ------------------------------------------------------------------

    def generate_table_embedding(self, table: Table) -> None:
        """
        Generate and write back the embedding vector for a single *table*.

        The embedding text is built from all available metadata and
        LLM-generated context fields.  The result is stored in
        ``table.embedding``.  On failure the field is left unchanged and
        the error is logged.

        Parameters
        ----------
        table:
            The :class:`Table` to embed.
        """
        text = _build_table_embedding_text(table)
        try:
            vector = self.llm_client.generate_embeddings(text, model=self.model)
            # generate_embeddings returns list[float] for a single string input
            table.embedding = vector  # type: ignore[assignment]
            logger.debug("  ✓ Embedded table '%s'", table.qualified_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  ✗ Failed to embed table '%s': %s",
                table.qualified_name,
                exc,
            )

    def generate_column_embedding(self, column: Column, table: Table) -> None:
        """
        Generate and write back the embedding vector for a single *column*.

        Parameters
        ----------
        column:
            The :class:`Column` to embed.
        table:
            The parent :class:`Table`; its metadata is woven into the
            column's embedding text for richer contextual retrieval.
        """
        text = _build_column_embedding_text(column, table)
        try:
            vector = self.llm_client.generate_embeddings(text, model=self.model)
            column.embedding = vector  # type: ignore[assignment]
            logger.debug("    ✓ Embedded column '%s'", column.qualified_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "    ✗ Failed to embed column '%s': %s",
                column.qualified_name,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API – batch helpers
    # ------------------------------------------------------------------

    def generate_table_embeddings_batch(self, tables: list[Table]) -> None:
        """
        Embed a list of tables in a single API call and write vectors back.

        Any individual table that caused an error (e.g. text too long) is
        skipped; the others are still written.

        Parameters
        ----------
        tables:
            Ordered list of :class:`Table` objects to embed.  The order
            must match the texts list so that vectors align correctly.
        """
        if not tables:
            return

        texts: list[str] = [_build_table_embedding_text(t) for t in tables]

        try:
            vectors = self.llm_client.generate_embeddings(texts, model=self.model)
            # generate_embeddings returns list[list[float]] for a sequence input
            for table, vector in zip(tables, vectors):  # type: ignore[arg-type]
                table.embedding = vector  # type: ignore[assignment]
                logger.debug("  ✓ Embedded table '%s'", table.qualified_name)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch table embedding failed (%s); falling back to per-item calls.",
                exc,
            )
            for table in tables:
                self.generate_table_embedding(table)

    def generate_column_embeddings_batch(
        self, columns: list[Column], table: Table
    ) -> None:
        """
        Embed all columns of one table in a single API call and write
        vectors back in-place.

        Falls back to per-column calls if the batch call fails.

        Parameters
        ----------
        columns:
            Ordered list of :class:`Column` objects to embed.
        table:
            The parent :class:`Table`; its context is embedded into every
            column text in the batch.
        """
        if not columns:
            return

        texts: list[str] = [_build_column_embedding_text(col, table) for col in columns]

        try:
            vectors = self.llm_client.generate_embeddings(texts, model=self.model)
            for col, vector in zip(columns, vectors):  # type: ignore[arg-type]
                col.embedding = vector  # type: ignore[assignment]
                logger.debug("    ✓ Embedded column '%s'", col.qualified_name)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch column embedding failed for table '%s' (%s); "
                "falling back to per-item calls.",
                table.table_name,
                exc,
            )
            for col in columns:
                self.generate_column_embedding(col, table)

    # ------------------------------------------------------------------
    # Public API – full KG pipeline
    # ------------------------------------------------------------------

    def generate_all(self, kg: KnowledgeGraph) -> None:
        """
        Run the full embedding pipeline over every table and column in *kg*,
        mutating each object's ``embedding`` field in-place.

        Pipeline
        --------
        1. **Tables** — build all table texts and embed them in one batch
           API call.
        2. **Columns** — for each table (alphabetical order), build all
           column texts for that table and embed them in one batch API call.

        The KG's ``last_updated`` timestamp is refreshed at the end.

        Parameters
        ----------
        kg:
            The :class:`KnowledgeGraph` to enrich.  Should already have
            ``description`` / ``business_meaning`` fields populated by
            :class:`~src.kg.generators.description_generator.DescriptionGenerator`
            for the richest possible embedding texts.
        """
        total_tables = len(kg.tables)
        total_columns = sum(len(t.columns) for t in kg.tables.values())

        logger.info(
            "EmbeddingGenerator: embedding %d table(s) and %d column(s) "
            "using model '%s'.",
            total_tables,
            total_columns,
            self.model,
        )

        # Step 1 — embed all tables in one batch call ----------------------
        logger.info("Step 1/2 — Embedding %d table(s) …", total_tables)
        all_tables = list(kg.tables.values())
        self.generate_table_embeddings_batch(all_tables)

        embedded_tables = sum(1 for t in kg.tables.values() if t.embedding is not None)
        logger.info("  → %d/%d table(s) embedded.", embedded_tables, total_tables)

        # Step 2 — embed columns per table (one batch call per table) ------
        logger.info("Step 2/2 — Embedding columns for %d table(s) …", total_tables)

        total_embedded_cols = 0
        for idx, (table_name, table) in enumerate(sorted(kg.tables.items()), start=1):
            ordered_columns = sorted(
                table.columns.values(),
                key=lambda c: c.column_position or 0,
            )
            logger.info(
                "  [%d/%d] %-40s %d column(s)",
                idx,
                total_tables,
                table_name,
                len(ordered_columns),
            )
            self.generate_column_embeddings_batch(ordered_columns, table)

            embedded_cols = sum(
                1 for c in table.columns.values() if c.embedding is not None
            )
            total_embedded_cols += embedded_cols
            logger.info(
                "    → %d/%d column(s) embedded.",
                embedded_cols,
                len(ordered_columns),
            )

        # Refresh timestamp --------------------------------------------------
        from datetime import datetime

        kg.last_updated = datetime.now()

        logger.info(
            "EmbeddingGenerator complete: %d/%d tables, %d/%d columns embedded.",
            embedded_tables,
            total_tables,
            total_embedded_cols,
            total_columns,
        )
