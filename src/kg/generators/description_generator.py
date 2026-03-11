from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from src.kg.models.column import Column
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.models.relationship import Relationship
from src.kg.models.table import Table
from src.openai_client import ChatMessage, OpenAILLMClient, get_default_llm_client

logger = logging.getLogger(__name__)

_DESCRIPTION_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class TableDescriptionResponse(BaseModel):
    """Structured output returned by the LLM for a single table."""

    description: str = Field(
        description=(
            "A clear, concise 1-2 sentence description of what this table stores "
            "and its role within the database. Written for a technical audience."
        )
    )
    business_domain: str = Field(
        description=(
            "A short label for the business domain this table belongs to. "
            "Examples: 'sales', 'inventory', 'user management', 'logistics', 'billing'."
        )
    )
    typical_use_cases: List[str] = Field(
        description=(
            "A list of 3 to 5 concrete business questions or operations that this "
            "table is commonly involved in. Each item should be a short, actionable phrase."
        )
    )


class ColumnDescriptionResponse(BaseModel):
    """Structured output returned by the LLM for a single column."""

    description: str = Field(
        description=(
            "A clear, concise single sentence describing exactly what value this "
            "column stores. Be specific about the data it holds."
        )
    )
    business_meaning: str = Field(
        description=(
            "A business-friendly explanation of why this column exists and what "
            "role it plays in the context of the table and the wider system. "
            "Avoid restating the technical type; focus on the business significance."
        )
    )


# ---------------------------------------------------------------------------
# Prompt builders  (pure functions – easy to test in isolation)
# ---------------------------------------------------------------------------

_TABLE_SYSTEM_PROMPT = """\
You are an expert database documentation engineer.
Given the technical metadata of a PostgreSQL table, your job is to produce \
accurate, concise, and business-friendly documentation.

Guidelines:
- Use plain English. Avoid jargon where possible.
- Do not invent data that is not implied by the metadata.
- The description should reflect what the table *actually* stores, not a generic statement.
- The business_domain must be a short, lower-case label (1-3 words).
- Each use-case in typical_use_cases must be a short actionable phrase, not a full sentence.
"""

_COLUMN_SYSTEM_PROMPT = """\
You are an expert database documentation engineer.
Given the technical metadata of a single PostgreSQL column, your job is to produce \
a precise, business-friendly description and a clear statement of business meaning.

Guidelines:
- The description must say what value the column stores (one sentence, no more).
- The business_meaning must explain *why* the column exists in business terms.
- Do not repeat technical types or constraint names verbatim in the description.
- If sample values or enum values are provided, use them to make the description more precise.
- For PII columns, describe the concept without referencing the actual data.
"""


def _format_column_flags(col: Column) -> str:
    """Return a compact bracket-notation constraint summary for a column."""
    flags: list[str] = []
    if col.is_primary_key:
        flags.append("PK")
    if col.is_foreign_key:
        flags.append("FK")
    if col.is_unique:
        flags.append("UNIQUE")
    if not col.is_nullable:
        flags.append("NOT NULL")
    if col.is_pii:
        flags.append("PII")
    return f"  [{', '.join(flags)}]" if flags else ""


def _build_table_user_prompt(
    table: Table,
    relationships: list[Relationship],
) -> str:
    """
    Assemble the user-turn message for table description generation.

    Includes: qualified name, type, row estimate, all columns with flags,
    and all FK relationships touching this table.
    """
    lines: list[str] = [
        "Document the following PostgreSQL table.\n",
        f"Table name : {table.qualified_name}",
        f"Table type : {table.table_type}",
    ]

    if table.row_count_estimate is not None:
        lines.append(f"Row count  : ~{table.row_count_estimate:,}  (estimate)")
    else:
        lines.append("Row count  : unknown")

    # ---- columns -----------------------------------------------------------
    lines.append(f"\nColumns ({len(table.columns)}):")
    for col in sorted(table.columns.values(), key=lambda c: c.column_position or 0):
        flags = _format_column_flags(col)
        extras: list[str] = []
        if col.enum_values:
            extras.append(
                f"enum({', '.join(col.enum_values[:6])}{'...' if len(col.enum_values) > 6 else ''})"
            )
        if col.cardinality:
            extras.append(f"cardinality={col.cardinality}")
        extra_str = f"  — {'; '.join(extras)}" if extras else ""
        lines.append(f"  {col.column_name:<30} {col.data_type:<25}{flags}{extra_str}")

    # ---- relationships -----------------------------------------------------
    if relationships:
        lines.append("\nRelationships:")
        for rel in relationships:
            direction = (
                f"{rel.from_table_name}.{rel.from_column}"
                f" → {rel.to_table_name}.{rel.to_column}"
            )
            lines.append(f"  {direction:<55} ({rel.relationship_type})")
    else:
        lines.append("\nRelationships: none")

    return "\n".join(lines)


def _build_column_user_prompt(column: Column, table: Table) -> str:
    """
    Assemble the user-turn message for column description generation.

    Uses the table's already-generated description as context so the LLM
    can relate the column to the broader purpose of the table.
    """
    table_desc = table.description or f"A table named '{table.table_name}'."

    lines: list[str] = [
        "Document the following PostgreSQL column.\n",
        f"Table        : {table.qualified_name}",
        f"Table desc   : {table_desc}",
        "",
        f"Column       : {column.column_name}",
        f"Data type    : {column.data_type}",
        f"Position     : {column.column_position}",
        f"Nullable     : {'yes' if column.is_nullable else 'no'}",
    ]

    # ---- constraint flags --------------------------------------------------
    flags: list[str] = []
    if column.is_primary_key:
        flags.append("PRIMARY KEY")
    if column.is_foreign_key:
        flags.append("FOREIGN KEY")
    if column.is_unique:
        flags.append("UNIQUE")
    if flags:
        lines.append(f"Constraints  : {', '.join(flags)}")

    # ---- statistics --------------------------------------------------------
    if column.cardinality:
        lines.append(f"Cardinality  : {column.cardinality}")
    if column.null_percentage is not None:
        lines.append(f"Null %       : {column.null_percentage:.1f}%")

    # ---- enum / sample values (skipped for PII) ----------------------------
    if column.enum_values:
        lines.append(f"Enum values  : {', '.join(column.enum_values)}")
    elif column.sample_values and not column.is_pii:
        lines.append(
            f"Sample values: {', '.join(str(v) for v in column.sample_values)}"
        )
    elif column.is_pii:
        lines.append("Sample values: [redacted – PII column]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DescriptionGenerator:
    """
    Generates natural-language descriptions for every table and column in a
    :class:`~src.kg.models.knowledge_graph.KnowledgeGraph` using
    ``gpt-4o-mini``.

    Each generation call uses OpenAI's structured-output feature
    (``beta.chat.completions.parse``) to guarantee that the response
    conforms to the declared Pydantic schema.  Results are written back
    in-place to the :class:`Table` and :class:`Column` objects so the KG
    is fully enriched after a single call to :meth:`generate_all`.

    Per-item errors are caught, logged, and skipped; a single failed
    table or column never aborts the whole run.

    Parameters
    ----------
    llm_client:
        The :class:`~src.openai_client.OpenAILLMClient` to use.  Defaults
        to the shared singleton returned by
        :func:`~src.openai_client.get_default_llm_client`.
    model:
        OpenAI model identifier.  Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(
        self,
        llm_client: Optional[OpenAILLMClient] = None,
        model: str = _DESCRIPTION_MODEL,
    ) -> None:
        self.llm_client = llm_client or get_default_llm_client()
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_table_description(
        self,
        table: Table,
        relationships: Optional[list[Relationship]] = None,
    ) -> None:
        """
        Generate and write back ``description``, ``business_domain``, and
        ``typical_use_cases`` for *table*.

        The fields are mutated directly on the passed object.  If the LLM
        call fails, the fields are left unchanged (typically ``None``) and
        the error is logged.

        Parameters
        ----------
        table:
            The :class:`Table` to document.
        relationships:
            All FK relationships that touch this table (from either side).
            Pass ``None`` or an empty list when there are none.
        """
        rels = relationships or []
        logger.info("Generating description for table '%s' …", table.qualified_name)

        messages: list[ChatMessage] = [
            {"role": "system", "content": _TABLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_table_user_prompt(table, rels),
            },
        ]

        try:
            response: TableDescriptionResponse = (
                self.llm_client.generate_structured_completion(
                    messages=messages,
                    response_model=TableDescriptionResponse,
                    model=self.model,
                )
            )
            table.description = response.description
            table.business_domain = response.business_domain
            table.typical_use_cases = response.typical_use_cases

            logger.info(
                "  ✓ table '%s'  domain=%r  use_cases=%d",
                table.table_name,
                table.business_domain,
                len(table.typical_use_cases),
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  ✗ Failed to generate description for table '%s': %s",
                table.qualified_name,
                exc,
            )

    def generate_column_description(
        self,
        column: Column,
        table: Table,
    ) -> None:
        """
        Generate and write back ``description`` and ``business_meaning``
        for *column*.

        The table's ``description`` field should already be populated
        (call :meth:`generate_table_description` first) so the LLM has
        context about the table's purpose.

        The fields are mutated directly on the passed object.  Errors are
        caught, logged, and silently skipped.

        Parameters
        ----------
        column:
            The :class:`Column` to document.
        table:
            The parent :class:`Table`; its description is embedded in the
            prompt as context.
        """
        logger.debug(
            "  Generating description for column '%s.%s' …",
            table.table_name,
            column.column_name,
        )

        messages: list[ChatMessage] = [
            {"role": "system", "content": _COLUMN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_column_user_prompt(column, table),
            },
        ]

        try:
            response: ColumnDescriptionResponse = (
                self.llm_client.generate_structured_completion(
                    messages=messages,
                    response_model=ColumnDescriptionResponse,
                    model=self.model,
                )
            )
            column.description = response.description
            column.business_meaning = response.business_meaning

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  ✗ Failed to generate description for column '%s': %s",
                column.qualified_name,
                exc,
            )

    def generate_all(self, kg: KnowledgeGraph) -> None:
        """
        Run the full description-generation pipeline over every table and
        column in *kg*, mutating each object in-place.

        Processing order
        ----------------
        For each table (alphabetical order for determinism):

        1. Generate the table description — this populates
           ``table.description``, ``table.business_domain``, and
           ``table.typical_use_cases``.
        2. Generate column descriptions for every column in that table,
           in ordinal position order — the table description from step 1
           is embedded in each column prompt as context.

        The KG's ``last_updated`` timestamp is refreshed once all items
        have been processed.

        Parameters
        ----------
        kg:
            The :class:`KnowledgeGraph` to enrich.  Must already contain
            tables and columns populated by
            :class:`~src.kg.extractors.schema_extractor.SchemaExtractor`.
        """
        total_tables = len(kg.tables)
        total_columns = sum(len(t.columns) for t in kg.tables.values())

        logger.info(
            "DescriptionGenerator: enriching %d table(s), %d column(s) "
            "using model '%s'.",
            total_tables,
            total_columns,
            self.model,
        )

        for idx, (table_name, table) in enumerate(sorted(kg.tables.items()), start=1):
            logger.info(
                "[%d/%d] Table: %s",
                idx,
                total_tables,
                table.qualified_name,
            )

            # Step 1 — table description (relationships passed as context)
            relationships = kg.get_relationships_for_table(table_name)
            self.generate_table_description(table, relationships)

            # Step 2 — column descriptions (ordered by position)
            ordered_columns = sorted(
                table.columns.values(),
                key=lambda c: c.column_position or 0,
            )
            for col_idx, column in enumerate(ordered_columns, start=1):
                logger.debug(
                    "  [col %d/%d] %s",
                    col_idx,
                    len(ordered_columns),
                    column.column_name,
                )
                self.generate_column_description(column, table)

            described_cols = sum(
                1 for c in table.columns.values() if c.description is not None
            )
            logger.info(
                "  → %d/%d column(s) described.",
                described_cols,
                len(table.columns),
            )

        # Refresh the KG timestamp now that all descriptions are written
        from datetime import datetime

        kg.last_updated = datetime.now()

        described_tables = sum(
            1 for t in kg.tables.values() if t.description is not None
        )
        described_columns = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.description is not None
        )

        logger.info(
            "DescriptionGenerator complete: %d/%d tables described, "
            "%d/%d columns described.",
            described_tables,
            total_tables,
            described_columns,
            total_columns,
        )
