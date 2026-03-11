#!/usr/bin/env python
"""
example_use_kg.py — Demonstrates loading and querying the Knowledge Graph.

This script shows how to:
    1. Connect to the repository and load an existing KG
    2. Inspect tables, columns, and relationships in-memory
    3. Run semantic table search  ("find tables relevant to X")
    4. Run semantic column search ("find columns relevant to X")
    5. Run a scoped column search filtered to one table

Usage
-----
    python scripts/example_use_kg.py
    python scripts/example_use_kg.py --source-prefix DB --repo-prefix REPO_DB
    python scripts/example_use_kg.py --query "customer purchase history"
    python scripts/example_use_kg.py --query "revenue totals" --top-k 3

Required environment variables
-------------------------------
Repository database  (prefix REPO_DB by default):
    REPO_DB_NAME      Name of the repository database
    REPO_DB_USER      Database username
    REPO_DB_PASSWORD  Database password
    REPO_DB_HOST      Host (default: localhost)
    REPO_DB_PORT      Port (default: 5432)

Source database  (prefix DB by default — used only to compute the hash):
    DB_NAME, DB_USER, DB_PASSWORD
    DB_HOST (default: localhost), DB_PORT (default: 5432)

Optional:
    CHROMA_PERSIST_DIR   Path for Chroma persistent storage (default: ./chroma_db)
    OPENAI_API_KEY       Required for semantic search (query embedding)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when this script is run directly.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import ChromaConfig, DatabaseConfig  # noqa: E402
from src.kg.builders.kg_builder import _compute_source_hash  # noqa: E402
from src.kg.manager.kg_manager import KGManager  # noqa: E402
from src.kg.models.knowledge_graph import KnowledgeGraph  # noqa: E402
from src.kg.storage.vector_store import (  # noqa: E402
    ColumnSearchResult,
    TableSearchResult,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="example_use_kg",
        description=textwrap.dedent(
            """\
            Load an existing Knowledge Graph from the repository and demonstrate
            in-memory inspection and semantic search capabilities.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        default="show me tables related to customers and orders",
        metavar="TEXT",
        help="Natural-language query to use for semantic search (default: customer/order example).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return per search (default: 5).",
    )
    parser.add_argument(
        "--source-prefix",
        default="DB",
        metavar="PREFIX",
        help="Env var prefix for the source database used to compute the KG hash (default: DB).",
    )
    parser.add_argument(
        "--repo-prefix",
        default="REPO_DB",
        metavar="PREFIX",
        help="Env var prefix for the repository database (default: REPO_DB).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — keeps output clean).",
    )
    return parser


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_SEP_WIDE = "═" * 68
_SEP_THIN = "─" * 68
_SEP_MID = "─" * 50


def _header(title: str) -> None:
    print()
    print(_SEP_WIDE)
    print(f"  {title}")
    print(_SEP_WIDE)


def _section(title: str) -> None:
    print()
    print(f"  ── {title}")
    print("  " + _SEP_THIN)


def _print_kg_overview(kg: KnowledgeGraph) -> None:
    """Print a top-level summary of the loaded KG."""
    _header("Knowledge Graph Overview")
    total_cols = sum(len(t.columns) for t in kg.tables.values())
    emb_tables = sum(1 for t in kg.tables.values() if t.embedding is not None)
    emb_cols = sum(
        1
        for t in kg.tables.values()
        for c in t.columns.values()
        if c.embedding is not None
    )
    print(f"  KG ID          : {kg.kg_id}")
    print(
        f"  Source DB      : {kg.source_db_name}  ({kg.source_db_host}:{kg.source_db_port})"
    )
    print(f"  Status         : {kg.status}")
    print(f"  Last updated   : {kg.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Tables         : {len(kg.tables)}")
    print(f"  Columns        : {total_cols}")
    print(f"  Relationships  : {len(kg.relationships)}")
    print(f"  Embedded tables: {emb_tables} / {len(kg.tables)}")
    print(f"  Embedded cols  : {emb_cols} / {total_cols}")


def _print_tables(kg: KnowledgeGraph) -> None:
    """Print each table with its columns and metadata."""
    _header("Tables & Columns")

    for table_name, table in sorted(kg.tables.items()):
        rows_str = (
            f"~{table.row_count_estimate:,} rows"
            if table.row_count_estimate
            else "row count unknown"
        )
        domain = f"  [{table.business_domain}]" if table.business_domain else ""
        print(f"\n  📋 {table.qualified_name}  ({rows_str}){domain}")

        if table.description:
            wrapped = textwrap.fill(
                table.description,
                width=62,
                initial_indent="     ",
                subsequent_indent="     ",
            )
            print(wrapped)

        if table.typical_use_cases:
            print(f"     Use cases: {' | '.join(table.typical_use_cases[:3])}")

        print(f"     {'Column':<28} {'Type':<20} {'Flags'}")
        print("     " + "─" * 60)

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
            if col.is_pii:
                flags.append("PII")
            flag_str = ", ".join(flags) if flags else "—"
            print(f"     {col.column_name:<28} {col.data_type:<20} {flag_str}")

            if col.description:
                wrapped = textwrap.fill(
                    col.description,
                    width=60,
                    initial_indent="       ↳ ",
                    subsequent_indent="         ",
                )
                print(wrapped)


def _print_relationships(kg: KnowledgeGraph) -> None:
    """Print all FK relationships."""
    _header("Relationships (Foreign Keys)")

    if not kg.relationships:
        print("  No relationships found in this KG.")
        return

    # Group by from_table for readability
    by_table: dict[str, list] = {}
    for rel in sorted(
        kg.relationships, key=lambda r: (r.from_table_name, r.from_column)
    ):
        by_table.setdefault(rel.from_table_name, []).append(rel)

    for from_table, rels in sorted(by_table.items()):
        print(f"\n  {from_table}")
        for rel in rels:
            constraint = f"  [{rel.constraint_name}]" if rel.constraint_name else ""
            print(
                f"    {rel.from_column:<28} → "
                f"{rel.to_table_name}.{rel.to_column:<28} "
                f"({rel.relationship_type}){constraint}"
            )


def _print_table_search_results(
    results: list[TableSearchResult],
    query: str,
) -> None:
    """Pretty-print table search results."""
    _header(f"Table Search Results  —  '{query}'")

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, start=1):
        similarity = 1.0 - r.distance  # cosine distance → similarity
        bar = "█" * int(similarity * 20) + "░" * (20 - int(similarity * 20))
        print(f"\n  [{i}] {r.qualified_name}")
        print(f"       Similarity : {similarity:.4f}  {bar}")
        if r.business_domain:
            print(f"       Domain     : {r.business_domain}")
        if r.description:
            wrapped = textwrap.fill(
                r.description,
                width=60,
                initial_indent="       Description: ",
                subsequent_indent="                    ",
            )
            print(wrapped)
        print(f"       table_id   : {r.table_id}")


def _print_column_search_results(
    results: list[ColumnSearchResult],
    query: str,
    scoped_to: str | None = None,
) -> None:
    """Pretty-print column search results."""
    scope_label = f" (scoped to table '{scoped_to}')" if scoped_to else ""
    _header(f"Column Search Results  —  '{query}'{scope_label}")

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results, start=1):
        similarity = 1.0 - r.distance
        bar = "█" * int(similarity * 20) + "░" * (20 - int(similarity * 20))
        pii_tag = "  ⚠ PII" if r.is_pii else ""
        fk_tag = "  FK" if r.is_foreign_key else ""
        pk_tag = "  PK" if r.is_primary_key else ""
        print(f"\n  [{i}] {r.qualified_name}  ({r.data_type}){pk_tag}{fk_tag}{pii_tag}")
        print(f"       Table      : {r.table_name}")
        print(f"       Similarity : {similarity:.4f}  {bar}")
        if r.description:
            wrapped = textwrap.fill(
                r.description,
                width=60,
                initial_indent="       Description: ",
                subsequent_indent="                    ",
            )
            print(wrapped)
        if r.business_meaning:
            wrapped = textwrap.fill(
                r.business_meaning,
                width=60,
                initial_indent="       Meaning    : ",
                subsequent_indent="                    ",
            )
            print(wrapped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # --- Load configs -------------------------------------------------------
    try:
        source_config = DatabaseConfig.from_env(prefix=args.source_prefix)
    except EnvironmentError as exc:
        print(f"\n✗  Source DB config error: {exc}", file=sys.stderr)
        print(
            f"   Expected: {args.source_prefix}_NAME, {args.source_prefix}_USER, "
            f"{args.source_prefix}_PASSWORD",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        repo_config = DatabaseConfig.from_env(prefix=args.repo_prefix)
    except EnvironmentError as exc:
        print(f"\n✗  Repo DB config error: {exc}", file=sys.stderr)
        print(
            f"   Expected: {args.repo_prefix}_NAME, {args.repo_prefix}_USER, "
            f"{args.repo_prefix}_PASSWORD",
            file=sys.stderr,
        )
        sys.exit(1)

    chroma_config = ChromaConfig.from_env()

    # --- Compute source hash and load KG ------------------------------------
    source_hash = _compute_source_hash(source_config)

    manager = KGManager(
        repo_config=repo_config,
        chroma_config=chroma_config,
    )

    print(f"\n  Loading KG for '{source_config.dbname}' …")
    kg = manager.load(source_db_hash=source_hash)

    if kg is None:
        print(
            f"\n✗  No Knowledge Graph found for database '{source_config.dbname}'.\n"
            "   Run `python scripts/build_kg.py` first to build the KG.",
            file=sys.stderr,
        )
        sys.exit(1)

    # =========================================================================
    # Section 1 — KG overview
    # =========================================================================
    _print_kg_overview(kg)

    # =========================================================================
    # Section 2 — Tables & columns
    # =========================================================================
    _print_tables(kg)

    # =========================================================================
    # Section 3 — Relationships
    # =========================================================================
    _print_relationships(kg)

    # =========================================================================
    # Section 4 — Semantic table search
    # =========================================================================
    # Check whether embeddings were indexed
    chroma_table_count = manager.table_count(kg.kg_id)
    chroma_column_count = manager.column_count(kg.kg_id)

    if chroma_table_count == 0 and chroma_column_count == 0:
        print()
        print(_SEP_WIDE)
        print("  ⚠  No embeddings found in Chroma for this KG.")
        print("     Run `python scripts/build_kg.py --force-rebuild` to generate them.")
        print(_SEP_WIDE)
        sys.exit(0)

    query = args.query
    top_k = args.top_k

    # Table search
    table_results = manager.search_tables(query=query, kg_id=kg.kg_id, n_results=top_k)
    _print_table_search_results(table_results, query)

    # =========================================================================
    # Section 5 — Semantic column search (global)
    # =========================================================================
    col_query = query  # reuse the same query for column search
    column_results = manager.search_columns(
        query=col_query,
        kg_id=kg.kg_id,
        n_results=top_k,
    )
    _print_column_search_results(column_results, col_query)

    # =========================================================================
    # Section 6 — Scoped column search  (first table from table results)
    # =========================================================================
    if table_results:
        top_table = table_results[0].table_name
        scoped_results = manager.search_columns(
            query=col_query,
            kg_id=kg.kg_id,
            n_results=min(top_k, 5),
            table_name=top_table,
        )
        _print_column_search_results(scoped_results, col_query, scoped_to=top_table)

    # =========================================================================
    # Footer
    # =========================================================================
    print()
    print(_SEP_WIDE)
    print("  Done.  Chroma index stats:")
    print(f"    Tables indexed : {chroma_table_count}")
    print(f"    Columns indexed: {chroma_column_count}")
    print(_SEP_WIDE)
    print()


if __name__ == "__main__":
    main()
