#!/usr/bin/env python
"""
build_kg.py — Build (or rebuild) the Knowledge Graph for a source database.

This script runs the full end-to-end KG pipeline:

    1. Schema extraction    (always)
    2. Description generation  (skippable via --skip-descriptions)
    3. Embedding generation    (skippable via --skip-embeddings)
    4. Persistence to PostgreSQL repository  (always)
    5. Indexing in Chroma vector store       (skipped when --skip-embeddings)

When the KG for the source database already exists in the repository and
--force-rebuild is NOT passed, the pipeline is short-circuited and the cached
KG is loaded and returned immediately — no LLM calls are made.

Usage
-----
    python scripts/build_kg.py
    python scripts/build_kg.py --force-rebuild
    python scripts/build_kg.py --skip-descriptions
    python scripts/build_kg.py --skip-embeddings
    python scripts/build_kg.py --skip-descriptions --skip-embeddings
    python scripts/build_kg.py --source-prefix DB --repo-prefix REPO_DB
    python scripts/build_kg.py --log-level DEBUG

Required environment variables
-------------------------------
Source database  (prefix DB by default):
    DB_NAME         Name of the PostgreSQL database to introspect
    DB_USER         Database username
    DB_PASSWORD     Database password
    DB_HOST         Host (default: localhost)
    DB_PORT         Port (default: 5432)
    DB_SCHEMA       Schema to introspect (default: public)

Repository database  (prefix REPO_DB by default):
    REPO_DB_NAME      Name of the repository database
    REPO_DB_USER      Database username
    REPO_DB_PASSWORD  Database password
    REPO_DB_HOST      Host (default: localhost)
    REPO_DB_PORT      Port (default: 5432)
    REPO_DB_SCHEMA    Schema (default: public)

Optional:
    CHROMA_PERSIST_DIR   Path for Chroma persistent storage (default: ./chroma_db)
    OPENAI_API_KEY       Required unless --skip-descriptions and --skip-embeddings
    OPENAI_CHAT_MODEL    Chat model for descriptions (default: gpt-4.1-mini)
    OPENAI_EMBEDDING_MODEL  Embedding model (default: text-embedding-3-small)
"""

from __future__ import annotations

import argparse
import logging

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when this script is run directly.
# ---------------------------------------------------------------------------
import os
import sys
import textwrap
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import ChromaConfig, DatabaseConfig  # noqa: E402
from src.kg.builders.kg_builder import KGBuilder  # noqa: E402
from src.kg.models.knowledge_graph import KnowledgeGraph  # noqa: E402

# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_kg",
        description=textwrap.dedent(
            """\
            Build (or rebuild) the Knowledge Graph for a source PostgreSQL database.

            On the first run the full pipeline executes:
              extract → describe → embed → persist to Postgres + Chroma.

            On subsequent runs the cached KG is returned instantly unless
            --force-rebuild is passed.
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              # First build (uses env vars)
              python scripts/build_kg.py

              # Force re-extraction and re-generation
              python scripts/build_kg.py --force-rebuild

              # Schema only — no LLM calls at all
              python scripts/build_kg.py --skip-descriptions --skip-embeddings

              # Use a non-default env var prefix
              python scripts/build_kg.py --source-prefix ANALYTICS_DB
            """
        ),
    )

    # ---- pipeline flags ----------------------------------------------------
    pipeline = parser.add_argument_group("Pipeline control")
    pipeline.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help=(
            "Force re-extraction and re-generation from scratch, even if a "
            "KG for this database already exists in the repository."
        ),
    )
    pipeline.add_argument(
        "--skip-descriptions",
        action="store_true",
        default=False,
        help=(
            "Skip LLM description generation (no gpt-4o-mini calls). "
            "Tables and columns will have null description fields."
        ),
    )
    pipeline.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=False,
        help=(
            "Skip embedding generation and Chroma indexing "
            "(no text-embedding-3-small calls). "
            "Semantic search will not be available for this KG."
        ),
    )

    # ---- env var prefixes --------------------------------------------------
    env = parser.add_argument_group("Environment variable prefixes")
    env.add_argument(
        "--source-prefix",
        default="DB",
        metavar="PREFIX",
        help=(
            "Env var prefix for the source database "
            "(default: DB → reads DB_HOST, DB_PORT, DB_NAME, …)."
        ),
    )
    env.add_argument(
        "--repo-prefix",
        default="REPO_DB",
        metavar="PREFIX",
        help=(
            "Env var prefix for the repository database "
            "(default: REPO_DB → reads REPO_DB_HOST, REPO_DB_PORT, …)."
        ),
    )

    # ---- logging -----------------------------------------------------------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return parser


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_source_config(prefix: str) -> DatabaseConfig:
    """Load and return the source DatabaseConfig, exiting on error."""
    logger = logging.getLogger("build_kg")
    try:
        cfg = DatabaseConfig.from_env(prefix=prefix)
        logger.info("Source DB  : %s", cfg.safe_repr())
        return cfg
    except EnvironmentError as exc:
        logger.error("Source database config error: %s", exc)
        logger.error(
            "Expected env vars: %s_NAME, %s_USER, %s_PASSWORD "
            "(and optionally %s_HOST, %s_PORT, %s_SCHEMA).",
            prefix, prefix, prefix, prefix, prefix, prefix,
            prefix,
            prefix,
            prefix,
            prefix,
            prefix,
            prefix,
        )
        sys.exit(1)


def _load_repo_config(prefix: str) -> DatabaseConfig:
    """Load and return the repository DatabaseConfig, exiting on error."""
    logger = logging.getLogger("build_kg")
    try:
        cfg = DatabaseConfig.from_env(prefix=prefix)
        logger.info("Repo   DB  : %s", cfg.safe_repr())
        return cfg
    except EnvironmentError as exc:
        logger.error("Repository database config error: %s", exc)
        logger.error(
            "Expected env vars: %s_NAME, %s_USER, %s_PASSWORD "
            "(and optionally %s_HOST, %s_PORT, %s_SCHEMA).",
            prefix,
            prefix,
            prefix,
            prefix,
            prefix,
            prefix,
        )
        sys.exit(1)


def _load_chroma_config() -> ChromaConfig:
    """Load and return the ChromaConfig."""
    logger = logging.getLogger("build_kg")
    cfg = ChromaConfig.from_env()
    logger.info("Chroma     : %s", cfg)
    return cfg


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(kg: KnowledgeGraph, elapsed_seconds: float) -> None:
    """Print a formatted build summary to stdout."""
    total_cols = sum(len(t.columns) for t in kg.tables.values())
    emb_tables = sum(1 for t in kg.tables.values() if t.embedding is not None)
    emb_cols = sum(
        1
        for t in kg.tables.values()
        for c in t.columns.values()
        if c.embedding is not None
    )
    desc_tables = sum(1 for t in kg.tables.values() if t.description is not None)
    desc_cols = sum(
        1
        for t in kg.tables.values()
        for c in t.columns.values()
        if c.description is not None
    )

    width = 62
    sep = "═" * width

    lines = [
        "",
        sep,
        "  ✅  Knowledge Graph Build Complete",
        sep,
        f"  KG ID            : {kg.kg_id}",
        f"  Source database  : {kg.source_db_name}",
        f"  Status           : {kg.status}",
        f"  Last updated     : {kg.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
        "  " + "─" * (width - 2),
        f"  Tables           : {len(kg.tables)}",
        f"  Columns          : {total_cols}",
        f"  Relationships    : {len(kg.relationships)}",
        "  " + "─" * (width - 2),
        f"  Described tables : {desc_tables} / {len(kg.tables)}",
        f"  Described cols   : {desc_cols} / {total_cols}",
        f"  Embedded tables  : {emb_tables} / {len(kg.tables)}",
        f"  Embedded cols    : {emb_cols} / {total_cols}",
        "  " + "─" * (width - 2),
        f"  Elapsed          : {elapsed_seconds:.1f}s",
        sep,
        "",
    ]
    print("\n".join(lines))

    # Print the table breakdown
    if kg.tables:
        print("  Tables in graph:")
        for name, table in sorted(kg.tables.items()):
            n_cols = len(table.columns)
            domain = f"  [{table.business_domain}]" if table.business_domain else ""
            rows = (
                f"  ~{table.row_count_estimate:,} rows"
                if table.row_count_estimate is not None
                else ""
            )
            print(f"    • {name:<35} {n_cols:>3} col(s){rows}{domain}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger = logging.getLogger("build_kg")

    logger.info("=" * 60)
    logger.info(
        "  KG Build Pipeline  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    if args.force_rebuild:
        logger.info("  Mode: FORCE REBUILD (ignoring any cached KG)")
    if args.skip_descriptions:
        logger.info("  Mode: SKIP DESCRIPTIONS")
    if args.skip_embeddings:
        logger.info("  Mode: SKIP EMBEDDINGS")
    logger.info("=" * 60)

    # --- Load configs -------------------------------------------------------
    source_config = _load_source_config(args.source_prefix)
    repo_config = _load_repo_config(args.repo_prefix)
    chroma_config = _load_chroma_config()

    # --- Run pipeline -------------------------------------------------------
    builder = KGBuilder(
        source_config=source_config,
        repo_config=repo_config,
        chroma_config=chroma_config,
        skip_descriptions=args.skip_descriptions,
        skip_embeddings=args.skip_embeddings,
    )

    start = datetime.now()
    try:
        kg = builder.build(force_rebuild=args.force_rebuild)
    except KeyboardInterrupt:
        logger.warning("Build interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("KG build pipeline failed with an unhandled error.")
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()

    # --- Print summary ------------------------------------------------------
    _print_summary(kg, elapsed)


if __name__ == "__main__":
    main()
