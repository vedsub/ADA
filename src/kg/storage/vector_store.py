from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

import chromadb
from pydantic import BaseModel

from src.kg.generators.embedding_generator import (
    _build_column_embedding_text,
    _build_table_embedding_text,
)
from src.kg.models.column import Column
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.models.table import Table

logger = logging.getLogger(__name__)

_TABLE_COLLECTION = "kg_tables"
_COLUMN_COLLECTION = "kg_columns"
_COSINE_METADATA = {"hnsw:space": "cosine"}


# ---------------------------------------------------------------------------
# Search result models
# ---------------------------------------------------------------------------


class TableSearchResult(BaseModel):
    """A single table document returned by a similarity search."""

    table_id: UUID
    kg_id: UUID
    table_name: str
    qualified_name: str
    schema_name: str
    table_type: str
    description: str
    business_domain: str
    distance: float
    document: str  # the text that was embedded


class ColumnSearchResult(BaseModel):
    """A single column document returned by a similarity search."""

    column_id: UUID
    table_id: UUID
    kg_id: UUID
    table_name: str
    column_name: str
    qualified_name: str
    data_type: str
    is_primary_key: bool
    is_foreign_key: bool
    is_pii: bool
    description: str
    business_meaning: str
    distance: float
    document: str  # the text that was embedded


# ---------------------------------------------------------------------------
# Metadata builders  (pure helpers – easy to test in isolation)
# ---------------------------------------------------------------------------


def _table_metadata(table: Table, kg_id: UUID) -> dict:
    """
    Build the Chroma metadata dict for a table document.

    All values are ``str | int | float | bool`` — Chroma does not accept
    ``None``, so optional fields fall back to an empty string.
    """
    return {
        "kg_id": str(kg_id),
        "table_id": str(table.table_id),
        "table_name": table.table_name,
        "schema_name": table.schema_name,
        "qualified_name": table.qualified_name,
        "table_type": table.table_type,
        "description": table.description or "",
        "business_domain": table.business_domain or "",
    }


def _column_metadata(column: Column, table: Table, kg_id: UUID) -> dict:
    """
    Build the Chroma metadata dict for a column document.

    All values are ``str | int | float | bool``.
    """
    return {
        "kg_id": str(kg_id),
        "column_id": str(column.column_id),
        "table_id": str(column.table_id),
        "table_name": table.table_name,
        "column_name": column.column_name,
        "qualified_name": column.qualified_name,
        "data_type": column.data_type,
        "is_primary_key": column.is_primary_key,
        "is_foreign_key": column.is_foreign_key,
        "is_pii": column.is_pii,
        "description": column.description or "",
        "business_meaning": column.business_meaning or "",
    }


# ---------------------------------------------------------------------------
# ChromaVectorStore
# ---------------------------------------------------------------------------


class ChromaVectorStore:
    """
    Chroma-backed vector store for semantic similarity search over KG tables
    and columns.

    Two persistent collections are maintained:

    +------------------+-----------------------------------------------+
    | Collection       | Contents                                      |
    +==================+===============================================+
    | ``kg_tables``    | One document per table across all KGs         |
    | ``kg_columns``   | One document per column across all KGs        |
    +------------------+-----------------------------------------------+

    Both collections use **cosine** similarity.  Pre-computed OpenAI
    embeddings (``Table.embedding`` / ``Column.embedding``) are always
    provided explicitly — the collection's default embedding function is
    never invoked.

    Every document stores a ``kg_id`` metadata field so that queries can
    be scoped to a single source database, while a single Chroma store
    can serve multiple KGs simultaneously.

    Parameters
    ----------
    persist_directory:
        Path to the directory where Chroma persists its data on disk.
        Defaults to ``"./chroma_db"``.  Pass the value from
        ``ChromaConfig.from_env().persist_directory`` in production.
    """

    def __init__(self, persist_directory: str = "./chroma_db") -> None:
        self.persist_directory = persist_directory
        self._client = chromadb.PersistentClient(path=persist_directory)
        logger.info("ChromaVectorStore initialised at '%s'.", persist_directory)

    # ------------------------------------------------------------------
    # Collection accessors  (lazy get-or-create)
    # ------------------------------------------------------------------

    def _table_collection(self):
        """Return (or create) the ``kg_tables`` collection."""
        return self._client.get_or_create_collection(
            name=_TABLE_COLLECTION,
            metadata=_COSINE_METADATA,
        )

    def _column_collection(self):
        """Return (or create) the ``kg_columns`` collection."""
        return self._client.get_or_create_collection(
            name=_COLUMN_COLLECTION,
            metadata=_COSINE_METADATA,
        )

    # ------------------------------------------------------------------
    # Upsert – single item
    # ------------------------------------------------------------------

    def upsert_table(self, table: Table, kg_id: UUID) -> None:
        """
        Add or update a single table document in ``kg_tables``.

        Skips silently when ``table.embedding`` is ``None``.

        Parameters
        ----------
        table:
            The :class:`~src.kg.models.table.Table` to index.
        kg_id:
            UUID of the owning :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.
        """
        if table.embedding is None:
            logger.warning(
                "upsert_table: '%s' has no embedding — skipped.", table.table_name
            )
            return

        self._table_collection().upsert(
            ids=[str(table.table_id)],
            embeddings=[table.embedding],
            documents=[_build_table_embedding_text(table)],
            metadatas=[_table_metadata(table, kg_id)],
        )
        logger.debug("Upserted table '%s'.", table.table_name)

    def upsert_column(self, column: Column, table: Table, kg_id: UUID) -> None:
        """
        Add or update a single column document in ``kg_columns``.

        Skips silently when ``column.embedding`` is ``None``.

        Parameters
        ----------
        column:
            The :class:`~src.kg.models.column.Column` to index.
        table:
            The parent :class:`~src.kg.models.table.Table`; its context is
            woven into the stored document text.
        kg_id:
            UUID of the owning KG.
        """
        if column.embedding is None:
            logger.warning(
                "upsert_column: '%s' has no embedding — skipped.",
                column.qualified_name,
            )
            return

        self._column_collection().upsert(
            ids=[str(column.column_id)],
            embeddings=[column.embedding],
            documents=[_build_column_embedding_text(column, table)],
            metadatas=[_column_metadata(column, table, kg_id)],
        )
        logger.debug("Upserted column '%s'.", column.qualified_name)

    # ------------------------------------------------------------------
    # Upsert – full KG  (batched)
    # ------------------------------------------------------------------

    def upsert_kg(self, kg: KnowledgeGraph) -> None:
        """
        Add or update all tables and columns in *kg* that carry a non-``None``
        embedding.

        Batching strategy
        -----------------
        * All tables are upserted in **one** Chroma call.
        * Columns are upserted **per table** (one call per table) to keep
          individual batches manageable for very wide schemas.

        Parameters
        ----------
        kg:
            The enriched :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.
            Should already have embeddings populated by
            :class:`~src.kg.generators.embedding_generator.EmbeddingGenerator`.
        """
        embedded_tables = [t for t in kg.tables.values() if t.embedding is not None]
        embedded_col_count = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.embedding is not None
        )

        logger.info(
            "ChromaVectorStore.upsert_kg: %d table(s), %d column(s) for '%s'.",
            len(embedded_tables),
            embedded_col_count,
            kg.source_db_name,
        )

        # ---- tables: one batch call ----------------------------------------
        if embedded_tables:
            self._table_collection().upsert(
                ids=[str(t.table_id) for t in embedded_tables],
                embeddings=[t.embedding for t in embedded_tables],  # type: ignore[misc]
                documents=[_build_table_embedding_text(t) for t in embedded_tables],
                metadatas=[_table_metadata(t, kg.kg_id) for t in embedded_tables],
            )
            logger.info("  → %d table document(s) upserted.", len(embedded_tables))

        # ---- columns: one batch call per table -----------------------------
        total_col_upserted = 0
        for table in kg.tables.values():
            embedded_cols = [
                c for c in table.columns.values() if c.embedding is not None
            ]
            if not embedded_cols:
                continue

            self._column_collection().upsert(
                ids=[str(c.column_id) for c in embedded_cols],
                embeddings=[c.embedding for c in embedded_cols],  # type: ignore[misc]
                documents=[
                    _build_column_embedding_text(c, table) for c in embedded_cols
                ],
                metadatas=[_column_metadata(c, table, kg.kg_id) for c in embedded_cols],
            )
            total_col_upserted += len(embedded_cols)

        logger.info("  → %d column document(s) upserted.", total_col_upserted)
        logger.info("ChromaVectorStore.upsert_kg complete for '%s'.", kg.source_db_name)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_tables(
        self,
        query_embedding: list[float],
        kg_id: UUID,
        n_results: int = 5,
    ) -> list[TableSearchResult]:
        """
        Return the *n_results* tables most semantically similar to
        *query_embedding* within the specified KG.

        Results are ordered by ascending cosine distance (closest first).
        Returns an empty list when the collection is empty or on any error.

        Parameters
        ----------
        query_embedding:
            Pre-computed query vector (must match the stored dimensionality).
        kg_id:
            Restrict results to this KG.
        n_results:
            Maximum number of tables to return.
        """
        collection = self._table_collection()
        safe_n = self._safe_n(collection, str(kg_id), n_results)
        if safe_n == 0:
            logger.debug("search_tables: no indexed tables for kg_id=%s.", kg_id)
            return []

        try:
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=safe_n,
                where={"kg_id": str(kg_id)},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("search_tables query failed: %s", exc)
            return []

        return self._parse_table_results(raw)

    def search_columns(
        self,
        query_embedding: list[float],
        kg_id: UUID,
        n_results: int = 10,
        table_name: Optional[str] = None,
    ) -> list[ColumnSearchResult]:
        """
        Return the *n_results* columns most semantically similar to
        *query_embedding* within the specified KG.

        Results are ordered by ascending cosine distance (closest first).
        Returns an empty list when the collection is empty or on any error.

        Parameters
        ----------
        query_embedding:
            Pre-computed query vector.
        kg_id:
            Restrict results to this KG.
        n_results:
            Maximum number of columns to return.
        table_name:
            Optional extra filter to restrict results to one table.
        """
        collection = self._column_collection()

        where: dict = (
            {"$and": [{"kg_id": str(kg_id)}, {"table_name": table_name}]}
            if table_name
            else {"kg_id": str(kg_id)}
        )
        safe_n = self._safe_n(collection, str(kg_id), n_results)
        if safe_n == 0:
            logger.debug("search_columns: no indexed columns for kg_id=%s.", kg_id)
            return []

        try:
            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=safe_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("search_columns query failed: %s", exc)
            return []

        return self._parse_column_results(raw)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_kg(self, kg_id: UUID) -> None:
        """
        Remove all table and column documents that belong to *kg_id*.

        Safe to call even when no documents exist for that KG.

        Parameters
        ----------
        kg_id:
            The KG whose documents should be purged from both collections.
        """
        where = {"kg_id": str(kg_id)}
        try:
            self._table_collection().delete(where=where)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_kg: could not delete tables (%s).", exc)

        try:
            self._column_collection().delete(where=where)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_kg: could not delete columns (%s).", exc)

        logger.info("Deleted all Chroma documents for kg_id=%s.", kg_id)

    def delete_table(self, table_id: UUID) -> None:
        """Remove a single table document by its ``table_id``."""
        try:
            self._table_collection().delete(ids=[str(table_id)])
            logger.debug("Deleted table document table_id=%s.", table_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_table failed for %s: %s", table_id, exc)

    def delete_column(self, column_id: UUID) -> None:
        """Remove a single column document by its ``column_id``."""
        try:
            self._column_collection().delete(ids=[str(column_id)])
            logger.debug("Deleted column document column_id=%s.", column_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_column failed for %s: %s", column_id, exc)

    # ------------------------------------------------------------------
    # Counts
    # ------------------------------------------------------------------

    def count_tables(self, kg_id: UUID) -> int:
        """
        Return the number of table documents indexed for *kg_id*.

        Uses Chroma's ``get()`` with a metadata filter rather than
        ``count()`` (which has no filter support).
        """
        try:
            result = self._table_collection().get(
                where={"kg_id": str(kg_id)},
                include=[],
            )
            return len(result["ids"])
        except Exception:  # noqa: BLE001
            return 0

    def count_columns(self, kg_id: UUID) -> int:
        """Return the number of column documents indexed for *kg_id*."""
        try:
            result = self._column_collection().get(
                where={"kg_id": str(kg_id)},
                include=[],
            )
            return len(result["ids"])
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Private – result parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_table_results(raw: dict) -> list[TableSearchResult]:
        """
        Convert the raw Chroma query response into a list of
        :class:`TableSearchResult` objects.

        Chroma returns nested lists (one inner list per query vector).
        Since we always send a single query vector the outer list always
        has exactly one element.
        """
        results: list[TableSearchResult] = []

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            try:
                results.append(
                    TableSearchResult(
                        table_id=UUID(meta["table_id"]),
                        kg_id=UUID(meta["kg_id"]),
                        table_name=meta["table_name"],
                        qualified_name=meta["qualified_name"],
                        schema_name=meta["schema_name"],
                        table_type=meta["table_type"],
                        description=meta.get("description", ""),
                        business_domain=meta.get("business_domain", ""),
                        distance=float(dist),
                        document=doc or "",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_parse_table_results: skipping malformed row id=%s (%s).",
                    doc_id,
                    exc,
                )

        return results

    @staticmethod
    def _parse_column_results(raw: dict) -> list[ColumnSearchResult]:
        """
        Convert the raw Chroma query response into a list of
        :class:`ColumnSearchResult` objects.
        """
        results: list[ColumnSearchResult] = []

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            try:
                results.append(
                    ColumnSearchResult(
                        column_id=UUID(meta["column_id"]),
                        table_id=UUID(meta["table_id"]),
                        kg_id=UUID(meta["kg_id"]),
                        table_name=meta["table_name"],
                        column_name=meta["column_name"],
                        qualified_name=meta["qualified_name"],
                        data_type=meta["data_type"],
                        is_primary_key=bool(meta.get("is_primary_key", False)),
                        is_foreign_key=bool(meta.get("is_foreign_key", False)),
                        is_pii=bool(meta.get("is_pii", False)),
                        description=meta.get("description", ""),
                        business_meaning=meta.get("business_meaning", ""),
                        distance=float(dist),
                        document=doc or "",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_parse_column_results: skipping malformed row id=%s (%s).",
                    doc_id,
                    exc,
                )

        return results

    # ------------------------------------------------------------------
    # Private – n_results guard
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_n(collection, kg_id_str: str, requested: int) -> int:
        """
        Return ``min(requested, actual_count_for_kg)``.

        Chroma raises an error when ``n_results`` exceeds the number of
        documents in the collection (or matching the filter).  This guard
        prevents that by counting matching documents first.
        """
        try:
            result = collection.get(
                where={"kg_id": kg_id_str},
                include=[],
            )
            available = len(result["ids"])
        except Exception:  # noqa: BLE001
            available = 0

        return min(requested, available)
