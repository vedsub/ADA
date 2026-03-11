from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from config.settings import ChromaConfig, DatabaseConfig
from src.kg.builders.kg_builder import KGBuilder
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.storage.kg_repository import KGRepository
from src.kg.storage.vector_store import (
    ChromaVectorStore,
    ColumnSearchResult,
    TableSearchResult,
)
from src.openai_client import OpenAILLMClient, get_default_llm_client

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"


class KGManager:
    """
    High-level manager for all knowledge graph operations.

    ``KGManager`` is the single entry point for application code that
    needs to interact with the KG.  It owns both the PostgreSQL repository
    and the Chroma vector store, and exposes four main capabilities:

    * **Build** — run the full pipeline (extract → describe → embed →
      persist) via :meth:`build`.
    * **Load** — retrieve a previously-built KG from the repository via
      :meth:`load`.
    * **Search** — perform semantic similarity search over tables or
      columns via :meth:`search_tables` and :meth:`search_columns`.
    * **Delete** — remove a KG from both the repository and Chroma via
      :meth:`delete`.

    The manager generates query embeddings on-the-fly using the same
    ``text-embedding-3-small`` model that was used during KG construction,
    so query vectors are always compatible with the stored document vectors.

    Parameters
    ----------
    repo_config:
        :class:`~config.settings.DatabaseConfig` for the repository
        database.  Build from ``REPO_DB_*`` env variables via
        :meth:`~config.settings.DatabaseConfig.repo_db_from_env`.
    chroma_config:
        :class:`~config.settings.ChromaConfig` for the Chroma vector
        store.  Build from ``CHROMA_*`` env variables via
        :meth:`~config.settings.ChromaConfig.from_env`.
    llm_client:
        Optional shared :class:`~src.openai_client.OpenAILLMClient`.
        When ``None`` the module-level singleton returned by
        :func:`~src.openai_client.get_default_llm_client` is used.

    Example
    -------
    .. code-block:: python

        from config.settings import ChromaConfig, DatabaseConfig
        from src.kg.manager.kg_manager import KGManager

        manager = KGManager(
            repo_config=DatabaseConfig.repo_db_from_env(),
            chroma_config=ChromaConfig.from_env(),
        )

        # Build (or load from cache)
        kg = manager.build(source_config=DatabaseConfig.source_db_from_env())

        # Search
        tables = manager.search_tables("customer orders", kg.kg_id)
        columns = manager.search_columns("total revenue amount", kg.kg_id)
    """

    def __init__(
        self,
        repo_config: DatabaseConfig,
        chroma_config: ChromaConfig,
        llm_client: Optional[OpenAILLMClient] = None,
    ) -> None:
        self.repo_config = repo_config
        self.chroma_config = chroma_config
        self.llm_client = llm_client or get_default_llm_client()
        self._vector_store = ChromaVectorStore(chroma_config.persist_directory)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        source_config: DatabaseConfig,
        *,
        force_rebuild: bool = False,
        skip_descriptions: bool = False,
        skip_embeddings: bool = False,
    ) -> KnowledgeGraph:
        """
        Build a :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`
        for the given source database and return it.

        When ``force_rebuild=False`` (the default) and a KG for the same
        source database already exists in the repository, it is loaded and
        returned immediately without contacting the source database or
        making any LLM calls.

        Parameters
        ----------
        source_config:
            Connection parameters for the PostgreSQL database to introspect.
        force_rebuild:
            When ``True``, re-run the full pipeline even if a cached KG
            already exists.
        skip_descriptions:
            When ``True``, skip LLM description generation for tables and
            columns.  Useful for fast/cheap builds where you only need
            schema structure and embeddings.
        skip_embeddings:
            When ``True``, skip embedding generation and Chroma indexing.
            The returned KG will have ``None`` for all ``embedding`` fields
            and no documents will be added to the vector store.

        Returns
        -------
        KnowledgeGraph
            A fully-populated graph with ``status == "ready"``.
        """
        logger.info(
            "KGManager.build: source='%s'  force_rebuild=%s  "
            "skip_descriptions=%s  skip_embeddings=%s",
            source_config.dbname,
            force_rebuild,
            skip_descriptions,
            skip_embeddings,
        )
        builder = KGBuilder(
            source_config=source_config,
            repo_config=self.repo_config,
            chroma_config=self.chroma_config,
            llm_client=self.llm_client,
            skip_descriptions=skip_descriptions,
            skip_embeddings=skip_embeddings,
        )
        return builder.build(force_rebuild=force_rebuild)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        *,
        source_db_hash: Optional[str] = None,
        kg_id: Optional[UUID] = None,
    ) -> Optional[KnowledgeGraph]:
        """
        Load an existing :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`
        from the repository.

        Exactly one of *source_db_hash* or *kg_id* must be provided.

        Parameters
        ----------
        source_db_hash:
            SHA-256 hex digest of ``host:port/dbname``.  This is the value
            stored in :attr:`~src.kg.models.knowledge_graph.KnowledgeGraph.source_db_hash`
            and can be computed via
            :func:`~src.kg.builders.kg_builder._compute_source_hash`.
        kg_id:
            The :class:`~uuid.UUID` of the KG (its ``kg_id`` primary key).

        Returns
        -------
        KnowledgeGraph or None
            The loaded graph, or ``None`` if no matching KG was found.

        Raises
        ------
        ValueError
            If neither or both of *source_db_hash* and *kg_id* are provided.
        """
        if source_db_hash is None and kg_id is None:
            raise ValueError("Provide exactly one of 'source_db_hash' or 'kg_id'.")
        if source_db_hash is not None and kg_id is not None:
            raise ValueError(
                "Provide exactly one of 'source_db_hash' or 'kg_id', not both."
            )

        logger.info(
            "KGManager.load: %s",
            f"source_db_hash={source_db_hash!r}"
            if source_db_hash
            else f"kg_id={kg_id}",
        )

        with KGRepository(self.repo_config) as repo:
            if source_db_hash is not None:
                kg = repo.load_kg(source_db_hash)
            else:
                kg = repo.load_kg_by_id(kg_id)  # type: ignore[arg-type]

        if kg is None:
            logger.info("KGManager.load: no KG found.")
        else:
            total_cols = sum(len(t.columns) for t in kg.tables.values())
            logger.info(
                "KGManager.load: loaded kg_id=%s  tables=%d  columns=%d  rels=%d",
                kg.kg_id,
                len(kg.tables),
                total_cols,
                len(kg.relationships),
            )
        return kg

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_tables(
        self,
        query: str,
        kg_id: UUID,
        n_results: int = 5,
    ) -> list[TableSearchResult]:
        """
        Semantic search for tables that are most relevant to *query*.

        The query text is embedded on-the-fly using the same model
        (``text-embedding-3-small``) as the stored table documents.
        Results are ordered by ascending cosine distance (closest first).

        Parameters
        ----------
        query:
            Natural-language query, e.g. ``"customer purchase history"``.
        kg_id:
            Restrict results to this KG.  Pass
            :attr:`~src.kg.models.knowledge_graph.KnowledgeGraph.kg_id`.
        n_results:
            Maximum number of tables to return (default 5).

        Returns
        -------
        list[TableSearchResult]
            Ordered list of matching tables with distance scores.
            Empty list when no indexed tables exist for the KG.
        """
        logger.debug("search_tables: query=%r  kg_id=%s  n=%d", query, kg_id, n_results)
        query_vector = self._embed_query(query)
        results = self._vector_store.search_tables(
            query_embedding=query_vector,
            kg_id=kg_id,
            n_results=n_results,
        )
        logger.debug("search_tables: %d result(s) returned.", len(results))
        return results

    def search_columns(
        self,
        query: str,
        kg_id: UUID,
        n_results: int = 10,
        table_name: Optional[str] = None,
    ) -> list[ColumnSearchResult]:
        """
        Semantic search for columns that are most relevant to *query*.

        The query text is embedded on-the-fly.  Results are ordered by
        ascending cosine distance (closest first).

        Parameters
        ----------
        query:
            Natural-language query, e.g. ``"total revenue amount"``.
        kg_id:
            Restrict results to this KG.
        n_results:
            Maximum number of columns to return (default 10).
        table_name:
            Optional table-name filter.  When provided, only columns
            belonging to *table_name* are considered.

        Returns
        -------
        list[ColumnSearchResult]
            Ordered list of matching columns with distance scores.
            Empty list when no indexed columns exist for the KG (or table).
        """
        logger.debug(
            "search_columns: query=%r  kg_id=%s  n=%d  table=%r",
            query,
            kg_id,
            n_results,
            table_name,
        )
        query_vector = self._embed_query(query)
        results = self._vector_store.search_columns(
            query_embedding=query_vector,
            kg_id=kg_id,
            n_results=n_results,
            table_name=table_name,
        )
        logger.debug("search_columns: %d result(s) returned.", len(results))
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, kg_id: UUID) -> None:
        """
        Permanently delete a KG from both the PostgreSQL repository and
        the Chroma vector store.

        The cascade constraints in the repository ensure that all associated
        tables, columns, relationships, and embeddings are removed when the
        ``kg_metadata`` row is deleted.

        Parameters
        ----------
        kg_id:
            The UUID of the KG to delete.
        """
        logger.info("KGManager.delete: deleting kg_id=%s …", kg_id)

        with KGRepository(self.repo_config) as repo:
            repo.delete_kg(kg_id)
        logger.debug("  Deleted from PostgreSQL repository.")

        self._vector_store.delete_kg(kg_id)
        logger.debug("  Deleted from Chroma vector store.")

        logger.info("KGManager.delete: kg_id=%s removed from all stores.", kg_id)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def exists(self, source_db_hash: str) -> bool:
        """
        Return ``True`` if a KG has already been built for the given
        source database hash.

        Parameters
        ----------
        source_db_hash:
            SHA-256 hex digest of ``host:port/dbname``.
        """
        with KGRepository(self.repo_config) as repo:
            repo.create_schema()
            return repo.exists(source_db_hash)

    def table_count(self, kg_id: UUID) -> int:
        """Return the number of table documents indexed in Chroma for *kg_id*."""
        return self._vector_store.count_tables(kg_id)

    def column_count(self, kg_id: UUID) -> int:
        """Return the number of column documents indexed in Chroma for *kg_id*."""
        return self._vector_store.count_columns(kg_id)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """
        Generate a ``text-embedding-3-small`` vector for *query*.

        Returns
        -------
        list[float]
            1 536-dimensional embedding vector.

        Raises
        ------
        Exception
            Any OpenAI API error is propagated to the caller.
        """
        vector = self.llm_client.generate_embeddings(query, model=_EMBEDDING_MODEL)
        # generate_embeddings returns list[float] for a single string input
        return vector  # type: ignore[return-value]
