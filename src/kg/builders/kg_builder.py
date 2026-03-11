from __future__ import annotations

import hashlib
import logging
from typing import Optional

from config.settings import ChromaConfig, DatabaseConfig
from src.kg.builders.builder import KGBuilderBase
from src.kg.extractors.schema_extractor import SchemaExtractor
from src.kg.generators.description_generator import DescriptionGenerator
from src.kg.generators.embedding_generator import EmbeddingGenerator
from src.kg.models.knowledge_graph import KnowledgeGraph
from src.kg.storage.kg_repository import KGRepository
from src.kg.storage.vector_store import ChromaVectorStore
from src.openai_client import OpenAILLMClient, get_default_llm_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_source_hash(config: DatabaseConfig) -> str:
    """
    Return the stable SHA-256 hex digest for a source database config.

    Mirrors the logic in
    :meth:`~src.kg.extractors.schema_extractor.SchemaExtractor._make_db_hash`
    so the builder can look up an existing KG *before* opening a connection
    to the source database.

    The hash is derived from ``host:port/dbname`` only — deliberately
    excluding the username so that credential rotation never invalidates a
    stored KG.
    """
    raw = f"{config.host}:{config.port}/{config.dbname}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# KGBuilder
# ---------------------------------------------------------------------------


class KGBuilder(KGBuilderBase):
    """
    End-to-end pipeline for building a
    :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.

    Pipeline stages
    ---------------
    1. **Cache check** (skipped when ``force_rebuild=True``) —
       compute the source-database hash and return the already-persisted KG
       from the repository if one exists.  This avoids redundant LLM calls
       on subsequent runs.

    2. **Schema extraction** —
       :class:`~src.kg.extractors.schema_extractor.SchemaExtractor` connects
       to the source database and discovers tables, columns, and FK
       relationships.

    3. **Description generation** (skipped when ``skip_descriptions=True``) —
       :class:`~src.kg.generators.description_generator.DescriptionGenerator`
       calls ``gpt-4o-mini`` to produce human-readable descriptions,
       business-domain labels, and typical use-cases for every table and column.

    4. **Embedding generation** (skipped when ``skip_embeddings=True``) —
       :class:`~src.kg.generators.embedding_generator.EmbeddingGenerator`
       calls ``text-embedding-3-small`` to generate 1 536-dimensional vectors
       for every table and column.

    5. **Persistence** —
       :class:`~src.kg.storage.kg_repository.KGRepository` upserts the full
       schema metadata and embedding vectors into PostgreSQL.

    6. **Vector-store indexing** (skipped when ``skip_embeddings=True``) —
       :class:`~src.kg.storage.vector_store.ChromaVectorStore` upserts all
       document-embedding pairs into the Chroma persistent store so that
       semantic search is immediately available.

    Parameters
    ----------
    source_config:
        :class:`~config.settings.DatabaseConfig` for the source database
        being introspected.  Build from ``DB_*`` environment variables via
        :meth:`~config.settings.DatabaseConfig.source_db_from_env`.
    repo_config:
        :class:`~config.settings.DatabaseConfig` for the repository database
        where the KG is persisted.  Build from ``REPO_DB_*`` env variables
        via :meth:`~config.settings.DatabaseConfig.repo_db_from_env`.
    chroma_config:
        :class:`~config.settings.ChromaConfig` for the Chroma vector store.
        Build from ``CHROMA_*`` env variables via
        :meth:`~config.settings.ChromaConfig.from_env`.
    llm_client:
        Optional shared :class:`~src.openai_client.OpenAILLMClient`.  When
        ``None`` the module-level singleton returned by
        :func:`~src.openai_client.get_default_llm_client` is used.
    skip_descriptions:
        When ``True``, stage 3 (description generation) is skipped.  The KG
        will have ``None`` for all ``description`` / ``business_domain`` /
        ``typical_use_cases`` fields on tables, and ``None`` for
        ``description`` / ``business_meaning`` on columns.
    skip_embeddings:
        When ``True``, stages 4 and 6 (embedding generation and Chroma
        indexing) are skipped.  Embedding fields remain ``None`` and the
        Chroma store is not touched.

    Example
    -------
    .. code-block:: python

        from config.settings import ChromaConfig, DatabaseConfig
        from src.kg.builders.kg_builder import KGBuilder

        kg = KGBuilder(
            source_config=DatabaseConfig.source_db_from_env(),
            repo_config=DatabaseConfig.repo_db_from_env(),
            chroma_config=ChromaConfig.from_env(),
        ).build()

        print(kg.kg_id, len(kg.tables), "tables")
    """

    def __init__(
        self,
        source_config: DatabaseConfig,
        repo_config: DatabaseConfig,
        chroma_config: ChromaConfig,
        llm_client: Optional[OpenAILLMClient] = None,
        skip_descriptions: bool = False,
        skip_embeddings: bool = False,
    ) -> None:
        self.source_config = source_config
        self.repo_config = repo_config
        self.chroma_config = chroma_config
        self.llm_client = llm_client or get_default_llm_client()
        self.skip_descriptions = skip_descriptions
        self.skip_embeddings = skip_embeddings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, force_rebuild: bool = False) -> KnowledgeGraph:
        """
        Run the KG pipeline and return the populated
        :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.

        When ``force_rebuild=False`` and a KG already exists in the
        repository for the configured source database, the stored KG
        (including all previously-generated descriptions and embeddings) is
        returned immediately without touching the source database or making
        any LLM calls.

        Parameters
        ----------
        force_rebuild:
            Pass ``True`` to re-extract and re-generate everything even when
            a cached KG exists.

        Returns
        -------
        KnowledgeGraph
            A fully-populated graph with ``status == "ready"``.
        """
        source_hash = _compute_source_hash(self.source_config)

        # ------------------------------------------------------------------ #
        # Stage 0  — cache check                                             #
        # ------------------------------------------------------------------ #
        if not force_rebuild:
            cached = self._try_load_from_cache(source_hash)
            if cached is not None:
                return cached

        # ------------------------------------------------------------------ #
        # Stage 1  — schema extraction                                       #
        # ------------------------------------------------------------------ #
        kg = self._extract_schema()

        # ------------------------------------------------------------------ #
        # Stage 2  — description generation                                  #
        # ------------------------------------------------------------------ #
        if not self.skip_descriptions:
            self._generate_descriptions(kg)
        else:
            logger.info(
                "Stage 2/4 — Skipping description generation (--skip-descriptions)."
            )

        # ------------------------------------------------------------------ #
        # Stage 3  — embedding generation                                    #
        # ------------------------------------------------------------------ #
        if not self.skip_embeddings:
            self._generate_embeddings(kg)
        else:
            logger.info(
                "Stage 3/4 — Skipping embedding generation (--skip-embeddings)."
            )

        # ------------------------------------------------------------------ #
        # Stage 4  — persistence                                             #
        # ------------------------------------------------------------------ #
        self._persist(kg)

        # ------------------------------------------------------------------ #
        # Summary                                                             #
        # ------------------------------------------------------------------ #
        total_cols = sum(len(t.columns) for t in kg.tables.values())
        emb_tables = sum(1 for t in kg.tables.values() if t.embedding is not None)
        emb_cols = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.embedding is not None
        )
        desc_tables = sum(1 for t in kg.tables.values() if t.description is not None)

        logger.info(
            "KGBuilder complete — "
            "tables=%d  cols=%d  rels=%d  "
            "described=%d  emb_tables=%d  emb_cols=%d  "
            "kg_id=%s",
            len(kg.tables),
            total_cols,
            len(kg.relationships),
            desc_tables,
            emb_tables,
            emb_cols,
            kg.kg_id,
        )
        return kg

    # ------------------------------------------------------------------
    # Private — pipeline stages
    # ------------------------------------------------------------------

    def _try_load_from_cache(self, source_hash: str) -> Optional[KnowledgeGraph]:
        """
        Attempt to load an existing KG from the repository by source hash.

        Returns the :class:`KnowledgeGraph` if found, otherwise ``None``.
        Also ensures the schema exists so later stages can assume it.
        """
        logger.debug(
            "Stage 0/4 — Checking repository cache for hash '%s' …", source_hash
        )
        try:
            with KGRepository(self.repo_config) as repo:
                repo.create_schema()
                if repo.exists(source_hash):
                    logger.info(
                        "Stage 0/4 — KG already exists for '%s' — loading from repository.",
                        self.source_config.dbname,
                    )
                    kg = repo.load_kg(source_hash)
                    if kg is not None:
                        logger.info(
                            "  Loaded KG kg_id=%s  tables=%d  rels=%d",
                            kg.kg_id,
                            len(kg.tables),
                            len(kg.relationships),
                        )
                        return kg
                    logger.warning(
                        "  repo.load_kg returned None for hash '%s' — proceeding with rebuild.",
                        source_hash,
                    )
        except Exception:  # noqa: BLE001
            logger.warning(
                "  Cache check failed — proceeding with full rebuild.",
                exc_info=True,
            )
        return None

    def _extract_schema(self) -> KnowledgeGraph:
        """
        Stage 1 — extract schema from the source database.

        Returns the freshly-built :class:`KnowledgeGraph` with status ``"ready"``.
        """
        logger.info(
            "Stage 1/4 — Extracting schema from '%s' …",
            self.source_config.dbname,
        )
        extractor = SchemaExtractor(self.source_config)
        kg = extractor.extract()
        logger.info(
            "  → %d table(s), %d relationship(s) extracted.",
            len(kg.tables),
            len(kg.relationships),
        )
        return kg

    def _generate_descriptions(self, kg: KnowledgeGraph) -> None:
        """
        Stage 2 — enrich every table and column with LLM-generated descriptions.

        Mutates *kg* in-place.  Individual failures are caught inside
        :class:`~src.kg.generators.description_generator.DescriptionGenerator`
        and do not abort the pipeline.
        """
        total_tables = len(kg.tables)
        total_cols = sum(len(t.columns) for t in kg.tables.values())
        logger.info(
            "Stage 2/4 — Generating descriptions for %d table(s) and %d column(s) …",
            total_tables,
            total_cols,
        )
        DescriptionGenerator(llm_client=self.llm_client).generate_all(kg)

        described = sum(1 for t in kg.tables.values() if t.description is not None)
        described_cols = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.description is not None
        )
        logger.info(
            "  → %d/%d table(s) described, %d/%d column(s) described.",
            described,
            total_tables,
            described_cols,
            total_cols,
        )

    def _generate_embeddings(self, kg: KnowledgeGraph) -> None:
        """
        Stage 3 — generate ``text-embedding-3-small`` vectors for every
        table and column.

        Mutates *kg* in-place.  Individual failures are caught inside
        :class:`~src.kg.generators.embedding_generator.EmbeddingGenerator`
        and do not abort the pipeline.
        """
        total_tables = len(kg.tables)
        total_cols = sum(len(t.columns) for t in kg.tables.values())
        logger.info(
            "Stage 3/4 — Generating embeddings for %d table(s) and %d column(s) …",
            total_tables,
            total_cols,
        )
        EmbeddingGenerator(llm_client=self.llm_client).generate_all(kg)

        emb_tables = sum(1 for t in kg.tables.values() if t.embedding is not None)
        emb_cols = sum(
            1
            for t in kg.tables.values()
            for c in t.columns.values()
            if c.embedding is not None
        )
        logger.info(
            "  → %d/%d table(s) embedded, %d/%d column(s) embedded.",
            emb_tables,
            total_tables,
            emb_cols,
            total_cols,
        )

    def _persist(self, kg: KnowledgeGraph) -> None:
        """
        Stage 4 — persist the KG to PostgreSQL and, when embeddings are
        available, to the Chroma vector store.

        Sub-steps
        ---------
        4a. Upsert schema metadata (kg_metadata, kg_tables, kg_columns,
            kg_relationships) into PostgreSQL.  UUIDs are stabilised via
            ``ON CONFLICT … RETURNING`` so re-running never creates duplicate
            rows and embedding FKs remain valid.
        4b. Upsert embedding vectors into ``kg_table_embeddings`` and
            ``kg_column_embeddings`` in PostgreSQL (skipped when no embeddings
            were generated).
        4c. Upsert all table and column documents into Chroma (skipped when
            no embeddings were generated).
        """
        logger.info("Stage 4/4 — Persisting KG to repository …")

        # 4a + 4b: PostgreSQL --------------------------------------------------
        with KGRepository(self.repo_config) as repo:
            repo.create_schema()

            logger.debug("  4a — Upserting schema metadata to PostgreSQL …")
            repo.save_kg(kg)
            logger.info("  4a — Schema metadata saved: kg_id=%s", kg.kg_id)

            if not self.skip_embeddings:
                table_emb_count = sum(
                    1 for t in kg.tables.values() if t.embedding is not None
                )
                col_emb_count = sum(
                    1
                    for t in kg.tables.values()
                    for c in t.columns.values()
                    if c.embedding is not None
                )
                if table_emb_count > 0 or col_emb_count > 0:
                    logger.debug(
                        "  4b — Upserting %d table and %d column embedding(s) to PostgreSQL …",
                        table_emb_count,
                        col_emb_count,
                    )
                    repo.save_embeddings(kg)
                    logger.info(
                        "  4b — Embeddings saved: %d table(s), %d column(s).",
                        table_emb_count,
                        col_emb_count,
                    )
                else:
                    logger.info(
                        "  4b — No embeddings to save (all embedding fields are None)."
                    )

        # 4c: Chroma -----------------------------------------------------------
        if not self.skip_embeddings:
            logger.debug(
                "  4c — Upserting documents to Chroma at '%s' …",
                self.chroma_config.persist_directory,
            )
            vector_store = ChromaVectorStore(self.chroma_config.persist_directory)
            vector_store.upsert_kg(kg)
            logger.info("  4c — Chroma vector store updated for kg_id=%s.", kg.kg_id)
        else:
            logger.info(
                "  4c — Skipping Chroma upsert (embeddings were not generated)."
            )
