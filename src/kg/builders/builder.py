from __future__ import annotations

from abc import ABC, abstractmethod

from src.kg.models.knowledge_graph import KnowledgeGraph


class KGBuilderBase(ABC):
    """
    Abstract base class for knowledge graph builders.

    Subclasses implement the :meth:`build` method to run a specific
    extraction and enrichment pipeline against a source database and
    return a fully-populated :class:`~src.kg.models.knowledge_graph.KnowledgeGraph`.

    Design contract
    ---------------
    * :meth:`build` **must** return a :class:`KnowledgeGraph` whose
      ``status`` field is ``"ready"`` on success.
    * Implementations are free to skip pipeline stages (e.g. description
      generation, embedding generation) when instructed via constructor flags,
      but the returned KG must always contain at minimum the extracted schema
      (tables, columns, relationships).
    * :meth:`build` is idempotent when ``force_rebuild=False``: calling it
      multiple times on the same source database must not duplicate data in
      any backing store.
    """

    @abstractmethod
    def build(self, force_rebuild: bool = False) -> KnowledgeGraph:
        """
        Build and return a fully-populated :class:`KnowledgeGraph`.

        Parameters
        ----------
        force_rebuild:
            When ``True``, bypass any existing cached / persisted KG and
            run the full pipeline from scratch (re-extract schema, re-generate
            descriptions and embeddings, overwrite all stored data).
            When ``False`` (default), return the cached KG if one already
            exists for the configured source database.

        Returns
        -------
        KnowledgeGraph
            A fully-populated graph with ``status == "ready"``.

        Raises
        ------
        Exception
            Any unrecoverable error encountered during extraction,
            generation, or persistence.  The KG's ``status`` will be
            ``"error"`` if the object was partially constructed before
            the failure.
        """
        ...
