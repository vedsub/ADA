from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .relationship import Relationship
from .table import Table


class KnowledgeGraph(BaseModel):
    """
    Complete knowledge graph for a source database.
    """

    kg_id: UUID = Field(default_factory=uuid4)
    source_db_host: str
    source_db_port: int
    source_db_name: str
    source_db_hash: str
    status: str = "building"  # building, ready, error
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    # In-memory structures
    tables: Dict[str, Table] = Field(default_factory=dict)  # table_name -> Table
    relationships: List[Relationship] = Field(default_factory=list)
    table_lookup: Dict[UUID, str] = Field(
        default_factory=dict
    )  # table_id -> table_name

    class Config:
        frozen = False

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def get_table(self, table_name: str) -> Optional[Table]:
        """Return the Table with the given name, or None if not found."""
        return self.tables.get(table_name)

    def get_table_by_id(self, table_id: UUID) -> Optional[Table]:
        """Return the Table with the given UUID, or None if not found."""
        table_name = self.table_lookup.get(table_id)
        return self.tables.get(table_name) if table_name else None

    def add_table(self, table: Table) -> None:
        """Register a Table in both the name-keyed dict and the UUID lookup."""
        self.tables[table.table_name] = table
        self.table_lookup[table.table_id] = table.table_name

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    def add_relationship(self, relationship: Relationship) -> None:
        """Append a Relationship to the graph."""
        self.relationships.append(relationship)

    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """Return all Relationships where the given table appears on either side."""
        return [
            rel
            for rel in self.relationships
            if rel.from_table_name == table_name or rel.to_table_name == table_name
        ]
