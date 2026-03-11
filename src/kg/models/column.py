from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Column(BaseModel):
    """
    Represents a database column with metadata.
    """

    column_id: UUID = Field(default_factory=uuid4)
    table_id: UUID
    column_name: str
    qualified_name: str  # table.column
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_unique: bool = False
    is_foreign_key: bool = False
    column_position: Optional[int] = None
    description: Optional[str] = None
    business_meaning: Optional[str] = None
    sample_values: Optional[List[str]] = None
    enum_values: Optional[List[str]] = None
    cardinality: Optional[str] = None  # low, medium, high
    null_percentage: Optional[float] = None
    is_pii: bool = False
    embedding: Optional[List[float]] = None  # text-embedding-3-small vector (1536-d)

    class Config:
        frozen = False
