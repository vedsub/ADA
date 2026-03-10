from typing import TYPE_CHECKING, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .column import Column


class Table(BaseModel):
    """
    Represents a database table with metadata.
    """

    table_id: UUID = Field(default_factory=uuid4)
    kg_id: UUID
    table_name: str
    schema_name: str = "public"
    qualified_name: str  # schema.table
    table_type: str = "base_table"
    row_count_estimate: Optional[int] = None
    description: Optional[str] = None
    business_domain: Optional[str] = None
    typical_use_cases: Optional[List[str]] = None

    columns: Dict[str, "Column"] = Field(
        default_factory=dict
    )  # For in-memory representation

    class Config:
        frozen = False
