from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Relationship(BaseModel):
    """
    Represents a foreign key relationship between tables.
    """

    relationship_id: UUID = Field(default_factory=uuid4)
    kg_id: UUID
    from_table_id: UUID
    to_table_id: UUID
    from_table_name: str
    to_table_name: str
    from_column: str
    to_column: str
    relationship_type: str  # many-to-one, one-to-many, one-to-one
    constraint_name: Optional[str] = None
    join_condition: str  # SQL: orders.customer_id = customers.id
    business_meaning: Optional[str] = None
    is_self_reference: bool = False

    class Config:
        frozen = False
