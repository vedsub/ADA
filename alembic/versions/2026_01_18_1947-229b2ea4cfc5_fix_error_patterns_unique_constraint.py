"""fix_error_patterns_unique_constraint

Revision ID: 229b2ea4cfc5
Revises: b730beacbefb
Create Date: 2026-01-18 19:47:26.588461

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '229b2ea4cfc5'
down_revision: Union[str, Sequence[str], None] = 'b730beacbefb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add unique constraint on kg_id and error_pattern"""
    op.create_unique_constraint(
        'uq_query_error_patterns_kg_pattern',
        'query_error_patterns',
        ['kg_id', 'error_pattern']
    )

def downgrade() -> None:
    """Remove unique constraint"""
    op.drop_constraint(
        'uq_query_error_patterns_kg_pattern',
        'query_error_patterns',
        type_='unique'
    )
