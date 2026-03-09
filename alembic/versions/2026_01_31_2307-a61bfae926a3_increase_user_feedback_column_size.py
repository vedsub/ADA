"""increase user_feedback column size

Revision ID: a61bfae926a3
Revises: 3778a3c2bf82
Create Date: 2026-01-31 23:07:55.839450

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a61bfae926a3'
down_revision: Union[str, Sequence[str], None] = '3778a3c2bf82'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Change user_feedback from varchar(20) to TEXT"""
    op.execute("""
        ALTER TABLE kg_query_log 
        ALTER COLUMN user_feedback TYPE TEXT
    """)

def downgrade() -> None:
    """Revert to varchar(20)"""
    op.execute("""
        ALTER TABLE kg_query_log 
        ALTER COLUMN user_feedback TYPE VARCHAR(20)
    """)
