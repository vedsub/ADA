"""add_pgvector_to_query_log

Revision ID: b730beacbefb
Revises: a281dc260eac
Create Date: 2026-01-13 22:46:22.949537

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b730beacbefb'
down_revision: Union[str, Sequence[str], None] = 'a281dc260eac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade kg_query_log table to use pgvector for query_embedding.
    """
    
    # Step 1: Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Step 2: Convert query_embedding from bytea to vector(1536)
    
    # Drop the column and recreate with vector type
    op.drop_column('kg_query_log', 'query_embedding')
    op.add_column(
        'kg_query_log',
        sa.Column('query_embedding', sa.Text(), nullable=True)
    )
    
    # Use raw SQL to alter the column to vector type
    # (Alembic doesn't have native vector type support)
    op.execute("""
        ALTER TABLE kg_query_log 
        ALTER COLUMN query_embedding TYPE vector(1536) 
        USING query_embedding::vector
    """)

def downgrade() -> None:
    """
    Revert back to bytea type.
    """
    
    # Convert back to bytea
    op.execute("""
        ALTER TABLE kg_query_log 
        ALTER COLUMN query_embedding TYPE bytea 
        USING query_embedding::text::bytea
    """)
