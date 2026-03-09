"""add_vector_similarity_index

Revision ID: 42ff66ad5775
Revises: 229b2ea4cfc5
Create Date: 2026-01-18 19:48:58.748991

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '42ff66ad5775'
down_revision: Union[str, Sequence[str], None] = '229b2ea4cfc5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add vector similarity search index for query_embedding"""
    
    # Create HNSW index for fast approximate nearest neighbor search
    op.execute("""
        DO $$
        BEGIN
            -- Check if there are any non-null embeddings
            IF EXISTS (
                SELECT 1 FROM kg_query_log 
                WHERE query_embedding IS NOT NULL 
                LIMIT 1
            ) THEN
                -- Create index for cosine distance
                CREATE INDEX IF NOT EXISTS idx_kg_query_log_embedding_cosine
                ON kg_query_log 
                USING hnsw (query_embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                
                RAISE NOTICE 'Created vector similarity index';
            ELSE
                RAISE NOTICE 'No embeddings found, skipping index creation';
            END IF;
        END $$;
    """)


def downgrade() -> None:
    """Remove vector similarity search index"""
    op.execute("DROP INDEX IF EXISTS idx_kg_query_log_embedding_cosine")
