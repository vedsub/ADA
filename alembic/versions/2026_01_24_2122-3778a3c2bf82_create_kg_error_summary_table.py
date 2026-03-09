"""create kg_error_summary table

Revision ID: 3778a3c2bf82
Revises: 42ff66ad5775
Create Date: 2026-01-24 21:22:16.879726

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '3778a3c2bf82'
down_revision: Union[str, Sequence[str], None] = '42ff66ad5775'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'kg_error_summary',
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('schema_lessons', sa.Text(), server_default='', nullable=False),
        sa.Column('sql_lessons', sa.Text(), server_default='', nullable=False),
        sa.Column('lesson_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('word_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('compression_threshold', sa.Integer(), server_default='500', nullable=False),
        sa.Column('last_compressed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_updated', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('kg_id'),
        sa.ForeignKeyConstraint(['kg_id'], ['kg_metadata.kg_id'], ondelete='CASCADE')
    )
    
    op.create_index('idx_kg_error_summary_updated', 'kg_error_summary', ['last_updated'])


def downgrade() -> None:
    op.drop_index('idx_kg_error_summary_updated', table_name='kg_error_summary')
    op.drop_table('kg_error_summary')