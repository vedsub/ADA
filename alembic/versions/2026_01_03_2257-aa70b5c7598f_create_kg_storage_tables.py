"""create_kg_storage_tables

Revision ID: aa70b5c7598f
Revises: 
Create Date: 2026-01-03 22:57:26.311126

"""
from typing import Sequence, Union
from sqlalchemy.dialects import postgresql
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aa70b5c7598f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all knowledge graph storage tables."""
    
    # 1. Create kg_metadata table
    op.create_table(
        'kg_metadata',
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('source_db_host', sa.String(255), nullable=False),
        sa.Column('source_db_port', sa.Integer(), nullable=False),
        sa.Column('source_db_name', sa.String(255), nullable=False),
        sa.Column('source_db_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('allowed_tables', postgresql.JSONB(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('status', sa.String(50), nullable=False, server_default='building'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('build_duration_seconds', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
    )
    
    # Create index on source_db_hash for fast lookup
    op.create_index('idx_kg_metadata_source_hash', 'kg_metadata', ['source_db_hash'])
    op.create_index('idx_kg_metadata_status', 'kg_metadata', ['status'])
    
    # 2. Create kg_tables table
    op.create_table(
        'kg_tables',
        sa.Column('table_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('table_name', sa.String(255), nullable=False),
        sa.Column('schema_name', sa.String(255), nullable=False, server_default='public'),
        sa.Column('qualified_name', sa.String(512), nullable=False),  # schema.table
        sa.Column('table_type', sa.String(50), nullable=False, server_default='base_table'),
        sa.Column('row_count_estimate', sa.BigInteger(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('business_domain', sa.String(255), nullable=True),
        sa.Column('typical_use_cases', postgresql.JSONB(), nullable=True),
        sa.Column('data_sensitivity', sa.String(50), nullable=True),
        sa.Column('update_frequency', sa.String(50), nullable=True),
        sa.Column('owner_team', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['kg_id'], ['kg_metadata.kg_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('kg_id', 'qualified_name', name='uq_kg_tables_kg_qualified_name')
    )
    
    # Create indexes for kg_tables
    op.create_index('idx_kg_tables_kg_id', 'kg_tables', ['kg_id'])
    op.create_index('idx_kg_tables_qualified_name', 'kg_tables', ['qualified_name'])
    op.create_index('idx_kg_tables_business_domain', 'kg_tables', ['business_domain'])
    
    # 3. Create kg_columns table
    op.create_table(
        'kg_columns',
        sa.Column('column_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('table_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('column_name', sa.String(255), nullable=False),
        sa.Column('qualified_name', sa.String(512), nullable=False),  # table.column
        sa.Column('data_type', sa.String(255), nullable=False),
        sa.Column('is_nullable', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_primary_key', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_unique', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_foreign_key', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('column_position', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('business_meaning', sa.Text(), nullable=True),
        sa.Column('sample_values', postgresql.JSONB(), nullable=True),
        sa.Column('enum_values', postgresql.JSONB(), nullable=True),
        sa.Column('value_format', sa.String(255), nullable=True),
        sa.Column('cardinality', sa.String(50), nullable=True),  # low, medium, high
        sa.Column('null_percentage', sa.Numeric(5, 2), nullable=True),  # 0.00 to 100.00
        sa.Column('typical_filters', postgresql.JSONB(), nullable=True),
        sa.Column('is_pii', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['table_id'], ['kg_tables.table_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('table_id', 'column_name', name='uq_kg_columns_table_column')
    )
    
    # Create indexes for kg_columns
    op.create_index('idx_kg_columns_table_id', 'kg_columns', ['table_id'])
    op.create_index('idx_kg_columns_qualified_name', 'kg_columns', ['qualified_name'])
    op.create_index('idx_kg_columns_is_pii', 'kg_columns', ['is_pii'])
    
    # 4. Create kg_relationships table
    op.create_table(
        'kg_relationships',
        sa.Column('relationship_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_table_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('to_table_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_column', sa.String(255), nullable=False),
        sa.Column('to_column', sa.String(255), nullable=False),
        sa.Column('relationship_type', sa.String(50), nullable=False),  # many-to-one, one-to-many, many-to-many, self-reference
        sa.Column('constraint_name', sa.String(255), nullable=True),
        sa.Column('join_condition', sa.Text(), nullable=False),
        sa.Column('business_meaning', sa.Text(), nullable=True),
        sa.Column('join_frequency', sa.Numeric(3, 2), nullable=True),  # 0.00 to 1.00
        sa.Column('is_self_reference', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('cascade_on_delete', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['kg_id'], ['kg_metadata.kg_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['from_table_id'], ['kg_tables.table_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['to_table_id'], ['kg_tables.table_id'], ondelete='CASCADE'),
    )
    
    # Create indexes for kg_relationships
    op.create_index('idx_kg_relationships_kg_id', 'kg_relationships', ['kg_id'])
    op.create_index('idx_kg_relationships_from_table', 'kg_relationships', ['from_table_id'])
    op.create_index('idx_kg_relationships_to_table', 'kg_relationships', ['to_table_id'])
    op.create_index('idx_kg_relationships_type', 'kg_relationships', ['relationship_type'])
    
    # 5. Create kg_embeddings table
    op.create_table(
        'kg_embeddings',
        sa.Column('embedding_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=False),  # table, column
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('embedding_text', sa.Text(), nullable=False),
        sa.Column('embedding_vector', postgresql.BYTEA(), nullable=True),  # Store as bytes, use pgvector if available
        sa.Column('embedding_model', sa.String(100), nullable=False, server_default='text-embedding-3-small'),
        sa.Column('vector_dimension', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['kg_id'], ['kg_metadata.kg_id'], ondelete='CASCADE'),
        sa.UniqueConstraint('entity_type', 'entity_id', name='uq_kg_embeddings_entity')
    )
    
    # Create indexes for kg_embeddings
    op.create_index('idx_kg_embeddings_kg_id', 'kg_embeddings', ['kg_id'])
    op.create_index('idx_kg_embeddings_entity', 'kg_embeddings', ['entity_type', 'entity_id'])
    
    # 6. Create kg_query_log table (optional but valuable for learning)
    op.create_table(
        'kg_query_log',
        sa.Column('query_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('kg_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_question', sa.Text(), nullable=False),
        sa.Column('generated_sql', sa.Text(), nullable=False),
        sa.Column('execution_success', sa.Boolean(), nullable=False),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('tables_used', postgresql.JSONB(), nullable=True),
        sa.Column('correction_applied', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('iterations_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['kg_id'], ['kg_metadata.kg_id'], ondelete='CASCADE'),
    )
    
    # Create indexes for kg_query_log
    op.create_index('idx_kg_query_log_kg_id', 'kg_query_log', ['kg_id'])
    op.create_index('idx_kg_query_log_success', 'kg_query_log', ['execution_success'])
    op.create_index('idx_kg_query_log_created_at', 'kg_query_log', ['created_at'])
    
    # 7. Create trigger to update updated_at timestamp
    # This is PostgreSQL-specific
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Apply trigger to tables with updated_at
    for table_name in ['kg_metadata', 'kg_tables', 'kg_columns', 'kg_relationships']:
        op.execute(f"""
            CREATE TRIGGER update_{table_name}_updated_at
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    """Drop all knowledge graph storage tables."""
    
    # Drop triggers first
    for table_name in ['kg_metadata', 'kg_tables', 'kg_columns', 'kg_relationships']:
        op.execute(f"DROP TRIGGER IF EXISTS update_{table_name}_updated_at ON {table_name}")
    
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('kg_query_log')
    op.drop_table('kg_embeddings')
    op.drop_table('kg_relationships')
    op.drop_table('kg_columns')
    op.drop_table('kg_tables')
    op.drop_table('kg_metadata')
