"""add_query_memory_enhancements

Revision ID: a281dc260eac
Revises: aa70b5c7598f
Create Date: 2026-01-08 15:52:19.605124

"""
from typing import Sequence, Union
from sqlalchemy.dialects import postgresql
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a281dc260eac'
down_revision: Union[str, Sequence[str], None] = 'aa70b5c7598f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add query memory enhancements to kg_query_log and create query_error_patterns table."""
    
    # Add query_embedding column for semantic similarity search
    op.add_column(
        'kg_query_log',
        sa.Column(
            'query_embedding',
            postgresql.BYTEA(),
            nullable=True,
            comment='Embedded representation of user query for semantic search'
        )
    )
    
    # Add intent_summary column for condensed query understanding
    op.add_column(
        'kg_query_log',
        sa.Column(
            'intent_summary',
            sa.Text(),
            nullable=True,
            comment='Condensed summary of query intent (e.g., "Find customers by product and date")'
        )
    )
    
    # Add selected_tables column to track Agent 1's table selection
    op.add_column(
        'kg_query_log',
        sa.Column(
            'selected_tables',
            postgresql.JSONB(),
            nullable=True,
            comment='Array of table names selected by schema selector agent'
        )
    )
    
    # Add error_category column for error classification
    op.add_column(
        'kg_query_log',
        sa.Column(
            'error_category',
            sa.String(50),
            nullable=True,
            comment='Error type: syntax_error, column_not_found, table_not_found, permission_denied, timeout, logic_error, data_error'
        )
    )
    
    op.add_column(
    'kg_query_log',
    sa.Column(
        'refined_query',
        sa.Text(),
        nullable=True,
        comment='User query after applying clarifications'
    )
)
    
    # Add correction_summary column to track fixes
    op.add_column(
        'kg_query_log',
        sa.Column(
            'correction_summary',
            sa.Text(),
            nullable=True,
            comment='Description of what correction was applied to fix the error'
        )
    )
    
    # Add schema_retrieval_time_ms for performance tracking
    op.add_column(
        'kg_query_log',
        sa.Column(
            'schema_retrieval_time_ms',
            sa.Integer(),
            nullable=True,
            comment='Time taken to retrieve schema context from KG (Agent 1)'
        )
    )
    
    # Add sql_generation_time_ms for performance tracking
    op.add_column(
        'kg_query_log',
        sa.Column(
            'sql_generation_time_ms',
            sa.Integer(),
            nullable=True,
            comment='Time taken to generate SQL (Agent 2)'
        )
    )
    
    # Add confidence_score for SQL generation confidence
    op.add_column(
        'kg_query_log',
        sa.Column(
            'confidence_score',
            sa.Numeric(3, 2),
            nullable=True,
            comment='Confidence score from 0.00 to 1.00 for generated SQL'
        )
    )
    
    # Add user_feedback for learning
    op.add_column(
        'kg_query_log',
        sa.Column(
            'user_feedback',
            sa.String(20),
            nullable=True,
            comment='User feedback: helpful, not_helpful, incorrect'
        )
    )
    

    # Index on error_category for fast error pattern analysis
    op.create_index(
        'idx_kg_query_log_error_category',
        'kg_query_log',
        ['error_category']
    )
    
    # Index on user_feedback for quality analysis
    op.create_index(
        'idx_kg_query_log_user_feedback',
        'kg_query_log',
        ['user_feedback']
    )
    
    # Composite index for finding similar successful queries
    op.create_index(
        'idx_kg_query_log_success_created',
        'kg_query_log',
        ['execution_success', 'created_at']
    )
    
    
    op.create_table(
        'query_error_patterns',
        sa.Column(
            'pattern_id',
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text('gen_random_uuid()'),
            comment='Unique identifier for error pattern'
        ),
        sa.Column(
            'kg_id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            comment='Reference to knowledge graph'
        ),
        sa.Column(
            'error_category',
            sa.String(50),
            nullable=False,
            comment='Category of error: syntax_error, column_not_found, table_not_found, etc.'
        ),
        sa.Column(
            'error_pattern',
            sa.Text(),
            nullable=False,
            comment='Description of the error pattern (e.g., "Always forgets table prefix in column names")'
        ),
        sa.Column(
            'example_error_message',
            sa.Text(),
            nullable=True,
            comment='Example error message from database'
        ),
        sa.Column(
            'fix_applied',
            sa.Text(),
            nullable=False,
            comment='Description of fix that resolved this error (e.g., "Added table prefix to column names")'
        ),
        sa.Column(
            'affected_tables',
            postgresql.JSONB(),
            nullable=True,
            comment='List of tables commonly involved in this error pattern'
        ),
        sa.Column(
            'occurrence_count',
            sa.Integer(),
            nullable=False,
            server_default='1',
            comment='Number of times this pattern has occurred'
        ),
        sa.Column(
            'success_rate_after_fix',
            sa.Numeric(5, 2),
            nullable=True,
            comment='Percentage of queries that succeeded after applying this fix (0.00 to 100.00)'
        ),
        sa.Column(
            'first_seen',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
            comment='When this pattern was first detected'
        ),
        sa.Column(
            'last_seen',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP'),
            comment='Most recent occurrence of this pattern'
        ),
        sa.Column(
            'is_active',
            sa.Boolean(),
            nullable=False,
            server_default='true',
            comment='Whether this pattern is still being monitored'
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP')
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('CURRENT_TIMESTAMP')
        ),
        sa.ForeignKeyConstraint(
            ['kg_id'],
            ['kg_metadata.kg_id'],
            ondelete='CASCADE'
        )
    )
    
    
    # Index on kg_id for fast lookup by knowledge graph
    op.create_index(
        'idx_query_error_patterns_kg_id',
        'query_error_patterns',
        ['kg_id']
    )
    
    # Index on error_category for grouping by error type
    op.create_index(
        'idx_query_error_patterns_category',
        'query_error_patterns',
        ['error_category']
    )
    
    # Index on occurrence_count for finding most common patterns
    op.create_index(
        'idx_query_error_patterns_occurrence',
        'query_error_patterns',
        ['occurrence_count']
    )
    
    # Index on is_active for filtering active patterns
    op.create_index(
        'idx_query_error_patterns_active',
        'query_error_patterns',
        ['is_active']
    )
    
    # Composite index for finding recent active patterns
    op.create_index(
        'idx_query_error_patterns_active_last_seen',
        'query_error_patterns',
        ['is_active', 'last_seen']
    )
    
    
    op.execute("""
        CREATE TRIGGER update_query_error_patterns_updated_at
        BEFORE UPDATE ON query_error_patterns
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)
    
    
    op.execute("COMMENT ON COLUMN kg_query_log.user_question IS 'Natural language query from user'")
    op.execute("COMMENT ON COLUMN kg_query_log.generated_sql IS 'SQL query generated by Agent 2'")
    op.execute("COMMENT ON COLUMN kg_query_log.execution_success IS 'Whether SQL executed successfully'")
    op.execute("COMMENT ON COLUMN kg_query_log.execution_time_ms IS 'Time taken to execute SQL on source database'")
    op.execute("COMMENT ON COLUMN kg_query_log.error_message IS 'Error message if execution failed'")
    op.execute("COMMENT ON COLUMN kg_query_log.tables_used IS 'Array of table names used in final SQL query'")
    op.execute("COMMENT ON COLUMN kg_query_log.correction_applied IS 'Whether this query needed correction (retry)'")
    op.execute("COMMENT ON COLUMN kg_query_log.iterations_count IS 'Number of retry attempts before success/failure'")


def downgrade() -> None:
    """Remove query memory enhancements."""
    
    # Drop trigger
    op.execute("DROP TRIGGER IF EXISTS update_query_error_patterns_updated_at ON query_error_patterns")
    
    # Drop indexes on query_error_patterns
    op.drop_index('idx_query_error_patterns_active_last_seen', 'query_error_patterns')
    op.drop_index('idx_query_error_patterns_active', 'query_error_patterns')
    op.drop_index('idx_query_error_patterns_occurrence', 'query_error_patterns')
    op.drop_index('idx_query_error_patterns_category', 'query_error_patterns')
    op.drop_index('idx_query_error_patterns_kg_id', 'query_error_patterns')
    
    # Drop query_error_patterns table
    op.drop_table('query_error_patterns')
    
    # Drop indexes on kg_query_log
    op.drop_index('idx_kg_query_log_success_created', 'kg_query_log')
    op.drop_index('idx_kg_query_log_user_feedback', 'kg_query_log')
    op.drop_index('idx_kg_query_log_error_category', 'kg_query_log')
    
    # Drop new columns from kg_query_log (preserves other data)
    op.drop_column('kg_query_log', 'user_feedback')
    op.drop_column('kg_query_log', 'confidence_score')
    op.drop_column('kg_query_log', 'sql_generation_time_ms')
    op.drop_column('kg_query_log', 'schema_retrieval_time_ms')
    op.drop_column('kg_query_log', 'correction_summary')
    op.drop_column('kg_query_log', 'refined_query')
    op.drop_column('kg_query_log', 'error_category')
    op.drop_column('kg_query_log', 'selected_tables')
    op.drop_column('kg_query_log', 'intent_summary')
    op.drop_column('kg_query_log', 'query_embedding')
