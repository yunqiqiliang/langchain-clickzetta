"""Cleanup utility for ClickZetta LangChain examples."""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain_clickzetta import ClickZettaEngine


def cleanup_test_data():
    """Clean up test data created by examples."""
    # Initialize ClickZetta engine
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster")
    )

    print("=== ClickZetta LangChain Cleanup ===\n")

    # Tables created by examples
    tables_to_clean = [
        "example_vectors",
        "example_chat_history",
        "example_documents",
        "example_hybrid_docs",
        "rag_documents",
        "rag_hybrid_docs",
        "rag_conversations"
    ]

    # Add workspace.schema prefix to table names
    workspace = engine.connection_config['workspace']
    schema = engine.connection_config['schema']

    # Clean up tables
    for table_name in tables_to_clean:
        full_table_name = f"{workspace}.{schema}.{table_name}"
        try:
            # Directly drop table (IF EXISTS handles non-existent tables)
            drop_sql = f"DROP TABLE IF EXISTS {full_table_name}"
            engine.execute_query(drop_sql)
            print(f"✓ Dropped table: {full_table_name}")

        except Exception as e:
            print(f"✗ Error cleaning table {full_table_name}: {e}")

    print("\n" + "="*50)

    # Clean up indexes (they have unique names based on table hash)
    print("\nCleaning up indexes...")

    try:
        # Since SHOW INDEXES has syntax issues, we'll try to drop specific indexes directly
        # Based on our naming pattern with table hashes
        print("Attempting to clean up indexes by pattern...")

        # We can't reliably list all indexes, so we'll skip index cleanup
        # Users can manually drop indexes if needed using: DROP INDEX index_name
        print("- Index cleanup skipped (use 'DROP INDEX index_name' manually if needed)")
        print("- Example index patterns: content_fts_*, embedding_idx_*")

    except Exception as e:
        print(f"✗ Error with index cleanup: {e}")

    # Close connection
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")

    print("\n=== Cleanup completed ===")


if __name__ == "__main__":
    cleanup_test_data()
