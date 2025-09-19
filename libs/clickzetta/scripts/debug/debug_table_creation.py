#!/usr/bin/env python3
"""Debug table creation issue."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


def test_table_creation():
    """Test table creation specifically."""
    print("🔄 Testing table creation...")

    # Load UAT connection
    config_path = Path.home() / ".clickzetta" / "connections.json"
    with open(config_path, encoding='utf-8') as f:
        config_data = json.load(f)

    uat_config = None
    for conn in config_data.get("connections", []):
        if conn.get("name") == "uat":
            uat_config = conn
            break

    if not uat_config:
        print("❌ UAT connection not found")
        return False

    try:
        engine = ClickZettaEngine(**uat_config)
        print("✅ Engine created")

        # Test 1: Create a simple table first without vector index
        simple_table_name = f"langchain_simple_test_{int(time.time())}"
        simple_sql = f"""
        CREATE TABLE IF NOT EXISTS {simple_table_name} (
            id String,
            content String,
            metadata String
        )
        """

        print(f"Creating simple table: {simple_table_name}")
        print(f"SQL: {simple_sql}")

        results, _ = engine.execute_query(simple_sql)
        print("✅ Simple table creation executed")

        # Verify simple table exists
        check_simple_sql = f"SHOW TABLES LIKE '{simple_table_name}'"
        results, _ = engine.execute_query(check_simple_sql)
        print(f"Simple table check result: {results}")

        if not results:
            print("❌ Simple table was not created")
            return False

        print("✅ Simple table verified")

        # Test 2: Try creating vector table with corrected syntax
        vector_table_name = f"langchain_vector_test_{int(time.time())}"

        # Start with the simplest possible vector table
        vector_sql = f"""
        CREATE TABLE IF NOT EXISTS {vector_table_name} (
            id String,
            content String,
            metadata String,
            embedding vector(float, 16)
        )
        """

        print(f"Creating vector table: {vector_table_name}")
        print(f"SQL: {vector_sql}")

        results, _ = engine.execute_query(vector_sql)
        print("✅ Vector table creation executed")

        # Verify vector table exists
        check_vector_sql = f"SHOW TABLES LIKE '{vector_table_name}'"
        results, _ = engine.execute_query(check_vector_sql)
        print(f"Vector table check result: {results}")

        if not results:
            print("❌ Vector table was not created")
            return False

        print("✅ Vector table verified")

        # Test 3: Try adding the vector index separately
        index_sql = f"""
        ALTER TABLE {vector_table_name}
        ADD index vector_idx (embedding) using vector properties (
            "scalar.type" = "f32",
            "distance.function" = "cosine_distance"
        )
        """

        print("Adding vector index")
        print(f"SQL: {index_sql}")

        try:
            results, _ = engine.execute_query(index_sql)
            print("✅ Vector index creation executed")
        except Exception as e:
            print(f"⚠️  Vector index creation failed: {e}")
            print("This might be expected - let's try a different approach")

        # Test 4: Insert a test row to see if the vector table is functional
        test_vector_str = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]"
        insert_sql = f"""
        INSERT INTO {vector_table_name} (id, content, metadata, embedding)
        VALUES ('test-1', 'test content', '{{}}', {test_vector_str})
        """

        print("Inserting test row")
        print(f"SQL: {insert_sql}")

        try:
            results, _ = engine.execute_query(insert_sql)
            print("✅ Test row insertion executed")

            # Verify the row was inserted
            select_sql = f"SELECT COUNT(*) as count FROM {vector_table_name}"
            results, _ = engine.execute_query(select_sql)
            print(f"Row count after insert: {results}")

        except Exception as e:
            print(f"❌ Test row insertion failed: {e}")
            return False

        print("✅ All table creation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_table_creation()
    print("✅ Test completed" if success else "❌ Test failed")
