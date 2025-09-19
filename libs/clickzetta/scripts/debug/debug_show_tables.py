#!/usr/bin/env python3
"""Debug SHOW TABLES syntax."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


def test_show_tables_syntax():
    """Test different SHOW TABLES syntaxes."""
    print("🔄 Testing SHOW TABLES syntax...")

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

        # Test 1: Show all tables
        print("1. SHOW TABLES:")
        results, _ = engine.execute_query("SHOW TABLES")
        print(f"   Found {len(results)} tables")
        for table in results[:5]:  # Show first 5
            print(f"   - {table}")

        # Test 2: Show tables with LIKE pattern for existing table
        if results:
            existing_table = list(results[0].values())[0]  # Get first table name
            print(f"\n2. SHOW TABLES LIKE '{existing_table}':")
            like_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{existing_table}'")
            print(f"   Results: {like_results}")

        # Test 3: Create a simple table and verify different ways
        test_table = f"debug_show_test_{int(time.time())}"
        print(f"\n3. Creating test table: {test_table}")

        create_sql = f"CREATE TABLE IF NOT EXISTS {test_table} (id String, name String)"
        engine.execute_query(create_sql)

        # Try different verification methods
        print("   Verifying with SHOW TABLES:")
        results, _ = engine.execute_query("SHOW TABLES")
        found = any(test_table in str(table.values()) for table in results)
        print(f"   Found in SHOW TABLES: {found}")

        print(f"   Verifying with SHOW TABLES LIKE '{test_table}':")
        like_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{test_table}'")
        print(f"   LIKE results: {like_results}")

        print("   Verifying with SELECT from table:")
        try:
            select_results, _ = engine.execute_query(f"SELECT COUNT(*) FROM {test_table}")
            print(f"   SELECT results: {select_results}")
            print("   ✅ Table exists and is accessible")
        except Exception as e:
            print(f"   ❌ SELECT failed: {e}")

        # Test 4: Try the vector table creation and verify
        print("\n4. Testing vector table creation...")
        vector_table = f"debug_vector_test_{int(time.time())}"

        # First, try without vector index
        print("   Creating vector table without index:")
        vector_sql_no_index = f"""
        CREATE TABLE IF NOT EXISTS {vector_table} (
            id String,
            content String,
            metadata String,
            embedding vector(float, 16)
        )
        """
        engine.execute_query(vector_sql_no_index)

        # Check if it exists
        try:
            select_results, _ = engine.execute_query(f"SELECT COUNT(*) FROM {vector_table}")
            print(f"   Vector table (no index) exists: {select_results}")

            # Now try to add index in a separate statement
            print("   Attempting to add vector index...")
            # Try different index creation syntaxes
            index_syntaxes = [
                f"CREATE INDEX vector_idx ON {vector_table} (embedding) USING VECTOR",
                f"ALTER TABLE {vector_table} ADD INDEX vector_idx (embedding) USING VECTOR",
            ]

            for i, index_sql in enumerate(index_syntaxes):
                print(f"   Syntax {i+1}: {index_sql[:50]}...")
                try:
                    engine.execute_query(index_sql)
                    print(f"   ✅ Index syntax {i+1} worked!")
                except Exception as e:
                    print(f"   ❌ Index syntax {i+1} failed: {str(e)[:100]}...")

        except Exception as e:
            print(f"   ❌ Vector table (no index) doesn't exist: {e}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_show_tables_syntax()
