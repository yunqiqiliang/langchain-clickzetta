#!/usr/bin/env python3
"""Manually test our exact CREATE TABLE SQL."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


def test_manual_create():
    """Manually test our exact CREATE TABLE SQL."""
    print("🔄 Testing manual CREATE TABLE with vector index...")

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

        table_name = f"manual_vector_test_{int(time.time())}"

        # This is our exact SQL
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id String,
            content String,
            metadata String,
            embedding vector(float, 16),
            index vector_idx (embedding) using vector properties (
                "scalar.type" = "f32",
                "distance.function" = "cosine_distance"
            )
        )
        """

        print(f"Creating table: {table_name}")
        print("SQL:")
        print(create_sql)
        print()

        try:
            results, columns = engine.execute_query(create_sql)
            print("✅ CREATE TABLE executed without error")
            print(f"   Results: {results}")
            print(f"   Columns: {columns}")
        except Exception as e:
            print(f"❌ CREATE TABLE failed: {e}")
            return False

        # Now verify the table exists using multiple methods
        print("\n🔍 Verifying table existence...")

        # Method 1: SELECT count
        print("Method 1: SELECT COUNT(*)...")
        try:
            results, _ = engine.execute_query(f"SELECT COUNT(*) FROM {table_name}")
            print(f"   ✅ SELECT works: {results}")
            table_exists = True
        except Exception as e:
            print(f"   ❌ SELECT failed: {e}")
            table_exists = False

        # Method 2: SHOW TABLES
        print("Method 2: SHOW TABLES...")
        try:
            results, _ = engine.execute_query("SHOW TABLES")
            found_in_show = any(table_name in str(table.values()) for table in results)
            print(f"   Found in SHOW TABLES: {found_in_show}")
        except Exception as e:
            print(f"   ❌ SHOW TABLES failed: {e}")

        # Method 3: SHOW TABLES LIKE
        print(f"Method 3: SHOW TABLES LIKE '{table_name}'...")
        try:
            results, _ = engine.execute_query(f"SHOW TABLES LIKE '{table_name}'")
            print(f"   LIKE results: {results}")
            found_in_like = len(results) > 0
            print(f"   Found in LIKE: {found_in_like}")
        except Exception as e:
            print(f"   ❌ SHOW TABLES LIKE failed: {e}")

        if table_exists:
            # Test inserting a vector
            print("\n📝 Testing vector insertion...")
            test_vector = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]"
            insert_sql = f"""
            INSERT INTO {table_name} (id, content, metadata, embedding)
            VALUES ('test-1', 'test content', '{{}}', {test_vector})
            """

            print("Insert SQL:")
            print(insert_sql)

            try:
                results, _ = engine.execute_query(insert_sql)
                print("✅ Vector insertion successful")

                # Test vector search
                print("\n🔍 Testing vector search...")
                search_sql = f"""
                SELECT id, content, cosine_distance(embedding, {test_vector}) as distance
                FROM {table_name}
                WHERE cosine_distance(embedding, {test_vector}) < 1.0
                ORDER BY distance
                LIMIT 1
                """

                print("Search SQL:")
                print(search_sql)

                try:
                    results, _ = engine.execute_query(search_sql)
                    print(f"✅ Vector search successful: {results}")
                    return True
                except Exception as e:
                    print(f"❌ Vector search failed: {e}")
                    return False

            except Exception as e:
                print(f"❌ Vector insertion failed: {e}")
                return False
        else:
            print("❌ Table doesn't exist, cannot test insertion")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_manual_create()
    print("✅ Test completed successfully!" if success else "❌ Test failed")
