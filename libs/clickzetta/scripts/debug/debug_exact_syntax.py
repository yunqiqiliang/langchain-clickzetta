#!/usr/bin/env python3
"""Debug exact table creation syntax."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


def test_exact_syntax():
    """Test the exact syntax we're using in vector store."""
    print("🔄 Testing exact vector store syntax...")

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

        table_name = f"test_exact_syntax_{int(time.time())}"

        # This is the exact SQL our vector store is generating
        create_table_sql = f"""
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

        print(f"Creating table with exact vector store syntax: {table_name}")
        print(f"SQL: {create_table_sql}")

        try:
            results, _ = engine.execute_query(create_table_sql)
            print("✅ Table with vector index created successfully!")

            # Verify table exists
            check_sql = f"SHOW TABLES LIKE '{table_name}'"
            results, _ = engine.execute_query(check_sql)
            print(f"Table verification result: {results}")

            if results:
                print("✅ Table verified to exist!")

                # Test inserting a vector
                test_vector_str = "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]"
                insert_sql = f"""
                INSERT INTO {table_name} (id, content, metadata, embedding)
                VALUES ('test-1', 'test content', '{{}}', {test_vector_str})
                """

                print("Testing vector insertion...")
                results, _ = engine.execute_query(insert_sql)
                print("✅ Vector insertion successful!")

                # Test vector search
                search_sql = f"""
                SELECT id, content, cosine_distance(embedding, {test_vector_str}) as distance
                FROM {table_name}
                WHERE cosine_distance(embedding, {test_vector_str}) < 1.0
                ORDER BY distance
                LIMIT 1
                """

                print("Testing vector search...")
                results, _ = engine.execute_query(search_sql)
                print(f"Vector search result: {results}")

                if results:
                    print("✅ Vector search successful!")
                    return True
                else:
                    print("⚠️  Vector search returned no results, but table creation worked")
                    return True
            else:
                print("❌ Table was not created")
                return False

        except Exception as e:
            print(f"❌ Table creation failed: {e}")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_syntax()
    print("✅ Test completed" if success else "❌ Test failed")
