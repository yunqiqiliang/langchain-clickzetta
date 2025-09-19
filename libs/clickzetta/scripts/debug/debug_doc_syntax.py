#!/usr/bin/env python3
"""Test with exact documentation syntax."""

import json
import sys
import time
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_clickzetta import ClickZettaEngine


def test_doc_syntax():
    """Test with exact ClickZetta documentation syntax."""
    print("🔄 Testing with ClickZetta documentation syntax...")

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

        # Test 1: Exact documentation syntax
        table_name1 = f"doc_test_1_{int(time.time())}"
        doc_sql = f"""
        CREATE TABLE {table_name1} (
            vec vector(float, 4),
            id int,
            index test_vector1_vec_idx (vec) using vector properties (
                "scalar.type" = "f32",
                "distance.function" = "l2_distance"
            )
        )
        """

        print(f"Test 1: Documentation syntax (table: {table_name1})")
        print(doc_sql)

        try:
            results, _ = engine.execute_query(doc_sql)
            print(f"✅ Doc syntax executed: {results}")

            # Check if table exists
            check_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{table_name1}'")
            print(f"Table exists: {len(check_results) > 0}")

        except Exception as e:
            print(f"❌ Doc syntax failed: {e}")

        # Test 2: Without IF NOT EXISTS
        table_name2 = f"doc_test_2_{int(time.time())}"
        no_if_sql = f"""
        CREATE TABLE {table_name2} (
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

        print(f"\nTest 2: Without IF NOT EXISTS (table: {table_name2})")
        print(no_if_sql)

        try:
            results, _ = engine.execute_query(no_if_sql)
            print(f"✅ No IF NOT EXISTS executed: {results}")

            # Check if table exists
            check_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{table_name2}'")
            print(f"Table exists: {len(check_results) > 0}")

        except Exception as e:
            print(f"❌ No IF NOT EXISTS failed: {e}")

        # Test 3: Different scalar types and distance functions
        table_name3 = f"doc_test_3_{int(time.time())}"
        different_props_sql = f"""
        CREATE TABLE {table_name3} (
            id String,
            content String,
            embedding vector(float, 16),
            index vector_idx (embedding) using vector properties (
                "scalar.type" = "f32",
                "distance.function" = "l2_distance"
            )
        )
        """

        print(f"\nTest 3: Different properties (table: {table_name3})")
        print(different_props_sql)

        try:
            results, _ = engine.execute_query(different_props_sql)
            print(f"✅ Different props executed: {results}")

            # Check if table exists
            check_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{table_name3}'")
            print(f"Table exists: {len(check_results) > 0}")

        except Exception as e:
            print(f"❌ Different props failed: {e}")

        # Test 4: Minimal vector index
        table_name4 = f"doc_test_4_{int(time.time())}"
        minimal_sql = f"""
        CREATE TABLE {table_name4} (
            id String,
            embedding vector(float, 16),
            index vector_idx (embedding) using vector
        )
        """

        print(f"\nTest 4: Minimal vector index (table: {table_name4})")
        print(minimal_sql)

        try:
            results, _ = engine.execute_query(minimal_sql)
            print(f"✅ Minimal executed: {results}")

            # Check if table exists
            check_results, _ = engine.execute_query(f"SHOW TABLES LIKE '{table_name4}'")
            print(f"Table exists: {len(check_results) > 0}")

        except Exception as e:
            print(f"❌ Minimal failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_doc_syntax()
