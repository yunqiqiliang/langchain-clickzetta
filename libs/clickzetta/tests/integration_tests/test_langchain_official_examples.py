"""Test compatibility with LangChain official usage examples."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.stores import BaseStore

from langchain_clickzetta import ClickZettaEngine, ClickZettaStore


def test_langchain_official_examples():
    """Test examples similar to LangChain official documentation."""
    print("=== Testing LangChain Official Usage Examples ===\n")

    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "test"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "test"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "test"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "test"),
        username=os.getenv("CLICKZETTA_USERNAME", "test"),
        password=os.getenv("CLICKZETTA_PASSWORD", "test"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "test"),
    )

    # Example 1: Basic Store Usage (similar to LangChain docs)
    print("1. Basic Store Usage Example:")
    try:
        # Create a store instance
        store = ClickZettaStore(engine=engine, table_name="example_store")

        # Set some values
        store.mset([("key1", b"value1"), ("key2", b"value2")])

        # Get values
        values = store.mget(["key1", "key2", "key3"])
        print(f"  Retrieved values: {values}")  # Should be [b'value1', b'value2', None]

        # Iterate over keys
        print("  Keys in store:")
        for key in store.yield_keys():
            print(f"    {key}")

        # Delete a key
        store.mdelete(["key1"])

        # Check deletion
        remaining = store.mget(["key1", "key2"])
        print(f"  After deletion: {remaining}")  # Should be [None, b'value2']

        print("  ✓ Basic usage example works correctly")

    except Exception as e:
        print(f"  ✗ Basic usage example failed: {e}")

    print()

    # Example 2: Store as a cache/memory component
    print("2. Store as Cache/Memory Component:")
    try:
        cache: BaseStore = ClickZettaStore(engine=engine, table_name="memory_cache")

        # Simulate LangChain memory usage
        session_id = "user_123"
        conversation_history = b'{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}'

        # Store conversation
        cache.mset([(f"conversation:{session_id}", conversation_history)])

        # Retrieve conversation
        retrieved = cache.mget([f"conversation:{session_id}"])
        if retrieved[0] == conversation_history:
            print("  ✓ Cache/memory usage works correctly")
        else:
            print("  ✗ Cache/memory usage failed")

    except Exception as e:
        print(f"  ✗ Cache/memory example failed: {e}")

    print()

    # Example 3: Prefix-based key retrieval
    print("3. Prefix-based Key Retrieval:")
    try:
        store = ClickZettaStore(engine=engine, table_name="prefix_test")

        # Set up test data with different prefixes
        test_data = [
            ("user:alice:profile", b"alice_data"),
            ("user:bob:profile", b"bob_data"),
            ("user:alice:settings", b"alice_settings"),
            ("session:123", b"session_data"),
            ("cache:expensive_op", b"cached_result"),
        ]

        store.mset(test_data)

        # Test prefix filtering
        user_keys = list(store.yield_keys(prefix="user:"))
        alice_keys = list(store.yield_keys(prefix="user:alice:"))

        print(f"  User keys: {sorted(user_keys)}")
        print(f"  Alice keys: {sorted(alice_keys)}")

        if len(user_keys) == 3 and len(alice_keys) == 2:
            print("  ✓ Prefix-based retrieval works correctly")
        else:
            print(
                f"  ✗ Prefix-based retrieval failed: got {len(user_keys)} user keys, {len(alice_keys)} alice keys"
            )

    except Exception as e:
        print(f"  ✗ Prefix-based retrieval failed: {e}")

    print()

    # Example 4: Polymorphic usage (store as BaseStore)
    print("4. Polymorphic Usage (Store as BaseStore):")
    try:

        def use_any_store(store: BaseStore, prefix: str = "test"):
            """Function that works with any BaseStore implementation."""
            # This simulates how LangChain components use stores
            store.mset([(f"{prefix}:key1", b"data1"), (f"{prefix}:key2", b"data2")])
            values = store.mget([f"{prefix}:key1", f"{prefix}:key2"])
            keys = list(store.yield_keys(prefix=f"{prefix}:"))
            store.mdelete([f"{prefix}:key1"])
            remaining = list(store.yield_keys(prefix=f"{prefix}:"))
            return len(values), len(keys), len(remaining)

        # Test with our ClickZetta store
        clickzetta_store = ClickZettaStore(engine=engine, table_name="polymorphic_test")
        result = use_any_store(clickzetta_store, "polymorphic")

        if result == (2, 2, 1):  # 2 values retrieved, 2 keys initially, 1 key remaining
            print("  ✓ Polymorphic usage works correctly")
        else:
            print(f"  ✗ Polymorphic usage failed: got {result}")

    except Exception as e:
        print(f"  ✗ Polymorphic usage failed: {e}")

    print()

    # Example 5: Error handling (LangChain expects graceful handling)
    print("5. Error Handling:")
    try:
        store = ClickZettaStore(engine=engine, table_name="error_test")

        # Test getting non-existent keys (should return None, not error)
        non_existent = store.mget(["does_not_exist", "also_missing"])
        if non_existent == [None, None]:
            print("  ✓ Non-existent keys return None correctly")
        else:
            print(f"  ✗ Non-existent keys handling failed: {non_existent}")

        # Test deleting non-existent keys (should not error)
        store.mdelete(["does_not_exist"])
        print("  ✓ Deleting non-existent keys doesn't error")

        # Test empty operations
        store.mset([])
        empty_get = store.mget([])
        store.mdelete([])
        if empty_get == []:
            print("  ✓ Empty operations work correctly")

    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")

    print("\n=== LangChain Official Examples Test Complete ===")


if __name__ == "__main__":
    test_langchain_official_examples()
