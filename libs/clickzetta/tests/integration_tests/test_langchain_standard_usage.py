"""Test LangChain standard usage patterns compatibility."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.stores import BaseStore

from langchain_clickzetta import (
    ClickZettaDocumentStore,
    ClickZettaEngine,
    ClickZettaFileStore,
    ClickZettaStore,
    ClickZettaUserVolumeStore,
)


def test_standard_synchronous_usage():
    """Test standard LangChain synchronous store usage patterns."""
    print("=== Testing Standard LangChain Synchronous Usage ===\n")

    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "test"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "test"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "test"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "test"),
        username=os.getenv("CLICKZETTA_USERNAME", "test"),
        password=os.getenv("CLICKZETTA_PASSWORD", "test"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "test"),
    )

    # Test 1: Basic Store Interface
    print("1. Testing Basic Store Interface:")
    store: BaseStore = ClickZettaStore(engine=engine, table_name="test_standard_usage")

    # Standard LangChain usage pattern
    try:
        # Set multiple key-value pairs
        store.mset([("key1", b"value1"), ("key2", b"value2"), ("key3", b"value3")])
        print("  ✓ mset() with multiple pairs")

        # Get multiple values
        values = store.mget(["key1", "key2", "nonexistent"])
        expected = [b"value1", b"value2", None]
        if values == expected:
            print("  ✓ mget() returns correct values and None for missing keys")
        else:
            print(f"  ✗ mget() returned {values}, expected {expected}")

        # Iterate over keys with prefix
        keys_with_prefix = list(store.yield_keys(prefix="key"))
        if set(keys_with_prefix) == {"key1", "key2", "key3"}:
            print("  ✓ yield_keys() with prefix works correctly")
        else:
            print(
                f"  ✗ yield_keys() returned {keys_with_prefix}, expected ['key1', 'key2', 'key3']"
            )

        # Delete keys
        store.mdelete(["key1", "key3"])
        remaining_values = store.mget(["key1", "key2", "key3"])
        if remaining_values == [None, b"value2", None]:
            print("  ✓ mdelete() works correctly")
        else:
            print(f"  ✗ mdelete() failed, got {remaining_values}")

    except Exception as e:
        print(f"  ✗ Basic store interface test failed: {e}")

    print()

    # Test 2: Type annotations compatibility
    print("2. Testing Type Annotations Compatibility:")
    try:
        # Test with string keys and bytes values (standard LangChain pattern)
        store.mset([("str_key", b"bytes_value")])
        result = store.mget(["str_key"])
        if result == [b"bytes_value"]:
            print("  ✓ String keys with bytes values work correctly")

        # Test empty operations
        store.mset([])  # Empty set
        empty_results = store.mget([])  # Empty get
        if empty_results == []:
            print("  ✓ Empty operations work correctly")

        store.mdelete([])  # Empty delete
        print("  ✓ Empty delete works correctly")

    except Exception as e:
        print(f"  ✗ Type annotations test failed: {e}")

    print()

    # Test 3: Different Store Types
    print("3. Testing Different Store Types:")

    store_types = [
        ("ClickZettaStore", ClickZettaStore, {"table_name": "test_standard_kv"}),
        (
            "ClickZettaDocumentStore",
            ClickZettaDocumentStore,
            {"table_name": "test_standard_doc"},
        ),
        (
            "ClickZettaFileStore",
            ClickZettaFileStore,
            {"volume_type": "user", "subdirectory": "test_standard_file"},
        ),
        (
            "ClickZettaUserVolumeStore",
            ClickZettaUserVolumeStore,
            {"subdirectory": "test_standard_volume"},
        ),
    ]

    for store_name, store_class, kwargs in store_types:
        try:
            test_store: BaseStore = store_class(engine=engine, **kwargs)

            # Standard LangChain operations
            test_store.mset([("test", b"data")])
            result = test_store.mget(["test"])
            if result == [b"data"]:
                print(f"  ✓ {store_name} works with standard LangChain interface")
            else:
                print(f"  ✗ {store_name} failed standard test")

        except Exception as e:
            print(f"  ✗ {store_name} failed: {e}")

    print()


async def test_async_interface():
    """Test async interface compatibility."""
    print("=== Testing Async Interface Compatibility ===\n")

    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "test"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "test"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "test"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "test"),
        username=os.getenv("CLICKZETTA_USERNAME", "test"),
        password=os.getenv("CLICKZETTA_PASSWORD", "test"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "test"),
    )

    store: BaseStore = ClickZettaStore(engine=engine, table_name="test_async_usage")

    # Check if async methods are available
    async_methods = ["amget", "amset", "amdelete", "ayield_keys"]
    available_async = []

    for method in async_methods:
        if hasattr(store, method):
            available_async.append(method)

    if available_async:
        print(f"Available async methods: {available_async}")

        # Test async methods if they exist
        try:
            if hasattr(store, "amset"):
                await store.amset([("async_key", b"async_value")])
                print("  ✓ amset() works")

            if hasattr(store, "amget"):
                result = await store.amget(["async_key"])
                if result == [b"async_value"]:
                    print("  ✓ amget() works")
                else:
                    print(f"  ✗ amget() returned {result}")

            if hasattr(store, "ayield_keys"):
                async_keys = []
                async for key in store.ayield_keys():
                    async_keys.append(key)
                    if len(async_keys) >= 5:  # Limit to avoid infinite loop
                        break
                print(f"  ✓ ayield_keys() works, found {len(async_keys)} keys")

            if hasattr(store, "amdelete"):
                await store.amdelete(["async_key"])
                print("  ✓ amdelete() works")

        except Exception as e:
            print(f"  ✗ Async methods test failed: {e}")
    else:
        print(
            "No async methods implemented (this is OK - async is optional in BaseStore)"
        )

    print()


def test_langchain_integration_patterns():
    """Test common LangChain integration patterns."""
    print("=== Testing LangChain Integration Patterns ===\n")

    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "test"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "test"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "test"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "test"),
        username=os.getenv("CLICKZETTA_USERNAME", "test"),
        password=os.getenv("CLICKZETTA_PASSWORD", "test"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "test"),
    )

    # Pattern 1: Memory/Cache pattern
    print("1. Memory/Cache Pattern:")
    try:
        cache_store: BaseStore = ClickZettaStore(
            engine=engine, table_name="langchain_cache"
        )

        # Simulate typical caching usage
        cache_key = "computation_result_123"
        cached_data = (
            b'{"result": "expensive_computation_output", "timestamp": "2024-01-01"}'
        )

        # Store in cache
        cache_store.mset([(cache_key, cached_data)])

        # Retrieve from cache
        cached_result = cache_store.mget([cache_key])
        if cached_result[0] == cached_data:
            print("  ✓ Cache pattern works correctly")
        else:
            print("  ✗ Cache pattern failed")

    except Exception as e:
        print(f"  ✗ Cache pattern failed: {e}")

    # Pattern 2: Document Storage pattern
    print("\n2. Document Storage Pattern:")
    try:
        doc_store = ClickZettaDocumentStore(
            engine=engine, table_name="langchain_documents"
        )

        # Store documents with metadata (typical LangChain usage)
        doc_id = "doc_001"
        content = "This is a sample document for LangChain processing."
        metadata = {"source": "test", "type": "text", "processed": True}

        doc_store.store_document(doc_id, content, metadata)

        # Retrieve document
        result = doc_store.get_document(doc_id)
        if result and result[0] == content and result[1] == metadata:
            print("  ✓ Document storage pattern works correctly")
        else:
            print(f"  ✗ Document storage pattern failed: {result}")

    except Exception as e:
        print(f"  ✗ Document storage pattern failed: {e}")

    # Pattern 3: File/Binary Storage pattern
    print("\n3. File/Binary Storage Pattern:")
    try:
        file_store = ClickZettaFileStore(
            engine=engine, volume_type="user", subdirectory="langchain_files"
        )

        # Store binary file (typical for embeddings, models, etc.)
        file_path = "embeddings/model.bin"
        binary_data = b"\x00\x01\x02\x03" * 100  # Simulated binary data
        mime_type = "application/octet-stream"

        file_store.store_file(file_path, binary_data, mime_type)

        # Retrieve file
        result = file_store.get_file(file_path)
        if result and result[0] == binary_data and result[1] == mime_type:
            print("  ✓ File storage pattern works correctly")
        else:
            print("  ✗ File storage pattern failed")

    except Exception as e:
        print(f"  ✗ File storage pattern failed: {e}")

    print("\n=== Integration Patterns Test Complete ===")


if __name__ == "__main__":
    # Run synchronous tests
    test_standard_synchronous_usage()
    test_langchain_integration_patterns()

    # Run async tests
    print("Running async tests...")
    asyncio.run(test_async_interface())

    print("\n=== All LangChain Standard Usage Tests Complete ===")
