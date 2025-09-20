"""Direct test of ClickZetta storage services without importing other components."""

import os
import sys

# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    "/Users/liangmo/Documents/GitHub/langchain-clickzetta/libs/clickzetta/langchain_clickzetta",
)

# Import modules directly to avoid dependency issues
from engine import ClickZettaEngine  # noqa: E402
from stores import (  # noqa: E402
    ClickZettaDocumentStore,
    ClickZettaFileStore,
    ClickZettaStore,
)
from volume_store import (  # noqa: E402
    ClickZettaNamedVolumeStore,
    ClickZettaTableVolumeStore,
    ClickZettaUserVolumeStore,
)


def main():
    """Test ClickZetta storage services directly."""
    # Initialize ClickZetta engine
    engine = ClickZettaEngine(
        service=os.getenv("CLICKZETTA_SERVICE", "your-service"),
        instance=os.getenv("CLICKZETTA_INSTANCE", "your-instance"),
        workspace=os.getenv("CLICKZETTA_WORKSPACE", "your-workspace"),
        schema=os.getenv("CLICKZETTA_SCHEMA", "your-schema"),
        username=os.getenv("CLICKZETTA_USERNAME", "your-username"),
        password=os.getenv("CLICKZETTA_PASSWORD", "your-password"),
        vcluster=os.getenv("CLICKZETTA_VCLUSTER", "your-vcluster"),
    )

    print("=== Testing ClickZetta Storage Services ===\n")

    # Test 1: Basic Key-Value Store
    print("1. Testing ClickZettaStore (Key-Value)")
    print("-" * 35)

    try:
        kv_store = ClickZettaStore(engine=engine, table_name="test_kv_store")
        print("✓ ClickZettaStore initialized successfully")

        # Test storage
        test_data = [("test:key1", b"value1"), ("test:key2", b"value2")]
        kv_store.mset(test_data)
        print("✓ Stored key-value pairs")

        # Test retrieval
        values = kv_store.mget(["test:key1", "test:key2", "nonexistent"])
        print(
            f"✓ Retrieved values: {[v.decode('utf-8') if v else None for v in values]}"
        )

        # Test key listing
        keys = list(kv_store.yield_keys(prefix="test:"))
        print(f"✓ Keys with prefix 'test:': {keys}")

    except Exception as e:
        print(f"✗ ClickZettaStore Error: {e}")

    print("\n")

    # Test 2: Document Store
    print("2. Testing ClickZettaDocumentStore")
    print("-" * 30)

    try:
        doc_store = ClickZettaDocumentStore(engine=engine, table_name="test_doc_store")
        print("✓ ClickZettaDocumentStore initialized successfully")

        # Test document storage
        doc_store.store_document(
            "doc1",
            "This is a test document",
            {"category": "test", "author": "test_user"},
        )
        print("✓ Stored document with metadata")

        # Test document retrieval
        result = doc_store.get_document("doc1")
        if result:
            content, metadata = result
            print(f"✓ Retrieved document: '{content}', metadata: {metadata}")
        else:
            print("✗ Document not found")

    except Exception as e:
        print(f"✗ ClickZettaDocumentStore Error: {e}")

    print("\n")

    # Test 3: File Store
    print("3. Testing ClickZettaFileStore Functionality")
    print("-" * 40)

    test_file_store_functionality(engine)

    print("\n")

    # Test 4: Volume Store Functionality
    print("4. Testing ClickZetta Volume Store Functionality")
    print("-" * 50)

    test_volume_store_functionality(engine)

    # Cleanup
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")

    print("\n=== Storage Services Test Complete ===")


def test_volume_store_functionality(engine: ClickZettaEngine):
    """Comprehensive test of ClickZetta Volume Store functionality."""
    print("Testing Volume Store functionality with verification...")

    # Test User Volume Store
    test_user_volume_store(engine)

    # Test Table Volume Store
    test_table_volume_store(engine)

    # Test Named Volume Store
    test_named_volume_store(engine)


def test_user_volume_store(engine: ClickZettaEngine):
    """Test User Volume Store with full verification."""
    print("\n  4.1 Testing User Volume Store")
    print("  " + "-" * 30)

    try:
        # Initialize User Volume Store
        user_store = ClickZettaUserVolumeStore(
            engine=engine,
            subdirectory="test_functionality"
        )
        print("  ✓ User Volume Store initialized")

        # Test data
        test_key_prefix = "test_user_vol"
        test_data = [
            (f"{test_key_prefix}:key1", b"value1"),
            (f"{test_key_prefix}:key2", b"value2"),
            (f"{test_key_prefix}:key3", b"value3"),
        ]
        test_keys = [item[0] for item in test_data]

        # Clean up any existing test data
        try:
            user_store.mdelete(test_keys)
        except:
            pass  # Ignore cleanup errors

        # TEST 1: mset (store data)
        print("\n    Testing mset (store data)...")
        user_store.mset(test_data)
        print("    ✓ Data stored successfully")

        # TEST 2: mget (retrieve data) - verify data was actually stored
        print("\n    Testing mget (retrieve data)...")
        retrieved_values = user_store.mget(test_keys)

        # Verify all values were retrieved correctly
        assert len(retrieved_values) == len(test_keys), f"Expected {len(test_keys)} values, got {len(retrieved_values)}"

        for i, (expected_key, expected_value) in enumerate(test_data):
            retrieved_value = retrieved_values[i]
            assert retrieved_value is not None, f"Key {expected_key} not found"
            assert retrieved_value == expected_value, f"Value mismatch for {expected_key}: expected {expected_value}, got {retrieved_value}"

        print("    ✓ All data retrieved correctly")
        print(f"    ✓ Retrieved: {[v.decode('utf-8') if v else None for v in retrieved_values]}")

        # TEST 3: mget with non-existent key
        print("\n    Testing mget with non-existent key...")
        mixed_keys = test_keys + ["nonexistent_key"]
        mixed_values = user_store.mget(mixed_keys)

        # Verify existing keys still return correct values
        for i in range(len(test_keys)):
            assert mixed_values[i] == test_data[i][1], f"Existing key {test_keys[i]} value changed"

        # Verify non-existent key returns None
        assert mixed_values[-1] is None, "Non-existent key should return None"
        print("    ✓ Non-existent key correctly returns None")

        # TEST 4: yield_keys (list keys with prefix)
        print("\n    Testing yield_keys with prefix...")
        found_keys = list(user_store.yield_keys(prefix=test_key_prefix))

        # Verify all test keys are found
        assert len(found_keys) >= len(test_keys), f"Expected at least {len(test_keys)} keys, found {len(found_keys)}"

        for key in test_keys:
            assert key in found_keys, f"Key {key} not found in yield_keys result"

        print(f"    ✓ Found {len(found_keys)} keys with prefix '{test_key_prefix}'")
        print(f"    ✓ Keys: {found_keys}")

        # TEST 5: yield_keys without prefix
        print("\n    Testing yield_keys without prefix...")
        all_keys = list(user_store.yield_keys())

        # Verify our test keys are in the full list
        for key in test_keys:
            assert key in all_keys, f"Key {key} not found in full key list"

        print(f"    ✓ Found {len(all_keys)} total keys")

        # TEST 6: mdelete (delete data) - verify deletion actually works
        print("\n    Testing mdelete (delete data)...")

        # Delete first two keys
        keys_to_delete = test_keys[:2]
        remaining_key = test_keys[2]

        user_store.mdelete(keys_to_delete)
        print(f"    ✓ Deleted keys: {keys_to_delete}")

        # Verify deleted keys are really gone
        post_delete_values = user_store.mget(test_keys)

        # First two should be None (deleted)
        assert post_delete_values[0] is None, f"Key {test_keys[0]} should be deleted but still exists"
        assert post_delete_values[1] is None, f"Key {test_keys[1]} should be deleted but still exists"

        # Third should still exist
        assert post_delete_values[2] is not None, f"Key {remaining_key} should still exist"
        assert post_delete_values[2] == test_data[2][1], f"Remaining key {remaining_key} value changed"

        print("    ✓ Deletion verified - deleted keys return None")
        print("    ✓ Remaining key still accessible")

        # Verify keys are also removed from yield_keys
        remaining_keys = list(user_store.yield_keys(prefix=test_key_prefix))

        for deleted_key in keys_to_delete:
            assert deleted_key not in remaining_keys, f"Deleted key {deleted_key} still appears in yield_keys"

        assert remaining_key in remaining_keys, f"Remaining key {remaining_key} missing from yield_keys"

        print("    ✓ Deletion verified in yield_keys")

        # Clean up remaining test data
        user_store.mdelete([remaining_key])
        print("    ✓ Test data cleaned up")

    except Exception as e:
        print(f"  ✗ User Volume Store Error: {e}")
        import traceback
        traceback.print_exc()


def test_table_volume_store(engine: ClickZettaEngine):
    """Test Table Volume Store with full verification."""
    print("\n  4.2 Testing Table Volume Store")
    print("  " + "-" * 30)

    table_name = "test_table_volume_store"

    try:
        # Create test table
        try:
            engine.execute_query(f"DROP TABLE IF EXISTS {table_name}")
            engine.execute_query(f"CREATE TABLE {table_name} (id INT, data STRING)")
            print(f"  ✓ Test table '{table_name}' created")
        except Exception as e:
            print(f"  Warning: Could not create test table: {e}")
            return

        # Initialize Table Volume Store
        table_store = ClickZettaTableVolumeStore(
            engine=engine,
            table_name=table_name,
            subdirectory="test_functionality"
        )
        print("  ✓ Table Volume Store initialized")

        # Test similar to User Volume but with table-specific data
        test_key_prefix = "test_table_vol"
        test_data = [
            (f"{test_key_prefix}:table_key1", b"table_value1"),
            (f"{test_key_prefix}:table_key2", b"table_value2"),
        ]
        test_keys = [item[0] for item in test_data]

        # Store, retrieve, and verify
        table_store.mset(test_data)
        retrieved_values = table_store.mget(test_keys)

        for i, (_, expected_value) in enumerate(test_data):
            assert retrieved_values[i] == expected_value, f"Table volume value mismatch at index {i}"

        print("  ✓ Table Volume Store operations verified")

        # Cleanup
        table_store.mdelete(test_keys)

        # Drop test table
        try:
            engine.execute_query(f"DROP TABLE IF EXISTS {table_name}")
        except:
            pass

    except Exception as e:
        print(f"  ✗ Table Volume Store Error: {e}")


def test_file_store_functionality(engine: ClickZettaEngine):
    """Comprehensive test of ClickZetta File Store functionality."""
    print("Testing File Store functionality with full verification...")

    try:
        # Initialize File Store (using User Volume)
        file_store = ClickZettaFileStore(
            engine=engine,
            volume_type="user",
            subdirectory="test_file_store"
        )
        print("  ✓ ClickZettaFileStore initialized successfully")

        # Test data
        test_files = [
            ("test/document.txt", b"This is a text document", "text/plain"),
            ("test/image.bin", b"\x89PNG\x0d\x0a\x1a\x0a\x00\x00\x00\x0dIHDR\x00\x00", "image/png"),
            ("data/config.json", b'{"setting": "value"}', "application/json"),
        ]

        # Clean up any existing test data
        try:
            file_store.mdelete([item[0] for item in test_files])
        except:
            pass  # Ignore cleanup errors

        print("\n    Testing store_file and get_file...")

        # TEST 1: Store files with metadata
        for file_path, content, mime_type in test_files:
            file_store.store_file(file_path, content, mime_type)
        print("    ✓ All test files stored successfully")

        # TEST 2: Retrieve and verify files
        for file_path, expected_content, expected_mime in test_files:
            result = file_store.get_file(file_path)
            assert result is not None, f"File {file_path} not found"

            content, mime_type = result
            assert content == expected_content, f"Content mismatch for {file_path}"
            assert mime_type == expected_mime, f"MIME type mismatch for {file_path}: expected {expected_mime}, got {mime_type}"

        print("    ✓ All files retrieved correctly with correct metadata")

        # TEST 3: Test get_file with non-existent file
        result = file_store.get_file("nonexistent/file.txt")
        assert result is None, "Non-existent file should return None"
        print("    ✓ Non-existent file correctly returns None")

        # TEST 4: Test list_files functionality
        print("\n    Testing list_files...")
        all_files = file_store.list_files()

        # Verify all test files are listed
        listed_paths = [f[0] for f in all_files]
        for file_path, content, mime_type in test_files:
            assert file_path in listed_paths, f"File {file_path} not found in list_files"

            # Find the file info and verify metadata
            file_info = next(f for f in all_files if f[0] == file_path)
            _, file_size, file_mime = file_info

            assert file_size == len(content), f"File size mismatch for {file_path}"
            assert file_mime == mime_type, f"MIME type mismatch in list for {file_path}"

        print(f"    ✓ list_files found {len(all_files)} files with correct metadata")

        # TEST 5: Test list_files with prefix
        test_prefix_files = file_store.list_files(prefix="test/")
        test_prefix_paths = [f[0] for f in test_prefix_files]

        # Should find test/document.txt and test/image.bin
        assert "test/document.txt" in test_prefix_paths, "test/document.txt not found with prefix"
        assert "test/image.bin" in test_prefix_paths, "test/image.bin not found with prefix"
        assert "data/config.json" not in test_prefix_paths, "data/config.json should not match test/ prefix"

        print(f"    ✓ Prefix filtering works correctly ({len(test_prefix_files)} files with 'test/' prefix)")

        # TEST 6: Test yield_keys functionality
        print("\n    Testing yield_keys...")
        all_keys = list(file_store.yield_keys())

        # Verify all test files are in keys (should not include metadata keys)
        for file_path, _, _ in test_files:
            assert file_path in all_keys, f"File {file_path} not found in yield_keys"

        # Verify no metadata keys are returned
        metadata_keys = [key for key in all_keys if key.startswith("_metadata_")]
        assert len(metadata_keys) == 0, f"yield_keys should not return metadata keys, but found: {metadata_keys}"

        print(f"    ✓ yield_keys returned {len(all_keys)} keys (no metadata keys)")

        # TEST 7: Test yield_keys with prefix
        test_keys = list(file_store.yield_keys(prefix="test/"))
        assert "test/document.txt" in test_keys, "test/document.txt not found in yield_keys with prefix"
        assert "test/image.bin" in test_keys, "test/image.bin not found in yield_keys with prefix"
        assert "data/config.json" not in test_keys, "data/config.json should not match test/ prefix in yield_keys"

        print(f"    ✓ yield_keys prefix filtering works correctly ({len(test_keys)} keys)")

        # TEST 8: Test mget functionality (BaseStore interface)
        print("\n    Testing mget (BaseStore interface)...")
        file_paths = [item[0] for item in test_files]
        mget_results = file_store.mget(file_paths)

        for i, (_, expected_content, _) in enumerate(test_files):
            assert mget_results[i] == expected_content, f"mget content mismatch at index {i}"

        print("    ✓ mget returns correct file contents")

        # TEST 9: Test mset functionality (BaseStore interface)
        print("\n    Testing mset (BaseStore interface)...")
        new_test_data = [
            ("bulk/file1.txt", b"Bulk content 1"),
            ("bulk/file2.txt", b"Bulk content 2"),
        ]

        file_store.mset(new_test_data)

        # Verify the files were stored (note: mset doesn't store metadata)
        bulk_results = file_store.mget([item[0] for item in new_test_data])
        for i, (_, expected_content) in enumerate(new_test_data):
            assert bulk_results[i] == expected_content, f"mset/mget content mismatch at index {i}"

        print("    ✓ mset stores files correctly")

        # TEST 10: Test mdelete - verify files and metadata are actually deleted
        print("\n    Testing mdelete (delete verification)...")

        # Delete first two test files
        files_to_delete = [test_files[0][0], test_files[1][0]]  # test/document.txt, test/image.bin
        remaining_file = test_files[2][0]  # data/config.json

        file_store.mdelete(files_to_delete)
        print(f"    ✓ Deleted files: {files_to_delete}")

        # Verify deleted files are really gone
        for deleted_file in files_to_delete:
            result = file_store.get_file(deleted_file)
            assert result is None, f"Deleted file {deleted_file} still accessible via get_file"

        # Verify remaining file still exists
        remaining_result = file_store.get_file(remaining_file)
        assert remaining_result is not None, f"Remaining file {remaining_file} should still exist"

        print("    ✓ Deletion verified - deleted files return None, remaining file accessible")

        # Verify files are removed from list_files
        post_delete_files = file_store.list_files()
        post_delete_paths = [f[0] for f in post_delete_files]

        for deleted_file in files_to_delete:
            assert deleted_file not in post_delete_paths, f"Deleted file {deleted_file} still appears in list_files"

        assert remaining_file in post_delete_paths, f"Remaining file {remaining_file} missing from list_files"

        print("    ✓ Deletion verified in list_files")

        # Verify files are removed from yield_keys
        post_delete_keys = list(file_store.yield_keys())

        for deleted_file in files_to_delete:
            assert deleted_file not in post_delete_keys, f"Deleted file {deleted_file} still appears in yield_keys"

        assert remaining_file in post_delete_keys, f"Remaining file {remaining_file} missing from yield_keys"

        print("    ✓ Deletion verified in yield_keys")

        # TEST 11: Verify metadata cleanup
        print("\n    Testing metadata cleanup...")

        # Check if metadata keys for deleted files are also gone
        # (This tests the internal volume_store directly)
        all_volume_keys = list(file_store.volume_store.yield_keys())

        for deleted_file in files_to_delete:
            metadata_key = f"_metadata_{deleted_file}"
            assert metadata_key not in all_volume_keys, f"Metadata key {metadata_key} still exists after deletion"

        # Verify remaining file's metadata still exists
        remaining_metadata_key = f"_metadata_{remaining_file}"
        assert remaining_metadata_key in all_volume_keys, f"Metadata key {remaining_metadata_key} missing for remaining file"

        print("    ✓ Metadata cleanup verified - deleted file metadata removed, remaining file metadata preserved")

        # Clean up all test data
        all_test_files = [item[0] for item in test_files] + [item[0] for item in new_test_data]
        file_store.mdelete(all_test_files)
        print("    ✓ All test data cleaned up")

        print("\n  ✓ ClickZettaFileStore all functionality tests passed")

    except Exception as e:
        print(f"  ✗ ClickZettaFileStore Error: {e}")
        import traceback
        traceback.print_exc()


def test_named_volume_store(engine: ClickZettaEngine):
    """Test Named Volume Store with full verification."""
    print("\n  4.3 Testing Named Volume Store")
    print("  " + "-" * 30)

    volume_name = "test_named_volume_store"

    try:
        # Create test named volume
        try:
            engine.execute_query(f"DROP VOLUME IF EXISTS {volume_name}")
            engine.execute_query(f"""
                CREATE VOLUME {volume_name}
                DIRECTORY = (
                    enable = true,
                    auto_refresh = true
                )
                RECURSIVE = true
            """)
            print(f"  ✓ Test named volume '{volume_name}' created")
        except Exception as e:
            print(f"  Warning: Could not create test named volume: {e}")
            print("  Skipping Named Volume Store test")
            return

        # Initialize Named Volume Store
        named_store = ClickZettaNamedVolumeStore(
            engine=engine,
            volume_name=volume_name,
            subdirectory="test_functionality"
        )
        print("  ✓ Named Volume Store initialized")

        # Test similar to User Volume but with named volume-specific data
        test_key_prefix = "test_named_vol"
        test_data = [
            (f"{test_key_prefix}:named_key1", b"named_value1"),
            (f"{test_key_prefix}:named_key2", b"named_value2"),
        ]
        test_keys = [item[0] for item in test_data]

        # Store, retrieve, and verify
        named_store.mset(test_data)
        # Named Volume may need time for eventual consistency
        import time
        time.sleep(1)  # Wait for eventual consistency
        retrieved_values = named_store.mget(test_keys)

        for i, (_, expected_value) in enumerate(test_data):
            assert retrieved_values[i] == expected_value, f"Named volume value mismatch at index {i}"

        print("  ✓ Named Volume Store operations verified")

        # Cleanup
        named_store.mdelete(test_keys)

        # Drop test volume
        try:
            engine.execute_query(f"DROP VOLUME IF EXISTS {volume_name}")
        except:
            pass

    except Exception as e:
        print(f"  ✗ Named Volume Store Error: {e}")


if __name__ == "__main__":
    main()
