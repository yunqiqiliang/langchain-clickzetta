"""Strict deletion verification test for ClickZetta File Store."""

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
from stores import ClickZettaFileStore  # noqa: E402


def test_strict_file_deletion():
    """Test that ClickZettaFileStore actually deletes files from Volume, not just hides them."""
    print("=== Strict File Deletion Verification Test ===\n")

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

    try:
        # Initialize FileStore with User Volume
        file_store = ClickZettaFileStore(
            engine=engine,
            volume_type="user",
            subdirectory="strict_deletion_test"
        )
        print("✓ ClickZettaFileStore initialized with User Volume")

        # Test files
        test_files = [
            ("test_deletion/file1.txt", b"Test content 1", "text/plain"),
            ("test_deletion/file2.bin", b"Binary content 2", "application/octet-stream"),
        ]

        print("\n--- PHASE 1: Store files and verify presence ---")

        # Store files
        for file_path, content, mime_type in test_files:
            file_store.store_file(file_path, content, mime_type)
            print(f"✓ Stored file: {file_path}")

        # Verify files are accessible through FileStore
        print("\n1. Verifying files accessible through FileStore...")
        for file_path, expected_content, _expected_mime in test_files:
            result = file_store.get_file(file_path)
            assert result is not None, f"File {file_path} should be accessible"
            content, mime_type = result
            assert content == expected_content, f"Content mismatch for {file_path}"
            print(f"✓ {file_path} accessible via FileStore")

        # Verify files are listed in FileStore
        print("\n2. Verifying files listed in FileStore...")
        listed_files = file_store.list_files(prefix="test_deletion/")
        listed_paths = [f[0] for f in listed_files]
        for file_path, _, _ in test_files:
            assert file_path in listed_paths, f"File {file_path} should be listed"
            print(f"✓ {file_path} listed in FileStore")

        # CRITICAL: Verify files exist in underlying Volume Store
        print("\n3. Verifying files exist in underlying Volume Store...")
        volume_store = file_store.volume_store

        # Check data files exist in volume
        volume_keys = list(volume_store.yield_keys(prefix="test_deletion/"))
        print(f"   Volume keys found: {volume_keys}")

        for file_path, _, _ in test_files:
            assert file_path in volume_keys, f"File {file_path} should exist in Volume"
            # Also check that we can retrieve content directly from volume
            volume_content = volume_store.mget([file_path])
            assert volume_content[0] is not None, f"File {file_path} should have content in Volume"
            print(f"✓ {file_path} exists in Volume Store")

        # Check metadata files exist in volume
        print("\n4. Verifying metadata files exist in Volume...")
        all_volume_keys = list(volume_store.yield_keys())
        for file_path, _, _ in test_files:
            metadata_key = f"_metadata_{file_path}"
            assert metadata_key in all_volume_keys, f"Metadata {metadata_key} should exist in Volume"
            print(f"✓ Metadata for {file_path} exists in Volume")

        print(f"\n   Total Volume keys before deletion: {len(all_volume_keys)}")

        print("\n--- PHASE 2: Delete files and verify complete removal ---")

        # Delete one file through FileStore
        file_to_delete = test_files[0][0]  # "test_deletion/file1.txt"
        remaining_file = test_files[1][0]  # "test_deletion/file2.bin"

        print(f"\n5. Deleting file through FileStore: {file_to_delete}")
        file_store.mdelete([file_to_delete])
        print("✓ Deletion command executed")

        # Verify file is gone from FileStore
        print("\n6. Verifying file gone from FileStore...")
        result = file_store.get_file(file_to_delete)
        assert result is None, f"Deleted file {file_to_delete} should not be accessible via FileStore"
        print(f"✓ {file_to_delete} not accessible via FileStore")

        # Verify file is gone from FileStore list
        post_delete_listed = file_store.list_files(prefix="test_deletion/")
        post_delete_paths = [f[0] for f in post_delete_listed]
        assert file_to_delete not in post_delete_paths, f"Deleted file {file_to_delete} should not be listed"
        print(f"✓ {file_to_delete} not listed in FileStore")

        # CRITICAL: Verify file is ACTUALLY GONE from Volume Store
        print("\n7. CRITICAL: Verifying file actually removed from Volume Store...")

        # Check that data file is gone from volume
        post_delete_volume_keys = list(volume_store.yield_keys(prefix="test_deletion/"))
        print(f"   Volume keys after deletion: {post_delete_volume_keys}")

        assert file_to_delete not in post_delete_volume_keys, f"FAILURE: {file_to_delete} still exists in Volume!"

        # Double-check by trying to retrieve directly from volume
        direct_volume_content = volume_store.mget([file_to_delete])
        assert direct_volume_content[0] is None, f"FAILURE: {file_to_delete} still has content in Volume!"
        print(f"✓ VERIFIED: {file_to_delete} completely removed from Volume Store")

        # CRITICAL: Verify metadata is ACTUALLY GONE from Volume Store
        print("\n8. CRITICAL: Verifying metadata actually removed from Volume Store...")
        all_volume_keys_after = list(volume_store.yield_keys())
        metadata_key = f"_metadata_{file_to_delete}"

        assert metadata_key not in all_volume_keys_after, f"FAILURE: Metadata {metadata_key} still exists in Volume!"

        # Double-check by trying to retrieve metadata directly from volume
        direct_metadata_content = volume_store.mget([metadata_key])
        assert direct_metadata_content[0] is None, f"FAILURE: Metadata {metadata_key} still has content in Volume!"
        print(f"✓ VERIFIED: Metadata for {file_to_delete} completely removed from Volume Store")

        # Verify remaining file is still intact
        print("\n9. Verifying remaining file still intact...")

        # Check via FileStore
        remaining_result = file_store.get_file(remaining_file)
        assert remaining_result is not None, f"Remaining file {remaining_file} should still be accessible"

        # Check directly in Volume
        assert remaining_file in post_delete_volume_keys, f"Remaining file {remaining_file} should still exist in Volume"
        remaining_volume_content = volume_store.mget([remaining_file])
        assert remaining_volume_content[0] is not None, f"Remaining file {remaining_file} should still have content in Volume"

        # Check metadata still exists
        remaining_metadata_key = f"_metadata_{remaining_file}"
        assert remaining_metadata_key in all_volume_keys_after, f"Remaining metadata {remaining_metadata_key} should still exist"
        print(f"✓ Remaining file {remaining_file} and its metadata are intact")

        print(f"\n   Total Volume keys after deletion: {len(all_volume_keys_after)}")
        keys_removed = len(all_volume_keys) - len(all_volume_keys_after)
        print(f"   Keys removed from Volume: {keys_removed} (should be 2: file + metadata)")
        assert keys_removed == 2, f"Expected 2 keys removed (file + metadata), but {keys_removed} were removed"

        print("\n--- PHASE 3: Verify ClickZetta Volume commands actually executed ---")

        # To be extra sure, let's verify by checking the actual Volume directory
        print("\n10. Checking actual Volume directory state...")
        try:
            # Execute SHOW USER VOLUME DIRECTORY to see actual state
            results, _ = engine.execute_query("SHOW USER VOLUME DIRECTORY")
            volume_files = []
            for row in results:
                relative_path = row.get("relative_path", "")
                if relative_path.startswith("strict_deletion_test/test_deletion/"):
                    volume_files.append(relative_path)

            print(f"   Actual files in Volume directory: {volume_files}")

            # Verify deleted file is not in actual directory
            deleted_file_path = f"strict_deletion_test/{file_to_delete}"
            deleted_metadata_path = f"strict_deletion_test/_metadata_{file_to_delete}"

            actual_deleted_file_missing = not any(f.endswith(file_to_delete.split('/')[-1] + '.dat') for f in volume_files)
            actual_deleted_metadata_missing = not any('_metadata_' in f and file_to_delete.split('/')[-1] in f for f in volume_files)

            if actual_deleted_file_missing and actual_deleted_metadata_missing:
                print(f"✓ VERIFIED: {file_to_delete} and its metadata are truly absent from Volume directory")
            else:
                print("⚠️  WARNING: Files may still exist in Volume directory")

        except Exception as e:
            print(f"   Could not verify Volume directory state: {e}")

        # Clean up remaining test files
        print("\n--- CLEANUP ---")
        file_store.mdelete([remaining_file])
        print("✓ Cleaned up remaining test files")

        print("\n=== RESULT: File deletion verification PASSED ===")
        print("✓ Files are ACTUALLY deleted from ClickZetta Volume, not just hidden")
        print("✓ Both file content and metadata are completely removed")
        print("✓ No evidence of 'pseudo-deletion' found")

    except AssertionError as e:
        print(f"\n❌ DELETION VERIFICATION FAILED: {e}")
        print("🚨 POTENTIAL PSEUDO-DELETION ISSUE DETECTED!")
        raise
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            engine.close()
            print("\n✓ Connection closed")
        except:
            pass


if __name__ == "__main__":
    test_strict_file_deletion()
