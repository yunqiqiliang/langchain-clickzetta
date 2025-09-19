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
    print("3. Testing ClickZettaFileStore")
    print("-" * 25)

    try:
        file_store = ClickZettaFileStore(engine=engine, table_name="test_file_store")
        print("✓ ClickZettaFileStore initialized successfully")

        # Test file storage
        test_content = b"This is test binary content"
        file_store.store_file("test/file.bin", test_content, "application/octet-stream")
        print("✓ Stored binary file")

        # Test file retrieval
        result = file_store.get_file("test/file.bin")
        if result:
            content, mime_type = result
            print(f"✓ Retrieved file: {len(content)} bytes, type: {mime_type}")
            print(f"  Content: {content.decode('utf-8')}")
        else:
            print("✗ File not found")

        # Test file listing
        files = file_store.list_files()
        print(f"✓ Listed files: {files}")

    except Exception as e:
        print(f"✗ ClickZettaFileStore Error: {e}")

    # Cleanup
    try:
        engine.close()
        print("\n✓ ClickZetta connection closed")
    except Exception as e:
        print(f"\nError closing connection: {e}")

    print("\n=== Storage Services Test Complete ===")


if __name__ == "__main__":
    main()
